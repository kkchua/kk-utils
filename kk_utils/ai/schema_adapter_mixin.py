"""
kk_utils/ai/schema_adapter_mixin.py

Mixin for AI adapters that load their system prompt and output schema from
external files rather than hardcoding them in Python.

File layout (relative to the adapter's _ADAPTER_DIR):
  master_prompt.txt      — system prompt; {{var}} tokens are substituted at
                           call time from the kwargs passed to build_prompt()
  {schema_name}.json     — JSON Schema that is compiled into a Pydantic model
                           at load time via build_output_type_schema()

JSON Schema conventions (superset of conductor's format):
  Standard JSON Schema fields used by the type builder:
    type         — "string" | "integer" | "number" | "boolean" | "array" | "object"
    enum         — string enum values  →  Literal["a", "b", ...]
    items        — for "array" fields; may be a primitive type or object
    properties   — for nested object items
    required     — list of required field names (others become Optional)
    description  — forwarded to Field(description=...)
    minLength    — field_validator: min weighted (CJK-aware) char count
    maxLength    — field_validator: max weighted (CJK-aware) char count

  kk-utils extension keys (prefixed with _):
    _array_field     — name of the primary array field at the root level
                       (used to build an OUTPUT SCHEMA injection block)
    _filename_field  — root field whose value is used as the output filename
    _csv_columns     — [{"header": ..., "field": ...}, ...] CSV column map
    _validation_rules — conductor-style per-mode array count + unique checks

Usage (schema-driven adapter):

    from kk_utils.ai.schema_adapter_mixin import SchemaAdapterMixin
    from kk_utils.ai.base_ai_adapter import BaseAIAdapter

    class MyAdapter(SchemaAdapterMixin, BaseAIAdapter):
        _ADAPTER_DIR = Path(__file__).parent

        def user_message(self, question: str, **kwargs) -> str:
            return question

    # Load once (e.g. at startup or on first use):
    adapter = MyAdapter()
    adapter.load_schema()                       # reads master_prompt.txt + schema.json
    # Then at call time:
    system_prompt = adapter.build_prompt(context="...", topic="...")
    user_msg      = adapter.user_message(question="What is X?")
    output_type   = adapter.output_type_schema()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Literal, Optional, Type

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# JSON Schema primitive type → Python type
_JSON_TYPE_MAP: dict[str, type] = {
    "string":  str,
    "integer": int,
    "number":  float,
    "boolean": bool,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weighted_len(text: str) -> int:
    """CJK-aware length — each CJK character counts as 2."""
    return sum(2 if ord(c) > 0x2E7F else 1 for c in text)


def _python_type(json_type: str) -> type:
    return _JSON_TYPE_MAP.get(json_type, str)


# ---------------------------------------------------------------------------
# Dynamic Pydantic model builder (generalised from conductor)
# ---------------------------------------------------------------------------

def build_output_type_schema(schema: dict) -> Type[BaseModel]:
    """
    Compile a JSON Schema dict into a Pydantic model class.

    Supports:
      • Primitive fields: string / integer / number / boolean
      • Enum strings    → Literal["a", "b", ...]
      • Array of primitives → List[str | int | float | bool]
      • Array of objects    → List[NestedModel]  (conductor batch pattern)
      • Optional fields     → any field not listed in schema["required"]
      • minLength/maxLength → field_validator with CJK-aware weighted length

    The returned class is always named "OutputSchema"; callers can rename it
    via the __name__ attribute if needed.
    """
    props    = schema.get("properties", {})
    required = set(schema.get("required", props.keys()))

    annotations: dict[str, Any] = {}
    validators:  dict[str, Any] = {}
    namespace:   dict[str, Any] = {}   # extra names for model_rebuild

    for field_name, field_def in props.items():
        ftype = field_def.get("type", "string")

        # ── Resolve Python annotation ──────────────────────────────────
        if ftype == "array":
            items_def  = field_def.get("items", {})
            items_type = items_def.get("type", "string")
            if items_type == "object":
                nested = _build_nested_model(field_name, items_def, validators, namespace)
                py_type: Any = List[nested]          # type: ignore[valid-type]
            else:
                py_type = List[_python_type(items_type)]
        elif "enum" in field_def:
            # Literal requires a tuple literal — build it via type()
            enum_vals = tuple(field_def["enum"])
            py_type   = Literal[enum_vals]           # type: ignore[valid-type]
        else:
            py_type = _python_type(ftype)

        if field_name not in required:
            py_type = Optional[py_type]              # type: ignore[assignment]

        annotations[field_name] = py_type

        # ── String length validators ───────────────────────────────────
        mn = field_def.get("minLength")
        mx = field_def.get("maxLength")
        if (mn is not None or mx is not None) and ftype == "string":
            mn_val = int(mn) if mn is not None else None
            mx_val = int(mx) if mx is not None else None
            validators[f"_validate_{field_name}_length"] = _make_length_validator(
                field_name, mn_val, mx_val
            )

    # ── Root model ────────────────────────────────────────────────────
    OutputSchema: Type[BaseModel] = type(
        "OutputSchema",
        (BaseModel,),
        {
            "__annotations__": annotations,
            "model_config": {"arbitrary_types_allowed": True},
            **validators,
        },
    )
    OutputSchema.model_rebuild(_types_namespace=namespace)
    return OutputSchema


def _build_nested_model(
    parent_field: str,
    items_def: dict,
    parent_validators: dict,
    namespace: dict,
) -> Type[BaseModel]:
    """Build a nested Pydantic model for array-of-objects fields."""
    item_props     = items_def.get("properties", {})
    item_required  = set(items_def.get("required", item_props.keys()))
    item_ann: dict[str, Any] = {}
    item_val: dict[str, Any] = {}

    for fname, fdef in item_props.items():
        ftype = fdef.get("type", "string")
        if "enum" in fdef:
            py_t: Any = Literal[tuple(fdef["enum"])]   # type: ignore[valid-type]
        elif ftype == "array":
            inner = fdef.get("items", {}).get("type", "string")
            py_t  = List[_python_type(inner)]
        else:
            py_t = _python_type(ftype)

        if fname not in item_required:
            py_t = Optional[py_t]                       # type: ignore[assignment]
        item_ann[fname] = py_t

        mn = fdef.get("minLength")
        mx = fdef.get("maxLength")
        if (mn is not None or mx is not None) and ftype == "string":
            item_val[f"_validate_{fname}_length"] = _make_length_validator(
                fname, int(mn) if mn is not None else None,
                int(mx) if mx is not None else None,
            )

    NestedModel: Type[BaseModel] = type(
        f"{parent_field.capitalize()}Item",
        (BaseModel,),
        {"__annotations__": item_ann, **item_val},
    )
    NestedModel.model_rebuild()
    # Register in caller's namespace for model_rebuild resolution
    namespace[NestedModel.__name__] = NestedModel
    return NestedModel


def _make_length_validator(fname: str, mn: int | None, mx: int | None):
    @field_validator(fname)
    @classmethod
    def _v(cls, v: str) -> str:
        wl = _weighted_len(v)
        if mn is not None and wl < mn:
            raise ValueError(f"{fname} too short ({wl} weighted chars, min={mn})")
        if mx is not None and wl > mx:
            raise ValueError(f"{fname} too long ({wl} weighted chars, max={mx})")
        return v
    _v.__name__ = f"validate_{fname}_length"
    return _v


# ---------------------------------------------------------------------------
# Schema injection block (appended to prompt — single source of truth)
# ---------------------------------------------------------------------------

def _build_schema_injection(schema: dict) -> str:
    """
    Generate a human-readable OUTPUT SCHEMA block from the schema dict.
    Only emitted when an _array_field is present (conductor batch pattern).
    """
    array_field = schema.get("_array_field")
    if not array_field:
        return ""

    props      = schema.get("properties", {})
    array_prop = props.get(array_field, {})
    item_props = array_prop.get("items", {}).get("properties", {})

    lines = [
        "",
        "=" * 55,
        "OUTPUT SCHEMA (AUTO-GENERATED — DO NOT MODIFY)",
        "=" * 55,
        "Return ONLY valid JSON with this exact structure:",
        "",
        "{",
    ]
    for fn, fd in props.items():
        if fn == array_field:
            continue
        lines.append(f'  "{fn}": {_field_annotation(fn, fd)},')
    lines += [f'  "{array_field}": [', "    {"]
    for fn, fd in item_props.items():
        lines.append(f'      "{fn}": {_field_annotation(fn, fd)},')
    lines += ["    }", "  ]", "}", ""]

    rules = schema.get("_validation_rules", {})
    mode_field = rules.get("mode_field")
    unique_rules = {
        k.replace("unique_field_in_", ""): v
        for k, v in rules.items() if k.startswith("unique_field_in_")
    }
    mode_counts = {
        k.replace("_mode_count", ""): v
        for k, v in rules.items()
        if k.endswith("_mode_count") and k != "mode_field"
    }
    if mode_field and (mode_counts or unique_rules):
        lines.append("VALIDATION RULES:")
        for mode_name, count in mode_counts.items():
            uf = unique_rules.get(mode_name)
            note = f", all {uf} values must be unique" if uf else ""
            lines.append(f"- {mode_field}={mode_name}: {count} {array_field}{note}")
        lines.append("")

    return "\n".join(lines)


def _field_annotation(field_name: str, field_def: dict) -> str:
    """Inline annotation string for the schema injection block."""
    ftype = field_def.get("type", "string")
    if "enum" in field_def:
        separator = '" | "'
        return f'"{separator.join(field_def["enum"])}"'
    parts = []
    mn = field_def.get("minLength")
    mx = field_def.get("maxLength")
    if mn and mx:
        parts.append(f"{mn}-{mx} chars")
    elif mn:
        parts.append(f"min {mn} chars")
    elif mx:
        parts.append(f"max {mx} chars")
    desc = field_def.get("description")
    if desc:
        parts.append(desc)
    suffix = "  // " + ", ".join(parts) if parts else ""
    return f'"({ftype})"{suffix}'


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class SchemaAdapterMixin:
    """
    Mixin for adapters that load their system prompt and output schema from
    files on disk rather than hardcoding them in Python.

    Inherit alongside BaseAIAdapter:

        class MyAdapter(SchemaAdapterMixin, BaseAIAdapter):
            _ADAPTER_DIR = Path(__file__).parent

            def user_message(self, question: str, **kwargs) -> str:
                return question

    Call load_schema() once before the adapter is used (e.g. on first call
    or at application startup). build_prompt() and output_type_schema() are
    fully implemented by this mixin; only user_message() must be provided by
    the concrete adapter.

    build_prompt() performs {{key}} → value substitution from **kwargs, so a
    master_prompt.txt containing:

        Context:
        {{context}}

    will have {{context}} replaced by whatever context=... is passed to
    AIRunner.run() (or directly to build_prompt()).
    """

    _ADAPTER_DIR: Path = None   # type: ignore[assignment]

    def __init__(self) -> None:
        self._prompt_template: str | None = None
        self._output_type: Type[BaseModel] | None = None
        self._raw_schema: dict = {}

    # ------------------------------------------------------------------
    # Schema / prompt loading
    # ------------------------------------------------------------------

    def load_schema(self, schema_name: str = "schema") -> None:
        """
        Load master_prompt.txt and {schema_name}.json from _ADAPTER_DIR.

        master_prompt.txt is required.
        {schema_name}.json is optional; if absent, an empty OutputSchema is used.
        """
        if self._ADAPTER_DIR is None:
            raise RuntimeError(
                f"{self.__class__.__name__} must set _ADAPTER_DIR = Path(__file__).parent"
            )
        adapter_dir = Path(self._ADAPTER_DIR)

        prompt_path = adapter_dir / "master_prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"master_prompt.txt not found: {prompt_path}")
        self._prompt_template = prompt_path.read_text(encoding="utf-8")

        schema_path = adapter_dir / f"{schema_name}.json"
        if schema_path.exists():
            with open(schema_path, encoding="utf-8") as f:
                self._raw_schema = json.load(f)
            self._output_type = build_output_type_schema(self._raw_schema)
            # Append auto-generated schema block if this is an array-centric schema
            injection = _build_schema_injection(self._raw_schema)
            if injection:
                self._prompt_template = self._prompt_template.rstrip() + "\n" + injection
        else:
            logger.warning(f"No schema file at {schema_path} — using empty OutputSchema")
            self._raw_schema = {}
            self._output_type = build_output_type_schema({})

        logger.info(
            f"[{self.__class__.__name__}] Loaded schema '{schema_name}' "
            f"from {adapter_dir.name}/"
        )

    def _ensure_loaded(self) -> None:
        if self._prompt_template is None:
            self.load_schema()

    # ------------------------------------------------------------------
    # BaseAIAdapter implementations
    # ------------------------------------------------------------------

    def build_prompt(self, **kwargs) -> str:
        """
        Return the system prompt with {{key}} placeholders substituted from kwargs.
        Loads the schema on first call if load_schema() was not called explicitly.
        """
        self._ensure_loaded()
        text = self._prompt_template
        for key, value in kwargs.items():
            text = text.replace(f"{{{{{key}}}}}", str(value))
        return text

    def output_type_schema(self) -> Type[BaseModel]:
        """Return the dynamically-built Pydantic output model."""
        self._ensure_loaded()
        return self._output_type
