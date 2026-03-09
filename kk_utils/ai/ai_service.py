"""
kk_utils.ai.ai_service — Backend-agnostic AI Service

Provides AI-powered text processing via the OpenAI Agents SDK.
Supports multiple providers via "provider/modelname" format.

Supported providers:
- openai: Official OpenAI or OpenAI-compatible endpoints
- qwen/dashscope: Alibaba DashScope (Qwen models)
- ollama: Local Ollama models
- anthropic: Anthropic Claude (via OpenAI-compatible endpoint)
- mock: Mock client for testing

Usage:
    from kk_utils.ai import get_ai_service

    ai = get_ai_service()          # reads API_MODEL from env
    result = await ai.generate_text("Hello!")
    structured = await ai.generate_structured(prompt, system, MyModel)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from pydantic import BaseModel, Field

from openai import AsyncOpenAI

try:
    from agents import Agent as SDKAgent
    from agents import Runner, trace, OpenAIChatCompletionsModel, AgentOutputSchema
    from agents.tracing import agent_span, generation_span, function_span
    AGENTS_SDK_AVAILABLE = True
except ImportError:
    AGENTS_SDK_AVAILABLE = False
    SDKAgent = None
    Runner = None
    trace = None
    OpenAIChatCompletionsModel = None
    AgentOutputSchema = None
    agent_span = None
    generation_span = None
    function_span = None

logger = logging.getLogger(__name__)


# =============================================================================
# Output Schemas (Pydantic Models)
# =============================================================================

class TextResult(BaseModel):
    """Structured output for plain text generation."""
    response: str = Field(..., description="Generated text response")


class SummaryResult(BaseModel):
    """Structured output for text summarization."""
    summary: str = Field(..., description="Concise summary of the input text")
    key_points: List[str] = Field(default_factory=list, description="Key bullet points")
    word_count: int = Field(..., description="Approximate word count of summary")


class RewriteResult(BaseModel):
    """Structured output for text rewriting."""
    rewritten_text: str = Field(..., description="Rewritten version of the input")
    tone: str = Field(..., description="Tone/style applied")
    changes_made: List[str] = Field(default_factory=list, description="List of major changes")


class TaskExtractionResult(BaseModel):
    """Structured output for task extraction."""
    tasks: List[dict] = Field(default_factory=list, description="Extracted tasks")
    total_tasks: int = Field(..., description="Number of tasks extracted")

    class TaskItem(BaseModel):
        title: str = Field(..., description="Task title/description")
        priority: str = Field(default="medium", description="high|medium|low")
        due_date: Optional[str] = Field(None, description="Extracted due date if any")
        context: Optional[str] = Field(None, description="Additional context or notes")


class IntentClassificationResult(BaseModel):
    """Structured output for intent classification."""
    model_config = {"extra": "forbid"}

    intent: str = Field(..., description="Classified intent category")
    confidence: float = Field(..., description="Confidence score 0.0-1.0")
    entities: dict = Field(default_factory=dict, description="Extracted entities")
    suggested_tools: List[str] = Field(default_factory=list, description="Recommended tools to use")
    requires_clarification: bool = Field(default=False, description="Whether clarification is needed")
    clarification_question: Optional[str] = Field(None, description="Clarification question if needed")


# =============================================================================
# CallContext - Attribution for usage tracking
# =============================================================================

@dataclass
class CallContext:
    """Caller identity for usage event attribution."""
    agent_name: str
    feature_name: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None


# =============================================================================
# AI Service
# =============================================================================

class AIService:
    """
    AI-powered text processing service (backend-agnostic).

    Uses OpenAI Agents SDK for structured output with guardrails.
    Supports multiple providers via api_model format: "provider/modelname"

    Provider format examples:
        openai/gpt-4o-mini
        qwen/qwen-turbo
        ollama/llama3
        anthropic/claude-3-haiku-20240307
        mock/test
    """

    def __init__(
        self,
        api_model: str = "openai/gpt-4o-mini",
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        enable_output_schema: bool = True,
    ):
        self.api_model = api_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_output_schema = enable_output_schema

        # Parse provider/model format
        if "/" in api_model:
            self.provider, self.model = api_model.split("/", 1)
        else:
            logger.warning(f"Invalid api_model '{api_model}', using mock provider")
            self.provider = "mock"
            self.model = "test"

        self.provider = self.provider.strip().lower()
        self.model = self.model.strip()

        # Per-provider API key env var
        api_key_env_map = {
            "openai": "OPENAI_API_KEY",
            "qwen": "DASHSCOPE_API_KEY",
            "dashscope": "DASHSCOPE_API_KEY",
            "ollama": "OLLAMA_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }

        api_key_env = api_key_env_map.get(self.provider)

        if not api_key_env:
            logger.warning(f"Unknown provider '{self.provider}' - using mock mode")
            self.provider = "mock"
            self.client = None
            self._prompts = self._load_prompts()
            return

        self.api_key = api_key or os.environ.get(api_key_env, "")

        if not self.api_key:
            logger.warning(f"{api_key_env} not set - using mock mode")
            self.client = None
            self._prompts = self._load_prompts()
            return

        # Base URL per provider
        self.base_url = api_base_url
        if not self.base_url:
            if self.provider in ("qwen", "dashscope"):
                self.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            elif self.provider == "ollama":
                self.base_url = "http://localhost:11434/v1"

        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = AsyncOpenAI(**client_kwargs)
        logger.info(f"AIService initialized: {api_model} (provider={self.provider})")

        self._prompts = self._load_prompts()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def summarize(
        self,
        text: str,
        max_length: int = 150,
        bullet_points: bool = True,
        context: Optional[CallContext] = None,
    ) -> SummaryResult:
        """Summarize text with configurable length and bullet points."""
        system_prompt = self._build_summarize_prompt(max_length, bullet_points)
        return await self._call_ai(
            system_prompt=system_prompt,
            user_text=text,
            output_type=SummaryResult,
            context=context,
        )

    async def rewrite(
        self,
        text: str,
        tone: str = "professional",
        style: Optional[str] = None,
        context: Optional[CallContext] = None,
    ) -> RewriteResult:
        """Rewrite text with a different tone/style."""
        system_prompt = self._build_rewrite_prompt(tone, style)
        return await self._call_ai(
            system_prompt=system_prompt,
            user_text=text,
            output_type=RewriteResult,
            context=context,
        )

    async def extract_tasks(
        self,
        text: str,
        context: Optional[CallContext] = None,
    ) -> TaskExtractionResult:
        """Extract actionable tasks from text."""
        system_prompt = self._build_extract_tasks_prompt()
        return await self._call_ai(
            system_prompt=system_prompt,
            user_text=text,
            output_type=TaskExtractionResult,
            context=context,
        )

    async def classify_intent(
        self,
        text: str,
        context: Optional[CallContext] = None,
    ) -> IntentClassificationResult:
        """Classify user intent and extract entities."""
        system_prompt = self._build_classify_intent_prompt()
        return await self._call_ai(
            system_prompt=system_prompt,
            user_text=text,
            output_type=IntentClassificationResult,
            context=context,
        )

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[CallContext] = None,
    ) -> Dict[str, Any]:
        """
        Generate a plain text response via the Agents SDK.

        Returns:
            Dict with 'response' key containing the generated text.
        """
        default_system = self._prompts.get("generate_text", {}).get("default_system", "You are a helpful assistant.")
        result = await self._call_ai(
            system_prompt=system_prompt or default_system,
            user_text=prompt,
            output_type=TextResult,
            context=context,
        )
        return {"response": result.response}


    # -------------------------------------------------------------------------
    # report_progress tool schema — injected automatically when progress_callback is set
    # -------------------------------------------------------------------------

    _REPORT_PROGRESS_TOOL: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "report_progress",
            "description": (
                "Report your current plan and progress. "
                "Call FIRST with all steps as pending to share your plan before doing any work. "
                "Then update individual steps to in_progress as you start them, "
                "and completed or failed when done. "
                "Keep plans concise and actionable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "description": "All plan steps with current status",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Step ID e.g. 1, 2, 3"},
                                "content": {"type": "string", "description": "What this step does"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed", "failed"],
                                },
                            },
                            "required": ["id", "content", "status"],
                        },
                    }
                },
                "required": ["steps"],
            },
        },
    }

    async def chat_with_tools(
        self,
        message: str,
        tools: List[Dict],
        system_prompt: str,
        conversation_history: Optional[List[Dict]] = None,
        context: Optional[CallContext] = None,
        max_iterations: int = 10,
        progress_callback: Optional[Callable[[List[Dict]], None]] = None,
        max_plan_steps: int = 8,
        trace_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Chat with native LLM tool calling.

        The LLM decides which tools to call. This method executes the tool-call
        loop: send message -> LLM picks tools -> execute via AgentRegistry -> feed
        results back -> repeat until LLM produces a final text response.

        When progress_callback is provided, a report_progress tool is injected
        automatically. The LLM calls it to broadcast its plan and update step
        status as it works. Progress calls do NOT count against max_iterations.

        Args:
            message: Current user message
            tools: OpenAI-compatible tool schemas (from AgentRegistry)
            system_prompt: System instruction injected before conversation
            conversation_history: Previous messages as [{"role":..., "content":...}]
            context: Call context for usage attribution
            max_iterations: Max real tool-call rounds before giving up
            progress_callback: Called with List[step_dict] on each report_progress call
            max_plan_steps: Max steps the LLM is allowed to plan (enforced in tool)
            trace_callback: Called with a string event message at each pipeline stage (showcase mode)

        Returns:
            Final text response from the LLM.
        """
        if not self.client or self.provider == "mock":
            logger.warning("chat_with_tools: AI in mock mode")
            return "[Mock] Configure API_MODEL for real AI responses."

        import json
        from kk_utils.agent_tools import get_registry

        registry = get_registry()

        # Inject report_progress tool and add planning instructions to system prompt
        active_tools = list(tools) if tools else []
        effective_system = system_prompt
        if progress_callback is not None:
            active_tools = [self._REPORT_PROGRESS_TOOL] + active_tools
            suffix_template = self._prompts.get("chat_with_tools", {}).get(
                "progress_workflow_suffix",
                "\n\nWORKFLOW — follow these steps in order:\n"
                "1. Call report_progress with your full plan (all steps as pending).\n"
                "2. Call report_progress again marking executing steps as in_progress, then call the tools.\n"
                "3. After tool results are returned, call report_progress once more marking all finished steps as completed or failed.\n"
                "4. Then give your final answer.\n"
                "Keep your plan to {max_plan_steps} steps or fewer.",
            )
            effective_system = system_prompt + suffix_template.format(max_plan_steps=max_plan_steps)

        # Build initial message list
        messages: List[Dict] = [{"role": "system", "content": effective_system}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": message})

        openai_tools = active_tools if active_tools else None

        # SDK tracing: outer trace + agent span wrap the entire tool-calling loop
        # Inner generation_span and function_span are created per-LLM-call and per-tool-call
        tool_names = [t["function"]["name"] for t in active_tools if t.get("function", {}).get("name") != "report_progress"]
        _sdk = AGENTS_SDK_AVAILABLE
        # Start trace first so it is the "current trace" when agent_span is created
        _t = trace("kk_utils_ai") if _sdk else None
        if _t:
            _t.start(mark_as_current=True)
        _a = agent_span("AIServiceAgent", tools=tool_names) if _sdk else None
        if _a:
            _a.start(mark_as_current=True)

        try:
            real_iterations = 0  # only real tool calls count; progress calls are free
            emit_llm_trace = True  # suppress duplicate "calling LLM" after report_progress-only loops

            while real_iterations < max_iterations:
                if trace_callback and emit_llm_trace:
                    trace_callback("agent_me: calling LLM")
                emit_llm_trace = True  # reset; will be set False below if only progress tools ran

                # --- LLM call wrapped in generation_span ---
                _gen = generation_span(input=list(messages), model=self.model) if _sdk else None
                if _gen:
                    _gen.start(mark_as_current=True)
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=openai_tools,
                        tool_choice="auto" if openai_tools else None,
                    )
                finally:
                    if _gen:
                        choice0 = response.choices[0] if response.choices else None
                        if choice0 and _gen.span_data:
                            _gen.span_data.output = [choice0.message.model_dump(exclude_unset=False)]
                        if response.usage and _gen.span_data:
                            _gen.span_data.usage = {
                                "input_tokens": response.usage.prompt_tokens,
                                "output_tokens": response.usage.completion_tokens,
                            }
                        _gen.finish(reset_current=True)

                choice = response.choices[0]

                if choice.finish_reason == "tool_calls":
                    messages.append(choice.message.model_dump(exclude_unset=False))
                    if trace_callback:
                        real_names = [tc.function.name for tc in choice.message.tool_calls if tc.function.name != "report_progress"]
                        if real_names:
                            trace_callback(f"agent_me: LLM requests tool(s): {', '.join(real_names)}")

                    has_real_tool = False
                    _executed_tools: Dict[str, Any] = {}  # cache: "tool_name:args_json" → result (dedup within batch)

                    for tool_call in choice.message.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            tool_args = json.loads(tool_call.function.arguments or "{}")
                        except json.JSONDecodeError:
                            tool_args = {}
                        _dedup_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"

                        if tool_name == "report_progress":
                            steps = tool_args.get("steps", [])

                            # Enforce max_plan_steps limit
                            if len(steps) > max_plan_steps:
                                tool_result = {
                                    "error": f"Plan has {len(steps)} steps but limit is {max_plan_steps}. Consolidate your plan.",
                                    "max_allowed": max_plan_steps,
                                }
                                logger.warning(f"report_progress: plan exceeds limit ({len(steps)} > {max_plan_steps})")
                            else:
                                if progress_callback is not None:
                                    progress_callback(steps)
                                logger.debug(f"Progress update: {len(steps)} steps")
                                tool_result = {"acknowledged": True, "steps_recorded": len(steps)}

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(tool_result),
                            })
                            # report_progress does NOT count against max_iterations

                        else:
                            has_real_tool = True

                            # Deduplicate: same tool + same args in one batch → execute once, reuse result
                            if _dedup_key in _executed_tools:
                                result = _executed_tools[_dedup_key]
                                logger.debug(f"Tool dedup: {tool_name} (same args) already ran this batch — reusing result")
                            else:
                                logger.debug(f"Tool call: {tool_name}({tool_args})")
                                if trace_callback:
                                    trace_callback(f"agent_me: executing {tool_name}")

                                # --- tool execution wrapped in function_span ---
                                _fn = function_span(name=tool_name, input=json.dumps(tool_args)) if _sdk else None
                                if _fn:
                                    _fn.start(mark_as_current=True)
                                try:
                                    result = registry.execute(tool_name, **tool_args)
                                finally:
                                    if _fn:
                                        if _fn.span_data:
                                            _fn.span_data.output = json.dumps(result)[:500]
                                        _fn.finish(reset_current=True)

                                _executed_tools[_dedup_key] = result
                                if trace_callback:
                                    trace_callback(f"agent_me: {tool_name} done")

                            logger.debug(f"Tool result [{tool_name}]: {str(result)[:200]}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result),
                            })

                    if has_real_tool:
                        real_iterations += 1
                    else:
                        # Only report_progress was called — suppress next "calling LLM" trace
                        emit_llm_trace = False

                elif choice.finish_reason == "stop":
                    self._on_usage(response, context, TextResult)
                    if trace_callback:
                        trace_callback("agent_me: final response received")
                    return choice.message.content or ""

                else:
                    logger.warning(f"chat_with_tools: unexpected finish_reason={choice.finish_reason!r}")
                    break

            logger.warning("chat_with_tools: max_iterations reached without final response")
            return "I am having trouble processing your request. Please try again."

        except Exception as e:
            logger.error(f"chat_with_tools failed: {e}", exc_info=True)
            return "I encountered an error. Please try again."

        finally:
            if _a:
                _a.finish(reset_current=True)
            if _t:
                _t.finish(reset_current=True)

    async def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        output_type: type[BaseModel],
        context: Optional[CallContext] = None,
    ) -> BaseModel:
        """
        Generate a typed, validated Pydantic response via the Agents SDK.

        Args:
            prompt: User prompt
            system_prompt: System instruction
            output_type: Pydantic model class for structured output
            context: Call context for attribution

        Returns:
            Instance of output_type validated by the SDK.
        """
        return await self._call_ai(
            system_prompt=system_prompt,
            user_text=prompt,
            output_type=output_type,
            context=context,
        )

    # -------------------------------------------------------------------------
    # Prompt Builders — load from kk_utils/ai/prompts/<name>.json
    # To swap a prompt for A/B testing, edit or replace the corresponding JSON file.
    # -------------------------------------------------------------------------

    def _load_prompts(self) -> Dict[str, Dict]:
        """
        Load all prompt templates from the prompts/ directory next to this file.

        Files MUST be saved as UTF-8 to support multilingual content (Chinese, etc.).
        Reading uses encoding="utf-8" explicitly.
        When writing prompt files programmatically, always use:
            json.dumps(data, ensure_ascii=False, indent=2)
        to preserve non-ASCII characters in readable form rather than \\uXXXX escapes.
        """
        prompts_dir = Path(__file__).parent / "prompts"
        prompts: Dict[str, Dict] = {}
        if prompts_dir.is_dir():
            for path in sorted(prompts_dir.glob("*.yaml")):
                try:
                    import yaml as _yaml
                    raw = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                    prompts[path.stem] = self._normalize_prompt(raw)
                except Exception as e:
                    logger.warning(f"Failed to load prompt file {path.name}: {e}")
        else:
            logger.warning(f"Prompts directory not found: {prompts_dir}")
        return prompts

    @staticmethod
    def _normalize_prompt(data: Dict) -> Dict:
        """
        Normalize a loaded prompt dict: join any list values with newlines.

        Prompt JSON files store text as arrays of lines for readability:
            "system": [
                "You are an expert assistant.",
                "",
                "GUIDELINES:",
                "- Be concise"
            ]
        This method joins them into a single string before use.
        Non-list values are left unchanged.
        """
        return {
            k: v.rstrip() if isinstance(v, str) else v
            for k, v in data.items()
        }

    @staticmethod
    def save_prompt(name: str, data: Dict) -> None:
        """
        Save a prompt template to prompts/<name>.json.

        Uses ensure_ascii=False so Chinese/multilingual characters are stored
        as readable Unicode rather than \\uXXXX escape sequences.

        Args:
            name: Prompt name (e.g. "summarize") — becomes the filename stem
            data: Prompt data dict to serialize
        """
        prompts_dir = Path(__file__).parent / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        path = prompts_dir / f"{name}.yaml"
        import yaml as _yaml
        path.write_text(
            _yaml.dump(data, allow_unicode=True, default_flow_style=False,
                       sort_keys=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Saved prompt: {path}")

    def _build_summarize_prompt(self, max_length: int, bullet_points: bool) -> str:
        p = self._prompts.get("summarize", {})
        bullet_instruction = p.get("bullet_instruction", "") if bullet_points else ""
        template = p.get("system", "You are a summarization assistant. Summarize in {max_length} words{bullet_instruction}.")
        return template.format(max_length=max_length, bullet_instruction=bullet_instruction)

    def _build_rewrite_prompt(self, tone: str, style: Optional[str]) -> str:
        p = self._prompts.get("rewrite", {})
        style_tpl = p.get("style_instruction_template", " and {style} style")
        style_instruction = style_tpl.format(style=style) if style else ""
        template = p.get("system", "You are a rewriting assistant. Tone: {tone}{style_instruction}.")
        return template.format(tone=tone, style_instruction=style_instruction)

    def _build_extract_tasks_prompt(self) -> str:
        return self._prompts.get("extract_tasks", {}).get(
            "system", "You are a task extraction assistant."
        )

    def _build_classify_intent_prompt(self) -> str:
        return self._prompts.get("classify_intent", {}).get(
            "system", "You are an intent classification assistant."
        )

    # -------------------------------------------------------------------------
    # Core AI Client
    # -------------------------------------------------------------------------

    async def _call_ai(
        self,
        system_prompt: str,
        user_text: str,
        output_type: type[BaseModel],
        context: Optional[CallContext] = None,
    ) -> BaseModel:
        """
        Call AI via OpenAI Agents SDK with structured output.

        Falls back to mock response if client is unavailable or SDK is missing.
        """
        if not self.client or self.provider == "mock":
            logger.warning(f"AI in mock mode (provider={self.provider})")
            return self._mock_response(output_type)

        if not AGENTS_SDK_AVAILABLE:
            logger.error("OpenAI Agents SDK not available - using mock response")
            return self._mock_response(output_type)

        try:
            model = OpenAIChatCompletionsModel(
                model=self.model,
                openai_client=self.client,
            )

            use_schema = self._get_output_schema_enabled(context)
            if use_schema and AgentOutputSchema:
                wrapped_output = AgentOutputSchema(output_type, strict_json_schema=False)
            elif use_schema:
                wrapped_output = output_type
            else:
                wrapped_output = None  # No schema enforcement — raw text output

            agent = SDKAgent(
                name="AIServiceAgent",
                instructions=system_prompt,
                model=model,
                output_type=wrapped_output,
            )

            user_messages = [{"role": "user", "content": user_text}]

            with trace("kk_utils_ai"):
                result = await Runner.run(agent, user_messages)

            final_output = result.final_output

            # Optional usage callback (injected by callers, not required)
            self._on_usage(result, context, output_type)

            if hasattr(final_output, "model_dump"):
                return final_output
            elif isinstance(final_output, dict):
                return output_type.model_validate(final_output)
            elif isinstance(final_output, str) and not use_schema:
                # Schema disabled — wrap raw text into a best-effort model
                try:
                    return output_type.model_validate({"response": final_output})
                except Exception:
                    return self._mock_response(output_type)
            else:
                logger.error(f"Unexpected output type: {type(final_output)}")
                return self._mock_response(output_type)

        except Exception as e:
            logger.error(f"AI call failed: {e}", exc_info=True)
            return self._mock_response(output_type)

    def _on_usage(self, result: Any, context: Optional[CallContext], output_type: type) -> None:
        """
        Hook for usage tracking. Override or replace in subclasses.
        Default: no-op (keeps kk-utils backend-agnostic).
        """
        pass

    def _get_output_schema_enabled(self, context: Optional[CallContext]) -> bool:
        """
        Hook for per-call output schema control. Override in subclasses.

        Called by _call_ai() before each SDK invocation. Subclasses can
        consult an external policy store (e.g. Governor) using the agent
        name from context to return a per-agent decision.

        Default: returns self.enable_output_schema (set at construction time).
        """
        return self.enable_output_schema

    def _mock_response(self, output_type: type[BaseModel]) -> BaseModel:
        """Generate a mock response for testing / no-API-key mode."""
        logger.debug("Generating mock AI response")

        if output_type == TextResult:
            return TextResult(response="[Mock] Configure API_MODEL for real AI responses.")
        elif output_type == SummaryResult:
            return SummaryResult(
                summary="Mock summary. Configure OPENAI_API_KEY for real AI responses.",
                key_points=["Mock point 1", "Mock point 2"],
                word_count=10,
            )
        elif output_type == RewriteResult:
            return RewriteResult(
                rewritten_text="Mock rewritten text. Configure OPENAI_API_KEY for real AI responses.",
                tone="professional",
                changes_made=["Mock change 1"],
            )
        elif output_type == TaskExtractionResult:
            return TaskExtractionResult(
                tasks=[{"title": "Mock task", "priority": "medium", "due_date": None, "context": "Mock"}],
                total_tasks=1,
            )
        elif output_type == IntentClassificationResult:
            return IntentClassificationResult(
                intent="casual_chat",
                confidence=0.8,
                entities={},
                suggested_tools=[],
            )
        else:
            try:
                return output_type.model_validate({})
            except Exception:
                raise ValueError(f"Cannot mock output type: {output_type.__name__}")


# =============================================================================
# Factory
# =============================================================================

def get_ai_service(
    api_model: Optional[str] = None,
    api_base_url: Optional[str] = None,
) -> AIService:
    """
    Return the singleton AIService instance.

    Reads API_MODEL and AI_BASE_URL from environment on first call.
    Call with no arguments from anywhere — kk-utils handles the rest.

    Args:
        api_model: Override model (format: "provider/modelname")
        api_base_url: Override base URL

    Returns:
        Configured AIService instance
    """
    if not hasattr(get_ai_service, "_instance"):
        resolved_model = api_model or os.environ.get("API_MODEL", "openai/gpt-4o-mini")
        resolved_url = api_base_url or os.environ.get("AI_BASE_URL")
        get_ai_service._instance = AIService(
            api_model=resolved_model,
            api_base_url=resolved_url,
        )
    return get_ai_service._instance
