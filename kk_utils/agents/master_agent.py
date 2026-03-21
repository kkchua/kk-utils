"""
kk_utils.agents.master_agent — Master Agent orchestrator

Backend-agnostic agent orchestrator that:
- Resolves persona from config
- Selects appropriate adapter based on persona.adapter_type
- Loads skills and tools via adapter
- Routes tool execution through skill handlers
- Delegates chat execution to adapter
- Returns standardized AgentResponse

Usage:
    from kk_utils.agents import MasterAgent

    agent = MasterAgent(personas_config_path="config/personas.yaml")
    response = await agent.chat(
        message="Hello",
        persona_name="ai_assistant",
        user_id="user123",
        user_role="demo",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from .agent_response import AgentResponse
from .base_agent_adapter import BaseAgentAdapter
from .agent_registry import AgentRegistry
from ..persona_config import PersonaConfig, load_persona

logger = logging.getLogger(__name__)


class MasterAgent:
    """
    Master Agent - backend-agnostic orchestrator.
    
    Reusable across:
    - personal-assistant backend
    - gradio-apps
    - future projects
    
    Responsibilities:
    - Load persona from config
    - Select adapter based on persona.adapter_type
    - Load skills and tools via adapter
    - Execute chat via adapter
    - Return standardized AgentResponse
    
    NOT responsible for:
    - Governor access control (backend-specific)
    - Guardrails (backend-specific)
    - Audit logging (backend-specific)
    - Usage tracking (backend-specific)
    """
    
    def __init__(
        self,
        personas_config_path: Optional[str] = None,
        auto_register_adapters: bool = True,
        auto_register_handlers: bool = True,
    ):
        """
        Initialize Master Agent.

        Args:
            personas_config_path: Path to personas.yaml (required for chat)
            auto_register_adapters: If True, auto-register built-in adapters
            auto_register_handlers: If True, auto-register skill handlers
        """
        self.personas_config_path = Path(personas_config_path) if personas_config_path else None
        self.adapter_registry = AgentRegistry.instance()
        
        # Initialize skill handler registry
        from .skill_handlers import SkillHandlerRegistry
        self.handler_registry = SkillHandlerRegistry.instance()

        if auto_register_adapters:
            self._register_builtin_adapters()
        
        if auto_register_handlers:
            self._register_builtin_handlers()
    
    def _register_builtin_adapters(self) -> None:
        """Register built-in adapters."""
        try:
            from .adapters import AgentMeAdapter, AIAssistantAdapter

            self.adapter_registry.register("agent_me", AgentMeAdapter, override=True)
            self.adapter_registry.register("ai_assistant", AIAssistantAdapter, override=True)
            logger.info("Registered built-in agent adapters: agent_me, ai_assistant")
        except ImportError as e:
            logger.warning(f"Could not register built-in adapters: {e}")
    
    def _register_builtin_handlers(self) -> None:
        """Register built-in skill handlers."""
        try:
            from .skill_handlers import StandardSkillHandler, ComfyUISkillHandler
            
            self.handler_registry.register("standard", StandardSkillHandler(), override=True)
            self.handler_registry.register("comfyui", ComfyUISkillHandler(), override=True)
            logger.info("Registered built-in skill handlers: standard, comfyui")
        except ImportError as e:
            logger.warning(f"Could not register built-in skill handlers: {e}")
    
    def set_personas_config_path(self, path: str) -> None:
        """
        Set personas config path.
        
        Args:
            path: Path to personas.yaml
        """
        self.personas_config_path = Path(path)
    
    async def chat(
        self,
        message: str,
        persona_name: str,
        user_id: str,
        user_role: str = "demo",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        model: str = "openai/gpt-5-nano",
        debug_mode: bool = False,
        skill_tags: Optional[List[str]] = None,
        skill: Optional[str] = None,
        # Pipeline execution context (bypasses LLM when handler is registered)
        execution_type: Optional[str] = None,   # e.g. "vision_pipeline"
        skill_adapter: Optional[str] = None,    # e.g. "image_variation"
        prompt_name: Optional[str] = None,      # e.g. "master_prompt_qwen"
        attachments: Optional[List[str]] = None,
        input_values: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        Process a chat message.

        This is backend-agnostic - no Governor, no audit, no usage tracking.
        Those are added by the backend's OrchestratorService.

        Args:
            message: User message
            persona_name: Name of persona to use
            user_id: User identifier
            user_role: User role (for adapter context)
            conversation_history: Previous messages [{role, content}, ...]
            model: AI model to use
            debug_mode: If True, show simulation without calling LLM
            skill_tags: Override persona's default skill tags
            skill: Explicit skill selection from UI (e.g., "deep_research", "None")

        Returns:
            AgentResponse with the AI reply

        Raises:
            ValueError: If persona not found or config path not set
            KeyError: If adapter not found
        """
        if not self.personas_config_path:
            raise ValueError("personas_config_path not set")
        
        # 1. Load persona
        persona = load_persona(persona_name, config_path=self.personas_config_path)
        if persona is None:
            raise ValueError(f"Persona '{persona_name}' not found")
        
        # 2. Get adapter
        adapter_type = persona.adapter_type or "agent_me"
        adapter_class = self.adapter_registry.get_adapter(adapter_type)
        adapter: BaseAgentAdapter = adapter_class()

        logger.info(f"MasterAgent: persona={persona_name!r} adapter={adapter_type!r}")

        # 2b. Pipeline check — if execution_type maps to a registered handler,
        #     skip LLM entirely and let the handler run the pipeline directly.
        if execution_type:
            handler = self.handler_registry.get_handler(execution_type)
            if handler:
                logger.info(
                    f"MasterAgent: routing to handler '{execution_type}' "
                    f"(skill={skill!r}, skill_adapter={skill_adapter!r}, prompt={prompt_name!r})"
                )
                from .skill_handlers import SkillContext
                context = SkillContext(
                    user_id=user_id,
                    user_role=user_role,
                    persona_name=persona_name,
                    persona_collection=persona.collection,
                    model=model,
                    attachments=attachments or [],
                    extra={
                        "skill_adapter": skill_adapter,
                        "prompt_name": prompt_name,
                        "input_values": input_values or {},
                        "adapter_instance": adapter,  # persona adapter (has generate_vision_raw)
                    },
                )
                skill_result = await handler.handle(skill or "", {}, context)
                output = skill_result.output or {}
                return AgentResponse(
                    response_text=output.get("response_text", skill_result.error or "Pipeline completed.") if isinstance(output, dict) else str(output),
                    agent_type=adapter_type,
                    persona_name=persona_name,
                    collection=persona.collection,
                    tools_available=0,
                    success=skill_result.success,
                    error=skill_result.error,
                    metadata=skill_result.to_dict(),
                )

        # 3. Load schema config (optional)
        schema_config = adapter.get_tools_config(persona) or {}

        # 4. Determine final skill_tags - Priority: UI skill_tags > persona config > adapter default
        if skill_tags is not None:
            # UI explicitly selected skills (or explicitly selected none)
            logger.info(f"MasterAgent: Using UI-provided skill_tags={skill_tags}")
            skills = []  # Skills not needed when we have explicit tags
            final_skill_tags = skill_tags
        else:
            # Use persona config defaults
            skills = (
                schema_config.get("skills")
                if schema_config.get("skills") is not None
                else (persona.skills if persona.skills is not None else adapter.get_skills())
            )
            final_skill_tags = (
                schema_config.get("skill_tags")
                if schema_config.get("skill_tags") is not None
                else (persona.skill_tags if persona.skill_tags is not None else adapter.get_skill_tags())
            )
            logger.info(f"MasterAgent: Using persona defaults skills={skills}, skill_tags={final_skill_tags}")

        # 5. If skill != "None", add the skill's tags to final_skill_tags
        if skill and skill.lower() != "none":
            from ..agent_tools import get_registry as get_tools_registry
            tools_registry = get_tools_registry()

            # Get skill info to extract its tags
            skill_info = tools_registry.get_skill(skill)
            if skill_info:
                skill_module_tags = skill_info.get("tags", [])
                # Combine persona tags + selected skill tags (use set to avoid duplicates)
                final_skill_tags = list(set(final_skill_tags + skill_module_tags))
                logger.info(f"MasterAgent: Added tags from skill '{skill}': {skill_module_tags}")
            else:
                logger.warning(f"MasterAgent: Skill '{skill}' not found in registry")

        # 6. Load tools from registry
        tools = self._load_tools(final_skill_tags)

        # 7. Build system prompt (MasterAgent controls loading, NOT adapter)
        system_prompt = self._load_system_prompt(adapter, persona)
        
        # GOVERNOR: Append global system prompt suffix (tool limits, rules, etc.)
        # This is done in MasterAgent so it applies uniformly to ALL adapters
        try:
            from app.core.governor import PersonalAssistantGovernor
            governor = PersonalAssistantGovernor.instance()
            governor_suffix = governor.get_global_system_prompt_suffix()
            
            if governor_suffix and governor_suffix.strip():
                system_prompt = system_prompt.rstrip() + "\n\n" + governor_suffix.rstrip()
                logger.info(f"MasterAgent: Appended Governor suffix ({len(governor_suffix)} chars)")
        except ImportError:
            logger.debug("MasterAgent: Governor not available - skipping system prompt suffix")
        except Exception as e:
            logger.warning(f"MasterAgent: Failed to append Governor suffix: {e}")

        # 7. Build messages
        messages = self._build_messages(system_prompt, conversation_history, message)

        # === DEBUG MODE: Show simulation before calling LLM ===
        if debug_mode:
            logger.info("=" * 80)
            logger.info("DEBUG MODE - SIMULATION (NO LLM CALL)")
            logger.info("=" * 80)
            logger.info(f"Persona: {persona_name}")
            logger.info(f"Display Name: {persona.display_name}")
            logger.info(f"Collection: {persona.collection}")
            logger.info(f"Adapter Type: {persona.adapter_type}")
            logger.info(f"Prompt Template: {persona.adapter_prompt_template}")
            logger.info("")
            logger.info(f"Skills: {skills}")
            logger.info(f"Skill Tags: {skill_tags}")
            logger.info(f"Tools Count: {len(tools)}")
            if tools:
                logger.info("Tools:")
                for i, tool in enumerate(tools[:10], 1):  # Show first 10
                    func = tool.get("function", {})
                    logger.info(f"  {i}. {func.get('name', 'unknown')}")
                    logger.info(f"     Description: {func.get('description', 'N/A')[:100]}")
                if len(tools) > 10:
                    logger.info(f"  ... and {len(tools) - 10} more tools")
            logger.info("")
            logger.info("SYSTEM PROMPT:")
            logger.info("-" * 40)
            logger.info(system_prompt[:2000] + ("..." if len(system_prompt) > 2000 else ""))
            logger.info("-" * 40)
            logger.info("")
            logger.info("MESSAGES:")
            for i, msg in enumerate(messages):
                logger.info(f"  [{i}] {msg['role']}: {msg['content'][:200]}...")
            logger.info("")
            logger.info(f"Model: {model}")
            logger.info("=" * 80)
            logger.info("🔍 END DEBUG MODE")
            logger.info("=" * 80)

            # Return debug response without calling LLM
            # Filter tools to remove non-serializable function refs
            serializable_tools = []
            for tool in tools:
                tool_copy = tool.copy() if hasattr(tool, 'copy') else dict(tool)
                # Remove function_ref if present
                if 'function_ref' in tool_copy:
                    del tool_copy['function_ref']
                serializable_tools.append(tool_copy)
            
            return AgentResponse(
                response_text=f"[DEBUG MODE] Simulation complete - {len(tools)} tools loaded. Check debug panel for details.",
                agent_type="debug",
                persona_name=persona_name,
                collection=persona.collection,
                tools_available=len(tools),
                metadata={
                    "debug_mode": True,
                    "persona_name": persona_name,
                    "persona_display_name": persona.display_name,
                    "collection": persona.collection,
                    "adapter_type": persona.adapter_type,
                    "prompt_template": persona.adapter_prompt_template,
                    "system_prompt": system_prompt,
                    "messages": messages,
                    "tools": serializable_tools,  # Filtered tools without function refs
                    "model": model,
                    "skills": list(skills) if skills else [],
                    "skill_tags": list(skill_tags) if skill_tags else [],
                },
            )
        # === END DEBUG MODE ===
        
        # 8. Execute chat via adapter
        # Pass full persona object for trace name and metadata extraction
        response = await adapter.execute_chat(
            messages=messages,
            tools=tools,
            model=model,
            persona=persona,
        )
        
        # 9. Post-process (adapter-specific)
        response = adapter.post_process(response)
        
        return response
    
    def _load_tools(self, skill_tags: List[str]) -> List[Dict[str, Any]]:
        """
        Load tools from AgentRegistry by skill tags.

        Special handling:
        - If skill_tags contains "none" or "no_tools", returns empty list
        - If skill_tags is empty, returns empty list

        Args:
            skill_tags: Tags to filter tools by

        Returns:
            List of OpenAI-compatible tool schemas
        """
        from ..agent_tools import get_registry

        # Explicit "none" or "no_tools" tag means NO tools
        if not skill_tags or "none" in skill_tags or "no_tools" in skill_tags:
            logger.info(f"_load_tools: skill_tags={skill_tags} - returning [] (no tools)")
            return []

        registry = get_registry()
        tools = registry.get_tools_for_tags(skill_tags)
        logger.info(f"_load_tools: skill_tags={skill_tags} - {len(tools)} tools")
        return tools

    def _load_system_prompt(
        self,
        adapter: BaseAgentAdapter,
        persona: PersonaConfig,
    ) -> str:
        """
        Load system prompt - centralized logic in MasterAgent.

        Priority:
        1. Centralized master prompt: prompts/master/{adapter_prompt_template}.txt
        2. Adapter default: adapters/{adapter_type}/prompts/default.txt

        Args:
            adapter: Adapter instance
            persona: Persona configuration

        Returns:
            System prompt string
        """
        # Get template name from persona, fallback to persona name
        template_name = persona.adapter_prompt_template or persona.name

        # Try centralized master prompts first
        try:
            from kk_utils.agents.prompts import load_master_prompt
            system_prompt = load_master_prompt(template_name)
            logger.info(f"MasterAgent: Loaded master prompt '{template_name}'")
            return system_prompt
        except FileNotFoundError:
            logger.warning(
                f"Master prompt template '{template_name}' not found, "
                f"falling back to adapter default"
            )

        # Fallback to adapter's default prompt
        try:
            return adapter.load_prompt_template("default")
        except FileNotFoundError:
            logger.warning(f"Adapter default prompt not found, using minimal fallback")
            return self._get_fallback_prompt()

    def _get_fallback_prompt(self) -> str:
        """
        Fallback system prompt when no template is found.

        Returns:
            Minimal system prompt string
        """
        return """You are a helpful AI assistant.

You have access to tools that can help you answer questions.
When appropriate, use the available tools to gather information before responding.
"""
    
    def _build_messages(
        self,
        system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]],
        message: str,
    ) -> List[Dict[str, str]]:
        """
        Build messages list for AI service.
        
        Args:
            system_prompt: System prompt
            conversation_history: Previous messages
            message: Current user message
            
        Returns:
            List of message dicts
        """
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": message})
        
        return messages
    
    def list_available_adapters(self) -> List[str]:
        """
        List all registered adapter names.
        
        Returns:
            List of adapter names
        """
        return self.adapter_registry.list_adapters()
    
    def get_adapter_info(self, adapter_name: str) -> Dict[str, Any]:
        """
        Get info about a registered adapter.
        
        Args:
            adapter_name: Adapter name
            
        Returns:
            Dict with adapter info
        """
        return self.adapter_registry.get_adapter_info(adapter_name)
