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

import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from openai import AsyncOpenAI

try:
    from agents import Agent as SDKAgent
    from agents import Runner, trace, OpenAIChatCompletionsModel, AgentOutputSchema
    AGENTS_SDK_AVAILABLE = True
except ImportError:
    AGENTS_SDK_AVAILABLE = False
    SDKAgent = None
    Runner = None
    trace = None
    OpenAIChatCompletionsModel = None
    AgentOutputSchema = None

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
    ):
        self.api_model = api_model
        self.temperature = temperature
        self.max_tokens = max_tokens

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
            return

        self.api_key = api_key or os.environ.get(api_key_env, "")

        if not self.api_key:
            logger.warning(f"{api_key_env} not set - using mock mode")
            self.client = None
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
        result = await self._call_ai(
            system_prompt=system_prompt or "You are a helpful assistant.",
            user_text=prompt,
            output_type=TextResult,
            context=context,
        )
        return {"response": result.response}


    async def chat_with_tools(
        self,
        message: str,
        tools: List[Dict],
        system_prompt: str,
        conversation_history: Optional[List[Dict]] = None,
        context: Optional[CallContext] = None,
        max_iterations: int = 10,
    ) -> str:
        """
        Chat with native LLM tool calling.

        The LLM decides which tools to call. This method executes the tool-call
        loop: send message -> LLM picks tools -> execute via AgentRegistry -> feed
        results back -> repeat until LLM produces a final text response.

        Args:
            message: Current user message
            tools: OpenAI-compatible tool schemas (from AgentRegistry)
            system_prompt: System instruction injected before conversation
            conversation_history: Previous messages as [{"role":..., "content":...}]
            context: Call context for usage attribution
            max_iterations: Max tool-call rounds before giving up (prevents loops)

        Returns:
            Final text response from the LLM.
        """
        if not self.client or self.provider == "mock":
            logger.warning("chat_with_tools: AI in mock mode")
            return "[Mock] Configure API_MODEL for real AI responses."

        import json
        from kk_utils.agent_tools import get_registry

        registry = get_registry()

        # Build initial message list
        messages: List[Dict] = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": message})

        openai_tools = tools if tools else None

        try:
            for iteration in range(max_iterations):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto" if openai_tools else None,
                )

                choice = response.choices[0]

                if choice.finish_reason == "tool_calls":
                    # Append assistant message (with tool_calls)
                    messages.append(choice.message.model_dump(exclude_unset=False))

                    # Execute each tool call
                    for tool_call in choice.message.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            tool_args = json.loads(tool_call.function.arguments or "{}")
                        except json.JSONDecodeError:
                            tool_args = {}

                        logger.debug(f"Tool call: {tool_name}({tool_args})")
                        result = registry.execute(tool_name, **tool_args)
                        logger.debug(f"Tool result [{tool_name}]: {str(result)[:200]}")

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        })

                elif choice.finish_reason == "stop":
                    self._on_usage(response, context, TextResult)
                    return choice.message.content or ""

                else:
                    logger.warning(f"chat_with_tools: unexpected finish_reason={choice.finish_reason!r}")
                    break

            logger.warning("chat_with_tools: max_iterations reached without final response")
            return "I am having trouble processing your request. Please try again."

        except Exception as e:
            logger.error(f"chat_with_tools failed: {e}", exc_info=True)
            return "I encountered an error. Please try again."

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
    # Prompt Builders
    # -------------------------------------------------------------------------

    def _build_summarize_prompt(self, max_length: int, bullet_points: bool) -> str:
        bullet_instruction = (
            "\n- Extract 3-5 key bullet points that capture the main ideas"
            if bullet_points else ""
        )
        return f"""You are an expert summarization assistant. Your task is to create concise, accurate summaries.

GUIDELINES:
- Capture the main ideas and key points
- Maintain the original meaning and tone
- Use clear, concise language
- Keep summary under {max_length} words{bullet_instruction}
- Highlight important facts, conclusions, and action items

OUTPUT FORMAT:
Provide a structured response with:
1. A concise summary paragraph
2. Key bullet points (if requested)
3. Word count of the summary"""

    def _build_rewrite_prompt(self, tone: str, style: Optional[str]) -> str:
        style_instruction = f" and {style} style" if style else ""
        return f"""You are an expert writing assistant. Your task is to rewrite text with a different tone{style_instruction}.

GUIDELINES:
- Maintain the original meaning and key information
- Transform the tone to be {tone}
- Improve clarity and readability
- Fix any grammar or awkward phrasing
- Keep similar length unless style requires otherwise

OUTPUT FORMAT:
Provide a structured response with:
1. The rewritten text
2. The tone/style applied
3. List of major changes made"""

    def _build_extract_tasks_prompt(self) -> str:
        return """You are an expert task extraction assistant. Your task is to identify actionable tasks from text.

GUIDELINES:
- Extract clear, actionable tasks with specific outcomes
- Identify priority levels (high/medium/low) based on urgency indicators
- Extract any due dates or time references
- Capture context or notes for each task
- Group related subtasks together
- Ignore non-actionable information

TASK CRITERIA:
- Must have a clear action verb
- Must have a specific outcome or deliverable
- Must be achievable and measurable

OUTPUT FORMAT:
Provide a structured response with:
1. List of tasks with title, priority, due_date, and context
2. Total count of tasks extracted"""

    def _build_classify_intent_prompt(self) -> str:
        return """You are an expert intent classification assistant. Analyze user input to determine intent and extract entities.

INTENT CATEGORIES:
- digital_me_query: Questions about user's profile, skills, experience
- notes_operation: Creating, updating, searching, or managing notes
- groups_operation: Managing note groups/folders
- task_management: Creating or managing tasks
- information_request: General questions or information seeking
- summarization: Request to summarize text
- rewriting: Request to rewrite/modify text
- casual_chat: General conversation or greetings

AVAILABLE TOOLS (use these EXACT tool IDs in suggested_tools):
- get_digital_me_summary: Get brief profile summary (no parameters needed)
- get_work_experience: Get work experience (parameters: company, search_query)
- get_skills: Get skills (parameters: category, min_proficiency, search_query)
- get_education: Get education (parameters: degree_level, field_of_study)
- get_projects: Get projects (parameters: technology, role, search_query)
- get_certifications: Get certifications (parameters: issuer, include_expired)
- search_digital_me: RAG search in documents (parameters: query, top_k)
- search_notes: Search notes (parameters: query, group_id, limit)
- create_note: Create note (parameters: title, content, group_id)
- update_note: Update note (parameters: note_id, title, content)
- summarize_text: Summarize text (parameters: text, max_length, bullet_points)
- rewrite_text: Rewrite text (parameters: text, tone, style)
- extract_tasks: Extract tasks from text (parameters: text)

GUIDELINES:
- Analyze the user's intent carefully
- Extract relevant entities using the EXACT parameter names shown above
- For digital_me_query tools, ALWAYS extract the user's question as 'search_query' parameter
- For search_digital_me, ALWAYS extract the user's question as 'query' parameter
- Provide confidence score (0.0-1.0)

OUTPUT FORMAT:
Provide a structured response with:
1. Primary intent category
2. Confidence score
3. Extracted entities as key-value pairs
4. Suggested tool IDs (use exact IDs from AVAILABLE TOOLS list)"""

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

            wrapped_output = (
                AgentOutputSchema(output_type, strict_json_schema=False)
                if AgentOutputSchema else output_type
            )

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
