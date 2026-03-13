"""
Personal Assistant Backend - Article Generation Service

Core logic for AI-powered article generation using Tavily web search + OpenAI.
Reusable by both REST API endpoints and Agent Skills (research_tools).

Pattern follows ai_service.py — service layer is framework-agnostic.
"""
from __future__ import annotations

import logging
import re
import time
from typing import List, Optional

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Output Schemas (Pydantic Models)
# =============================================================================

class GeneratedArticle(BaseModel):
    """Result of AI article generation."""
    title: str
    slug: str
    content: str          # Full markdown article
    excerpt: str
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    search_results_used: int = 0
    post_id: Optional[str] = None  # DB id after save


# =============================================================================
# Article Generation Service
# =============================================================================

ARTICLE_SYSTEM_PROMPT = """You are a technical writer creating high-quality articles for an AI engineering portfolio blog.

Given a topic and optional research context (web search results), write a well-structured technical article.

The article should follow this structure:
- A clear, descriptive title
- Introduction: what this is and why it matters
- Background / Theory: core concepts and principles
- Design & Architecture: how it is structured or designed
- Implementation: practical details, code patterns, or steps
- Key Learnings: insights, tradeoffs, or lessons
- References: cite any URLs from the research context

Respond ONLY with a valid JSON object in this exact format:
{
  "title": "Full article title",
  "slug": "url-friendly-slug-max-60-chars",
  "excerpt": "One to two sentence summary of the article (max 200 chars)",
  "tags": ["tag1", "tag2", "tag3"],
  "content": "# Title\\n\\n## Introduction\\n\\nContent here...\\n\\n## Background / Theory\\n\\n..."
}

The content field must be a complete markdown article. Use ## for section headers. Escape newlines as \\n in the JSON string."""


class ArticleGenerationService:
    """
    AI-powered article generation service.

    Combines Tavily web search for research with OpenAI GPT to produce
    structured markdown articles saved to the portfolio blog_posts table.

    Usage:
        service = get_article_generation_service()
        with get_db_context() as db:
            article = await service.generate(topic="RAG pipeline design", db_session=db)
    """

    async def generate(
        self,
        topic: str,
        db_session: Session,
        category: Optional[str] = None,
        tone: str = "technical",
        style_hints: Optional[str] = None,
        num_search_results: int = 4,
    ) -> GeneratedArticle:
        """
        Research a topic and generate a structured markdown article.

        Args:
            topic: Article topic / subject
            db_session: Active SQLAlchemy session for saving the draft
            category: Optional category label
            tone: Writing tone — technical | accessible | reference
            style_hints: Optional extra style instructions
            num_search_results: Number of Tavily search results to use

        Returns:
            GeneratedArticle with all fields populated and post saved to DB
        """
        from app.services.web_search_service import get_web_search_service

        start_time = time.time()

        # ── Step 1: Web search for research context ──────────────────────────
        search_service = get_web_search_service()
        search_results = await search_service.search(
            query=topic,
            max_results=num_search_results,
            search_depth="basic",
        )

        research_context = self._build_research_context(search_results.results)
        logger.info(f"Research: {len(search_results.results)} results for '{topic}'")

        # ── Step 2: Build generation prompt ──────────────────────────────────
        user_prompt = self._build_user_prompt(
            topic=topic,
            tone=tone,
            style_hints=style_hints,
            research_context=research_context,
            search_urls=[r.url for r in search_results.results],
        )

        # ── Step 3: Generate article via AIService ────────────────────────────
        from app.services.ai_service import get_ai_service
        ai_service = get_ai_service()
        article = await ai_service.generate_structured(
            prompt=user_prompt,
            system_prompt=ARTICLE_SYSTEM_PROMPT,
            output_type=GeneratedArticle,
        )

        # ── Step 4: Ensure unique slug ────────────────────────────────────────
        slug = self._unique_slug(article.slug, topic, db_session)

        # ── Step 5: Save as draft to DB ───────────────────────────────────────
        from app.models.portfolio import BlogPost
        post = BlogPost(
            title=article.title,
            slug=slug,
            content=article.content,
            excerpt=article.excerpt[:500],
            category=category,
            tags=article.tags,
            status="draft",
            author_id=None,
            metadata_json={
                "generated_by": "ai",
                "model": ai_service.api_model,
                "search_results_used": len(search_results.results),
                "generation_time_s": round(time.time() - start_time, 2),
            },
        )
        db_session.add(post)
        db_session.commit()
        db_session.refresh(post)

        logger.info(f"Article generated and saved: '{post.title}' (slug: {slug})")

        return GeneratedArticle(
            title=post.title,
            slug=post.slug,
            content=post.content,
            excerpt=post.excerpt or "",
            tags=post.tags or [],
            category=post.category,
            search_results_used=len(search_results.results),
            post_id=str(post.id),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_research_context(self, results) -> str:
        """Combine search results into a research context string (max ~4000 chars)."""
        if not results:
            return "No web search results available. Use your training knowledge."

        parts = []
        total = 0
        for i, r in enumerate(results, 1):
            snippet = f"[{i}] {r.title}\nURL: {r.url}\n{r.content[:600]}"
            if total + len(snippet) > 4000:
                break
            parts.append(snippet)
            total += len(snippet)

        return "\n\n---\n\n".join(parts)

    def _build_user_prompt(
        self,
        topic: str,
        tone: str,
        style_hints: Optional[str],
        research_context: str,
        search_urls: List[str],
    ) -> str:
        tone_instruction = {
            "technical": "Write for experienced engineers. Use precise technical language.",
            "accessible": "Write for a mixed audience. Balance technical depth with clarity.",
            "reference": "Write as a structured reference guide. Prioritise completeness.",
        }.get(tone, "Write for experienced engineers.")

        hints = f"\nAdditional style guidance: {style_hints}" if style_hints else ""

        return f"""Topic: {topic}

Tone: {tone_instruction}{hints}

Research context (from web search):
{research_context}

Reference URLs to cite in the References section:
{chr(10).join(f'- {url}' for url in search_urls) if search_urls else '- No URLs available'}

Write a comprehensive technical article on this topic. Follow the JSON format exactly."""

    def _unique_slug(self, candidate: str, topic: str, db_session: Session) -> str:
        """Sanitise slug and ensure it is unique in the database."""
        from app.models.portfolio import BlogPost

        if not candidate:
            candidate = topic

        # Sanitise
        slug = re.sub(r"[^a-z0-9]+", "-", candidate.lower()).strip("-")[:60]

        # Check uniqueness
        base = slug
        counter = 2
        while db_session.query(BlogPost).filter(BlogPost.slug == slug).first():
            slug = f"{base}-{counter}"
            counter += 1

        return slug


# =============================================================================
# Singleton Factory
# =============================================================================

_article_generation_service: Optional[ArticleGenerationService] = None


def get_article_generation_service() -> ArticleGenerationService:
    """Get or create the ArticleGenerationService singleton."""
    global _article_generation_service
    if _article_generation_service is None:
        _article_generation_service = ArticleGenerationService()
    return _article_generation_service
