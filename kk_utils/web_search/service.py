"""
Personal Assistant Backend - Web Search Service

Core search logic using Tavily API.
Reusable by both REST API endpoints and Agent Skills.

Pattern follows ai_service.py — service layer is framework-agnostic.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import httpx
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Output Schemas (Pydantic Models)
# =============================================================================

class SearchResult(BaseModel):
    """A single web search result."""
    title: str = Field(..., description="Page title")
    url: str = Field(..., description="Page URL")
    content: str = Field(..., description="Relevant content snippet")
    score: float = Field(default=0.0, description="Relevance score")


class SearchResults(BaseModel):
    """Collection of web search results."""
    results: List[SearchResult] = Field(default_factory=list)
    total: int = Field(default=0)
    query: str = Field(default="")
    success: bool = Field(default=True)
    error: Optional[str] = None


# =============================================================================
# Web Search Service
# =============================================================================

class WebSearchService:
    """
    Web search service using Tavily API.

    Provides research-quality web search results optimised for LLM consumption.
    Gracefully degrades (returns empty results) when no API key is configured.

    Usage:
        service = get_web_search_service()
        results = await service.search("ChromaDB vector database tutorial")
    """

    TAVILY_URL = "https://api.tavily.com/search"

    def __init__(self):
        self.settings = get_settings()

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",  # basic | advanced
    ) -> SearchResults:
        """
        Search the web using Tavily API.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (1–10)
            search_depth: "basic" (fast) or "advanced" (slower, higher quality)

        Returns:
            SearchResults with results list
        """
        if not self.settings.TAVILY_API_KEY:
            logger.warning("TAVILY_API_KEY not configured — returning empty search results")
            return SearchResults(
                query=query,
                success=False,
                error="Web search not configured (TAVILY_API_KEY missing)",
            )

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    self.TAVILY_URL,
                    json={
                        "api_key": self.settings.TAVILY_API_KEY,
                        "query": query,
                        "max_results": max_results,
                        "search_depth": search_depth,
                        "include_answer": False,
                        "include_raw_content": False,
                    },
                )
                response.raise_for_status()
                data = response.json()

            raw_results = data.get("results", [])
            results = [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                    score=float(r.get("score", 0.0)),
                )
                for r in raw_results
            ]

            # Avoid non-ASCII characters in log message to prevent UnicodeEncodeError
            logger.info(f"Web search: '%s' -> %d results", query, len(results))
            return SearchResults(results=results, total=len(results), query=query)

        except httpx.HTTPStatusError as e:
            logger.error(f"Tavily API HTTP error: {e.response.status_code} — {e.response.text}")
            return SearchResults(
                query=query,
                success=False,
                error=f"Search API error: {e.response.status_code}",
            )
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return SearchResults(query=query, success=False, error=str(e))


# =============================================================================
# Singleton Factory
# =============================================================================

_web_search_service: Optional[WebSearchService] = None


def get_web_search_service() -> WebSearchService:
    """Get or create the WebSearchService singleton."""
    global _web_search_service
    if _web_search_service is None:
        _web_search_service = WebSearchService()
    return _web_search_service
