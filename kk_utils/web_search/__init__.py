"""
kk_utils.web_search — Web Search service

Tavily-powered web search for real-time information retrieval:
- Search the web for current information
- Returns titles, URLs, and content snippets
- Basic and advanced search depths

Usage:
    from kk_utils.web_search.service import get_web_search_service
    
    service = get_web_search_service()
    result = await service.search(query="AI trends 2026", max_results=5)
    
    print(f"Found {result.total} results")
    for r in result.results:
        print(f"- {r.title}: {r.url}")
"""

from .service import get_web_search_service

__all__ = [
    'get_web_search_service',
]
