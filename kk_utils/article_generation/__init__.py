"""
kk_utils.article_generation — Article Generation service

Research-based article generation:
- Research topic using web search
- Generate structured articles
- Save drafts to portfolio

Usage:
    from kk_utils.article_generation.service import get_article_generation_service
    
    service = get_article_generation_service()
    result = await service.generate(
        topic="AI Trends 2026",
        db_session=db,
        category="Technology",
        tone="technical",
    )
    
    print(f"Created: {result.title} (slug: {result.slug})")
"""

from .service import get_article_generation_service

__all__ = [
    'get_article_generation_service',
]
