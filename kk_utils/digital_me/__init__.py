"""
kk_utils.digital_me — Digital Me service and tools

Digital Me provides access to personal profile data:
- Work experience
- Skills
- Education
- Projects
- Certifications
- RAG-powered semantic search

Usage:
    from kk_utils.digital_me.service import (
        get_digital_me_summary,
        get_work_experience,
        get_skills,
    )
    
    # Get profile summary
    summary = get_digital_me_summary()
    
    # Get work experience
    experience = get_work_experience(company="Tech Corp")
    
    # Get skills
    skills = get_skills(category="technical")
"""

from .service import (
    get_digital_me_summary,
    get_work_experience,
    get_skills,
    get_education,
    get_projects,
    get_certifications,
)

from .rag import get_digital_me_rag

__all__ = [
    # Service functions
    'get_digital_me_summary',
    'get_work_experience',
    'get_skills',
    'get_education',
    'get_projects',
    'get_certifications',
    # RAG
    'get_digital_me_rag',
]
