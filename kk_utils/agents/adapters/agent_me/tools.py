"""
kk_utils.agents.adapters.agent_me.tools — AgentMe Tools

Digital Me tools for the AgentMe adapter.
Uses RAG for knowledge base search and structured data for fallback.

Tools:
  - search_digital_me: Search Digital Me knowledge base using RAG
  - get_work_experience: Get work experience (RAG first, structured fallback)
  - get_skills: Get skills (RAG first, structured fallback)
  - get_education: Get education (RAG first, structured fallback)
  - get_projects: Get projects (RAG first, structured fallback)
  - get_certifications: Get certifications (RAG first, structured fallback)
  - get_digital_me_summary: Get public-friendly summary
"""

from typing import Optional, List, Dict, Any
import logging
from kk_utils.agent_tools import agent_tool

logger = logging.getLogger(__name__)


@agent_tool(
    name="Search Digital Me Knowledge",
    description="Search Digital Me knowledge base using RAG (resume, projects, documents)",
    tags=["digital_me", "rag", "search"],
    access_level="user",
    sensitivity="medium",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language question or search query"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 3,
                "minimum": 1,
                "maximum": 10
            },
            "source_type": {
                "type": "string",
                "enum": ["resume", "projects", "skills", "all"],
                "description": "Filter by document type"
            }
        },
        "required": ["query"]
    },
)
def search_digital_me(
    query: str,
    top_k: int = 3,
    source_type: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """
    Search Digital Me knowledge base using RAG.

    Args:
        query: Natural language question
        top_k: Number of chunks to retrieve
        source_type: Optional filter (resume, projects, skills, or all)
        user_id: User ID for access control

    Returns:
        dict with chunks, confidence, sources
    """
    from kk_utils.rag.rag_engine import RAGEngine

    rag = RAGEngine(collection_name="digital_me")

    filter_metadata = {}
    if source_type and source_type != "all":
        filter_metadata["type"] = source_type

    result = rag.query(
        question=query,
        top_k=top_k * 2,
        filter_metadata=filter_metadata,
        min_confidence=0.1,
    )

    # Sanitize chunks (remove user_id and sensitive metadata)
    sanitized_chunks = []
    for chunk in result.chunks if result.has_results else []:
        sanitized_chunk = {
            "content": chunk.get("content", ""),
            "metadata": {
                k: v for k, v in chunk.get("metadata", {}).items()
                if k not in ["user_id", "access_level"]
            },
        }
        sanitized_chunks.append(sanitized_chunk)

    return {
        "query": query,
        "chunks": sanitized_chunks[:top_k],
        "confidence": result.confidence if result.has_results else 0.0,
        "sources": result.sources if result.has_results else [],
        "security_filter_applied": True,
        "filtered_count": len(result.chunks if result.has_results else []) - len(sanitized_chunks),
        "message": result.message,
        "retrieval_time_ms": result.retrieval_time_ms,
        "chunks_searched": result.chunks_searched,
        "avg_distance": result.avg_distance,
    }


@agent_tool(
    name="Get Work Experience",
    description="Get the person's work experience and employment history from their Digital Me profile",
    tags=["digital_me", "experience", "resume"],
    access_level="demo",
    sensitivity="low",
)
def get_work_experience(
    company: Optional[str] = None,
    search_query: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """
    Get work experience — RAG first, structured fallback.

    Args:
        company: Filter by company name
        search_query: Natural language query (uses RAG)
        user_id: User ID (auto-injected)

    Returns:
        dict with experiences or RAG chunks
    """
    # Try RAG first
    rag_query = search_query or (f"work experience at {company}" if company else "work experience and employment history")
    rag_result = search_digital_me(query=rag_query, top_k=5, source_type=None, user_id=user_id)

    if rag_result.get("confidence", 0.0) > 0.1:
        return {
            "source": "rag",
            "confidence": rag_result["confidence"],
            "chunks": rag_result["chunks"],
            "sources": rag_result["sources"],
        }

    # Fallback to structured data
    from kk_utils.digital_me.service import get_work_experience as get_work_exp_svc
    experiences = get_work_exp_svc(company=company)
    if not experiences:
        return {"available": False, "message": "Work experience information is not available in my profile yet."}
    return {"source": "structured", "experiences": experiences, "count": len(experiences)}


@agent_tool(
    name="Get Skills",
    description="Get the person's technical and professional skills from their Digital Me profile",
    tags=["digital_me", "skills"],
    access_level="demo",
    sensitivity="low",
)
def get_skills(
    category: Optional[str] = None,
    min_proficiency: int = 1,
    search_query: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """
    Get skills — RAG first, structured fallback.

    Args:
        category: Filter by category (technical, soft, languages)
        min_proficiency: Minimum proficiency level (1-5)
        search_query: Natural language query (uses RAG)
        user_id: User ID (auto-injected)

    Returns:
        dict with skills or RAG chunks
    """
    # Try RAG first
    rag_query = search_query or (f"{category} skills" if category else "technical skills and expertise")
    rag_result = search_digital_me(query=rag_query, top_k=5, source_type=None, user_id=user_id)

    if rag_result.get("confidence", 0.0) > 0.1:
        return {"source": "rag", "confidence": rag_result["confidence"], "chunks": rag_result["chunks"]}

    # Fallback to structured data
    from kk_utils.digital_me.service import get_skills as get_skills_svc
    skills = get_skills_svc(category=category, min_proficiency=min_proficiency)
    if not skills:
        return {"available": False, "message": "Skills information is not available in my profile yet."}
    return {"source": "structured", "skills": skills, "count": len(skills)}


@agent_tool(
    name="Get Education",
    description="Get the person's education history (university, degree, field of study) from their Digital Me profile",
    tags=["digital_me", "education", "resume"],
    access_level="demo",
    sensitivity="low",
)
def get_education(
    degree_level: Optional[str] = None,
    field_of_study: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """
    Get education history — RAG first, structured fallback.

    Args:
        degree_level: Filter by degree (bachelor, master, phd)
        field_of_study: Filter by field
        user_id: User ID (auto-injected)

    Returns:
        dict with education or RAG chunks
    """
    # Try RAG first
    rag_query = " ".join(filter(None, ["education academic background university degree", degree_level, field_of_study]))
    rag_result = search_digital_me(query=rag_query, top_k=5, source_type=None, user_id=user_id)

    if rag_result.get("confidence", 0.0) > 0.1:
        return {"source": "rag", "confidence": rag_result["confidence"], "chunks": rag_result["chunks"]}

    # Fallback to structured data
    from kk_utils.digital_me.service import get_education_service as get_edu_svc
    education = get_edu_svc(degree_level=degree_level, field_of_study=field_of_study)
    if not education:
        return {"available": False, "message": "Education information is not available in my profile yet."}
    return {"source": "structured", "education": education, "count": len(education)}


@agent_tool(
    name="Get Projects",
    description="Get the person's projects and technical work from their Digital Me profile",
    tags=["digital_me", "projects"],
    access_level="demo",
    sensitivity="low",
)
def get_projects(
    technology: Optional[str] = None,
    role: Optional[str] = None,
    search_query: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """
    Get projects — RAG first, structured fallback.

    Args:
        technology: Filter by technology
        role: Filter by role
        search_query: Natural language query (uses RAG)
        user_id: User ID (auto-injected)

    Returns:
        dict with projects or RAG chunks
    """
    # Try RAG first
    rag_query = search_query or (
        f"{technology} projects" if technology else
        f"{role} role projects" if role else
        "projects and accomplishments"
    )
    rag_result = search_digital_me(query=rag_query, top_k=5, source_type=None, user_id=user_id)

    if rag_result.get("confidence", 0.0) > 0.1:
        return {"source": "rag", "confidence": rag_result["confidence"], "chunks": rag_result["chunks"]}

    # Fallback to structured data
    from kk_utils.digital_me.service import get_projects_service as get_proj_svc
    projects = get_proj_svc(technology=technology, role=role)
    if not projects:
        return {"available": False, "message": "Project information is not available in my profile yet."}
    return {"source": "structured", "projects": projects, "count": len(projects)}


@agent_tool(
    name="Get Certifications",
    description="Get the person's professional certifications and credentials from their Digital Me profile",
    tags=["digital_me", "certifications", "resume"],
    access_level="demo",
    sensitivity="low",
)
def get_certifications(
    issuer: Optional[str] = None,
    include_expired: bool = False,
    user_id: Optional[str] = None,
) -> dict:
    """
    Get certifications — RAG first, structured fallback.

    Args:
        issuer: Filter by issuer
        include_expired: Include expired certifications
        user_id: User ID (auto-injected)

    Returns:
        dict with certifications or RAG chunks
    """
    # Try RAG first
    rag_query = " ".join(filter(None, ["professional certifications credentials qualifications", issuer]))
    rag_result = search_digital_me(query=rag_query, top_k=5, source_type=None, user_id=user_id)

    if rag_result.get("confidence", 0.0) > 0.1:
        return {"source": "rag", "confidence": rag_result["confidence"], "chunks": rag_result["chunks"]}

    # Fallback to structured data
    from kk_utils.digital_me.service import get_certifications_service as get_cert_svc
    certs = get_cert_svc(issuer=issuer, include_expired=include_expired)
    if not certs:
        return {"available": False, "message": "Certification information is not available in my profile yet."}
    return {"source": "structured", "certifications": certs, "count": len(certs)}


@agent_tool(
    name="Get Digital Me Summary",
    description="Get a brief overview of the person's Digital Me profile (name, title, top skills)",
    tags=["digital_me", "summary"],
    access_level="anonymous",
    sensitivity="low",
)
def get_digital_me_summary(user_id: Optional[str] = None) -> dict:
    """
    Get public-friendly Digital Me summary.

    Args:
        user_id: User ID (auto-injected)

    Returns:
        dict with profile summary
    """
    from kk_utils.digital_me.service import get_digital_me_summary_service
    return get_digital_me_summary_service()


# Auto-register tools when module is imported
# The @agent_tool decorator handles registration automatically
from kk_utils.agent_tools import _auto_register
_auto_register()
