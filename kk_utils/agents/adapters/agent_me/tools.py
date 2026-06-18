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
from kk_utils.execution_trace import emit_trace

logger = logging.getLogger(__name__)
_RUNTIME_PARAMETERS = ["user_id", "persona_collection"]


def _trace(message: str) -> None:
    emit_trace(f"agent_me.{message}")


def _rag_tool_result(rag_result: dict) -> dict:
    """Return a consistent RAG payload for all Digital Me profile tools."""
    return {
        "source": "rag",
        "confidence": rag_result["confidence"],
        "chunks": rag_result["chunks"],
        "sources": rag_result.get("sources", []),
        "retrieval_time_ms": rag_result.get("retrieval_time_ms", 0.0),
        "chunks_searched": rag_result.get("chunks_searched", 0),
        "avg_distance": rag_result.get("avg_distance", 0.0),
        "collection_name": rag_result.get("collection_name"),
    }


@agent_tool(
    name="Search Digital Me Knowledge",
    description="Search Digital Me knowledge base using RAG (resume, projects, documents)",
    tags=["digital_me", "rag", "search"],
    access_level="user",
    sensitivity="medium",
    runtime_parameters=_RUNTIME_PARAMETERS,
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
    persona_collection: Optional[str] = None,
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
    _trace(
        f"search_digital_me start query={query!r} top_k={top_k} source_type={source_type!r} "
        f"user_id={user_id!r} persona_collection={persona_collection!r}"
    )
    from kk_utils.rag.rag_engine import RAGEngine

    filter_metadata = {}
    if source_type and source_type != "all":
        filter_metadata["type"] = source_type

    def _run_query(collection_name: str) -> dict:
        rag = RAGEngine(collection_name=collection_name)
        result = rag.query(
            question=query,
            top_k=top_k * 2,
            filter_metadata=filter_metadata,
            min_confidence=0.1,
        )

        sanitized_chunks = []
        seen_content = set()
        for chunk in result.chunks if result.has_results else []:
            content = chunk.get("text", chunk.get("content", ""))
            content_key = " ".join(content.split())
            if not content_key or content_key in seen_content:
                continue
            seen_content.add(content_key)
            sanitized_chunk = {
                # RAGEngine.query() returns chunk text under "text".
                "content": content,
                "relevance_score": chunk.get("relevance_score", 0.0),
                "distance": chunk.get("distance", 0.0),
                "metadata": {
                    k: v for k, v in chunk.get("metadata", {}).items()
                    if k not in ["user_id", "access_level"]
                },
            }
            sanitized_chunks.append(sanitized_chunk)

        selected_chunks = sanitized_chunks[:top_k]
        selected_scores = [
            chunk["relevance_score"]
            for chunk in selected_chunks
            if isinstance(chunk.get("relevance_score"), (int, float))
        ]
        selected_confidence = (
            sum(selected_scores) / len(selected_scores)
            if selected_scores else 0.0
        )

        return {
            "query": query,
            "chunks": selected_chunks,
            "confidence": round(selected_confidence, 6),
            "sources": result.sources if result.has_results else [],
            "security_filter_applied": False,
            "search_scope": "collection",
            "filtered_count": len(result.chunks if result.has_results else []) - len(sanitized_chunks),
            "message": result.message,
            "retrieval_time_ms": result.retrieval_time_ms,
            "chunks_searched": result.chunks_searched,
            "avg_distance": result.avg_distance,
            "collection_name": collection_name,
        }

    preferred_collection = persona_collection or "digital_me"
    output = _run_query(preferred_collection)

    if output["confidence"] <= 0.1 and preferred_collection != "digital_me":
        _trace(
            f"search_digital_me fallback to shared collection after {preferred_collection!r}"
        )
        fallback_output = _run_query("digital_me")
        if fallback_output["confidence"] >= output["confidence"]:
            output = fallback_output
        else:
            output["fallback_collection_name"] = "digital_me"
            output["fallback_confidence"] = fallback_output["confidence"]

    _trace(
        "search_digital_me done "
        f"collection={output.get('collection_name')!r} "
        f"confidence={output['confidence']:.3f} "
        f"chunks={len(output['chunks'])} "
        f"time_ms={output['retrieval_time_ms']:.0f}"
    )
    return output


@agent_tool(
    name="Get Work Experience",
    description="Get the person's work experience and employment history from their Digital Me profile",
    tags=["digital_me", "experience", "resume"],
    access_level="demo",
    sensitivity="low",
    runtime_parameters=_RUNTIME_PARAMETERS,
)
def get_work_experience(
    company: Optional[str] = None,
    search_query: Optional[str] = None,
    user_id: Optional[str] = None,
    persona_collection: Optional[str] = None,
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
    _trace("get_work_experience start")
    # Try RAG first
    rag_query = search_query or (f"work experience at {company}" if company else "work experience and employment history")
    rag_result = search_digital_me(
        query=rag_query,
        top_k=5,
        source_type=None,
        user_id=user_id,
        persona_collection=persona_collection,
    )

    if rag_result.get("confidence", 0.0) > 0.1:
        _trace("get_work_experience using RAG")
        return _rag_tool_result(rag_result)

    # Fallback to structured data
    from kk_utils.digital_me.service import get_work_experience as get_work_exp_svc
    experiences = get_work_exp_svc(company=company)
    if not experiences:
        _trace("get_work_experience no structured data")
        return {"available": False, "message": "Work experience information is not available in my profile yet."}
    _trace(f"get_work_experience structured count={len(experiences)}")
    return {"source": "structured", "experiences": experiences, "count": len(experiences)}


@agent_tool(
    name="Get Skills",
    description="Get the person's technical and professional skills from their Digital Me profile",
    tags=["digital_me", "skills"],
    access_level="demo",
    sensitivity="low",
    runtime_parameters=_RUNTIME_PARAMETERS,
)
def get_skills(
    category: Optional[str] = None,
    min_proficiency: int = 1,
    search_query: Optional[str] = None,
    user_id: Optional[str] = None,
    persona_collection: Optional[str] = None,
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
    _trace("get_skills start")
    # Try RAG first
    rag_query = search_query or (f"{category} skills" if category else "technical skills and expertise")
    rag_result = search_digital_me(
        query=rag_query,
        top_k=5,
        source_type=None,
        user_id=user_id,
        persona_collection=persona_collection,
    )

    if rag_result.get("confidence", 0.0) > 0.1:
        _trace("get_skills using RAG")
        return _rag_tool_result(rag_result)

    # Fallback to structured data
    from kk_utils.digital_me.service import get_skills as get_skills_svc
    skills = get_skills_svc(category=category, min_proficiency=min_proficiency)
    if not skills:
        _trace("get_skills no structured data")
        return {"available": False, "message": "Skills information is not available in my profile yet."}
    _trace(f"get_skills structured count={len(skills)}")
    return {"source": "structured", "skills": skills, "count": len(skills)}


@agent_tool(
    name="Get Education",
    description="Get the person's education history (university, degree, field of study) from their Digital Me profile",
    tags=["digital_me", "education", "resume"],
    access_level="demo",
    sensitivity="low",
    runtime_parameters=_RUNTIME_PARAMETERS,
)
def get_education(
    degree_level: Optional[str] = None,
    field_of_study: Optional[str] = None,
    user_id: Optional[str] = None,
    persona_collection: Optional[str] = None,
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
    _trace("get_education start")
    # Try RAG first
    rag_query = " ".join(filter(None, ["education academic background university degree", degree_level, field_of_study]))
    rag_result = search_digital_me(
        query=rag_query,
        top_k=5,
        source_type=None,
        user_id=user_id,
        persona_collection=persona_collection,
    )

    if rag_result.get("confidence", 0.0) > 0.1:
        _trace("get_education using RAG")
        return _rag_tool_result(rag_result)

    # Fallback to structured data
    from kk_utils.digital_me.service import get_education as get_edu_svc
    education = get_edu_svc(degree_level=degree_level, field_of_study=field_of_study)
    if not education:
        _trace("get_education no structured data")
        return {"available": False, "message": "Education information is not available in my profile yet."}
    _trace(f"get_education structured count={len(education)}")
    return {"source": "structured", "education": education, "count": len(education)}


@agent_tool(
    name="Get Projects",
    description="Get the person's projects and technical work from their Digital Me profile",
    tags=["digital_me", "projects"],
    access_level="demo",
    sensitivity="low",
    runtime_parameters=_RUNTIME_PARAMETERS,
)
def get_projects(
    technology: Optional[str] = None,
    role: Optional[str] = None,
    search_query: Optional[str] = None,
    user_id: Optional[str] = None,
    persona_collection: Optional[str] = None,
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
    _trace("get_projects start")
    # Try RAG first
    rag_query = search_query or (
        f"{technology} projects" if technology else
        f"{role} role projects" if role else
        "projects and accomplishments"
    )
    rag_result = search_digital_me(
        query=rag_query,
        top_k=5,
        source_type=None,
        user_id=user_id,
        persona_collection=persona_collection,
    )

    if rag_result.get("confidence", 0.0) > 0.1:
        _trace("get_projects using RAG")
        return _rag_tool_result(rag_result)

    # Fallback to structured data
    from kk_utils.digital_me.service import get_projects as get_proj_svc
    projects = get_proj_svc(technology=technology, role=role)
    if not projects:
        _trace("get_projects no structured data")
        return {"available": False, "message": "Project information is not available in my profile yet."}
    _trace(f"get_projects structured count={len(projects)}")
    return {"source": "structured", "projects": projects, "count": len(projects)}


@agent_tool(
    name="Get Certifications",
    description="Get the person's professional certifications and credentials from their Digital Me profile",
    tags=["digital_me", "certifications", "resume"],
    access_level="demo",
    sensitivity="low",
    runtime_parameters=_RUNTIME_PARAMETERS,
)
def get_certifications(
    issuer: Optional[str] = None,
    include_expired: bool = False,
    user_id: Optional[str] = None,
    persona_collection: Optional[str] = None,
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
    _trace("get_certifications start")
    # Try RAG first
    rag_query = " ".join(filter(None, ["professional certifications credentials qualifications", issuer]))
    rag_result = search_digital_me(
        query=rag_query,
        top_k=5,
        source_type=None,
        user_id=user_id,
        persona_collection=persona_collection,
    )

    if rag_result.get("confidence", 0.0) > 0.1:
        _trace("get_certifications using RAG")
        return _rag_tool_result(rag_result)

    # Fallback to structured data
    from kk_utils.digital_me.service import get_certifications as get_cert_svc
    certs = get_cert_svc(issuer=issuer, include_expired=include_expired)
    if not certs:
        _trace("get_certifications no structured data")
        return {"available": False, "message": "Certification information is not available in my profile yet."}
    _trace(f"get_certifications structured count={len(certs)}")
    return {"source": "structured", "certifications": certs, "count": len(certs)}


@agent_tool(
    name="Get Digital Me Summary",
    description="Get a brief overview of the person's Digital Me profile (name, title, top skills)",
    tags=["digital_me", "summary"],
    access_level="anonymous",
    sensitivity="low",
    runtime_parameters=_RUNTIME_PARAMETERS,
)
def get_digital_me_summary(
    user_id: Optional[str] = None,
    persona_collection: Optional[str] = None,
) -> dict:
    """
    Get public-friendly Digital Me summary.

    Args:
        user_id: User ID (auto-injected)

    Returns:
        dict with profile summary
    """
    _trace("get_digital_me_summary start")
    from kk_utils.digital_me.service import get_digital_me_summary as get_summary_svc
    result = get_summary_svc()
    _trace("get_digital_me_summary done")
    return result


# Auto-register tools when module is imported
# The @agent_tool decorator handles registration automatically
from kk_utils.agent_tools import _auto_register
_auto_register()
