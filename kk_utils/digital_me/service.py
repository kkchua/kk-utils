"""
kk_utils.digital_me.service — Digital Me Service

Business logic for optional structured Digital Me data access.

This module is backend-agnostic and can be used in:
- FastAPI backend
- Gradio apps
- Standalone scripts
- Any Python project

Usage:
    from kk_utils.digital_me.service import (
        get_digital_me_summary,
        get_work_experience,
        get_skills,
    )
    
    summary = get_digital_me_summary()
    experience = get_work_experience()
"""
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

# Global cache for Digital Me data
_digital_me_data = None


def _empty_data() -> Dict[str, Any]:
    """Return the empty structured profile shape."""
    return {
        "profile": {},
        "work_experience": [],
        "skills": [],
        "education": [],
        "projects": [],
        "certifications": [],
    }


def _find_config_path() -> Path:
    """
    Find digital_me/profile.yaml config file.
    
    Searches in order:
    1. Current working directory: config/digital_me/profile.yaml
    2. Parent directories (up to 3 levels)
    3. kk-utils parent: ../config/digital_me/profile.yaml
    
    Returns:
        Path to config file or None if not found
    """
    # Try current working directory first
    cwd_path = Path.cwd() / "config" / "digital_me" / "profile.yaml"
    if cwd_path.exists():
        return cwd_path
    
    # Try parent directories (for nested projects)
    current = Path.cwd()
    for _ in range(3):
        parent_path = current.parent / "config" / "digital_me" / "profile.yaml"
        if parent_path.exists():
            return parent_path
        current = current.parent
    
    # Try relative to kk-utils location
    kk_utils_path = Path(__file__).parent.parent.parent / "config" / "digital_me" / "profile.yaml"
    if kk_utils_path.exists():
        return kk_utils_path
    
    return None


def _load_digital_me_data() -> Dict[str, Any]:
    """Load optional structured Digital Me data from YAML config."""
    global _digital_me_data

    if _digital_me_data is not None:
        return _digital_me_data

    # Find config file
    config_path = _find_config_path()

    if config_path and config_path.exists():
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
            if _looks_like_placeholder_data(loaded):
                logger.warning(
                    "Structured Digital Me config at %s contains placeholder data; ignoring it",
                    config_path,
                )
                _digital_me_data = _empty_data()
                return _digital_me_data

            _digital_me_data = loaded
            logger.info(f"Loaded Digital Me data from {config_path}")
            return _digital_me_data
        except Exception as e:
            logger.error(f"Failed to load Digital Me data: {e}")

    _digital_me_data = _empty_data()
    logger.info("Structured Digital Me data not configured")
    return _digital_me_data


def _value_looks_placeholder(value: Any) -> bool:
    """Heuristic check for placeholder/template values."""
    if not isinstance(value, str):
        return False

    stripped = value.strip()
    if not stripped:
        return False

    return (
        (stripped.startswith("[") and stripped.endswith("]"))
        or stripped == "YYYY-MM"
    )


def _looks_like_placeholder_data(data: Dict[str, Any]) -> bool:
    """Detect obviously unedited placeholder structured profile data."""
    profile = data.get("profile") or {}
    work_experience = data.get("work_experience") or []
    education = data.get("education") or []
    projects = data.get("projects") or []
    certifications = data.get("certifications") or []

    placeholder_hits = 0

    for value in (
        profile.get("title"),
        profile.get("summary"),
        profile.get("location"),
        profile.get("email"),
        profile.get("linkedin"),
        profile.get("github"),
        profile.get("website"),
    ):
        if _value_looks_placeholder(value):
            placeholder_hits += 1

    if work_experience:
        first = work_experience[0] or {}
        for value in (
            first.get("company"),
            first.get("position"),
            first.get("start_date"),
            first.get("location"),
            first.get("description"),
        ):
            if _value_looks_placeholder(value):
                placeholder_hits += 1

    if education:
        first = education[0] or {}
        for value in (
            first.get("institution"),
            first.get("degree"),
            first.get("field"),
            first.get("gpa"),
        ):
            if _value_looks_placeholder(value):
                placeholder_hits += 1

    if projects:
        first = projects[0] or {}
        for value in (
            first.get("name"),
            first.get("role"),
            first.get("start_date"),
            first.get("description"),
            first.get("url"),
        ):
            if _value_looks_placeholder(value):
                placeholder_hits += 1

    if certifications:
        first = certifications[0] or {}
        for value in (
            first.get("name"),
            first.get("issuer"),
            first.get("date"),
            first.get("expiry_date"),
            first.get("credential_id"),
            first.get("credential_url"),
        ):
            if _value_looks_placeholder(value):
                placeholder_hits += 1

    return placeholder_hits >= 3


def get_work_experience(company: Optional[str] = None) -> List[Dict]:
    """Get work experience."""
    data = _load_digital_me_data()
    experiences = data.get("work_experience", [])
    
    if company:
        experiences = [
            e for e in experiences
            if company.lower() in e.get("company", "").lower()
        ]
    
    return experiences


def get_skills(
    category: Optional[str] = None,
    min_proficiency: int = 1,
) -> List[Dict]:
    """Get skills."""
    data = _load_digital_me_data()
    skills = data.get("skills", [])
    
    if category:
        skills = [s for s in skills if s.get("category") == category]
    
    if min_proficiency:
        skills = [s for s in skills if s.get("proficiency", 0) >= min_proficiency]
    
    return skills


def get_education(
    degree_level: Optional[str] = None,
    field_of_study: Optional[str] = None,
) -> List[Dict]:
    """Get education history."""
    data = _load_digital_me_data()
    education = data.get("education", [])
    
    if degree_level:
        degree_map = {
            "bachelor": ["Bachelor", "BS", "BA"],
            "master": ["Master", "MS", "MA", "MBA"],
            "phd": ["PhD", "Doctorate"],
        }
        
        allowed = degree_map.get(degree_level.lower(), [degree_level])
        education = [
            e for e in education
            if any(d in e.get("degree", "") for d in allowed)
        ]
    
    if field_of_study:
        education = [
            e for e in education
            if field_of_study.lower() in e.get("field", "").lower()
        ]
    
    return education


def get_projects(
    technology: Optional[str] = None,
    role: Optional[str] = None,
) -> List[Dict]:
    """Get projects."""
    data = _load_digital_me_data()
    projects = data.get("projects", [])
    
    if technology:
        projects = [
            p for p in projects
            if any(technology.lower() in t.lower() for t in p.get("technologies", []))
        ]
    
    if role:
        projects = [
            p for p in projects
            if role.lower() in p.get("role", "").lower()
        ]
    
    return projects


def get_certifications(
    issuer: Optional[str] = None,
    include_expired: bool = False,
) -> List[Dict]:
    """Get certifications."""
    from datetime import datetime
    
    data = _load_digital_me_data()
    certs = data.get("certifications", [])
    
    if issuer:
        certs = [
            c for c in certs
            if issuer.lower() in c.get("issuer", "").lower()
        ]
    
    if not include_expired:
        now = datetime.now()
        filtered = []
        for c in certs:
            expiry = c.get("expiry_date")
            if not expiry:
                filtered.append(c)  # No expiry = valid
            else:
                try:
                    expiry_date = datetime.fromisoformat(expiry)
                    if expiry_date > now:
                        filtered.append(c)
                except ValueError:
                    filtered.append(c)  # Invalid date = keep
        
        certs = filtered
    
    return certs


def get_digital_me_summary() -> Dict[str, Any]:
    """Get Digital Me summary."""
    data = _load_digital_me_data()
    
    profile = data.get("profile", {})
    work_exp = data.get("work_experience", [])
    skills = data.get("skills", [])
    projects = data.get("projects", [])
    certifications = data.get("certifications", [])

    if not any([profile, work_exp, skills, projects, certifications]):
        return {
            "available": False,
            "message": "Structured profile summary information is not available.",
            "name": None,
            "title": None,
            "summary": "",
            "total_experience_years": 0,
            "current_position": None,
            "top_skills": [],
            "total_projects": 0,
            "total_certifications": 0,
        }
    
    # Get top skills
    top_skills = sorted(
        skills,
        key=lambda s: s.get("proficiency", 0),
        reverse=True
    )[:5]
    
    return {
        "name": profile.get("name", "Unknown"),
        "title": profile.get("title", "Unknown"),
        "summary": profile.get("summary", ""),
        "total_experience_years": len(work_exp) * 3,  # Rough estimate
        "current_position": work_exp[0] if work_exp else None,
        "top_skills": top_skills,
        "total_projects": len(projects),
        "total_certifications": len(certifications),
    }
