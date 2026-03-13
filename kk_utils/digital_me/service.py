"""
kk_utils.digital_me.service — Digital Me Service

Business logic for Digital Me structured data access.
Data loaded from config/digital_me/profile.yaml

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
    """Load Digital Me data from YAML config."""
    global _digital_me_data

    if _digital_me_data is not None:
        return _digital_me_data

    # Find config file
    config_path = _find_config_path()

    if config_path and config_path.exists():
        try:
            _digital_me_data = yaml.safe_load(config_path.read_text(encoding='utf-8'))
            logger.info(f"Loaded Digital Me data from {config_path}")
            return _digital_me_data
        except Exception as e:
            logger.error(f"Failed to load Digital Me data: {e}")

    # Return sample data if file not found
    _digital_me_data = _get_sample_data()
    logger.info("Using sample Digital Me data")
    return _digital_me_data


def _get_sample_data() -> Dict[str, Any]:
    """Get sample Digital Me data."""
    return {
        "profile": {
            "name": "John Doe",
            "title": "Senior Software Engineer",
            "summary": "Experienced software engineer with 10+ years in full-stack development, specializing in Python, JavaScript, and cloud architectures.",
            "location": "San Francisco, CA",
            "email": "john.doe@example.com",
            "linkedin": "linkedin.com/in/johndoe",
            "github": "github.com/johndoe",
        },
        "work_experience": [
            {
                "company": "Tech Corp",
                "position": "Senior Software Engineer",
                "start_date": "2020-01",
                "end_date": "present",
                "location": "San Francisco, CA",
                "description": "Led development of scalable microservices architecture",
                "achievements": [
                    "Reduced API latency by 40% through optimization",
                    "Mentored 5 junior developers",
                    "Led migration from monolith to microservices",
                ],
                "technologies": ["Python", "FastAPI", "Kubernetes", "AWS"],
            },
            {
                "company": "StartupXYZ",
                "position": "Full Stack Developer",
                "start_date": "2017-03",
                "end_date": "2019-12",
                "location": "San Francisco, CA",
                "description": "Built scalable web applications for e-commerce platform",
                "achievements": [
                    "Developed React-based frontend used by 100K+ users",
                    "Implemented CI/CD pipeline reducing deployment time by 60%",
                ],
                "technologies": ["JavaScript", "React", "Node.js", "PostgreSQL"],
            },
        ],
        "skills": [
            {"name": "Python", "category": "technical", "proficiency": 5, "years": 10},
            {"name": "JavaScript", "category": "technical", "proficiency": 4, "years": 8},
            {"name": "FastAPI", "category": "technical", "proficiency": 5, "years": 4},
            {"name": "React", "category": "technical", "proficiency": 4, "years": 6},
            {"name": "AWS", "category": "technical", "proficiency": 4, "years": 7},
            {"name": "Kubernetes", "category": "technical", "proficiency": 3, "years": 3},
            {"name": "Leadership", "category": "soft", "proficiency": 4, "years": 5},
            {"name": "Communication", "category": "soft", "proficiency": 5, "years": 10},
        ],
        "education": [
            {
                "institution": "University of California, Berkeley",
                "degree": "Bachelor of Science",
                "field": "Computer Science",
                "graduation_year": 2016,
                "gpa": "3.8",
            }
        ],
        "projects": [
            {
                "name": "E-commerce Platform",
                "role": "Lead Developer",
                "technologies": ["Python", "Django", "React", "PostgreSQL"],
                "description": "Built scalable e-commerce platform handling $1M+ monthly transactions",
                "url": "https://example.com",
            },
            {
                "name": "Real-time Analytics Dashboard",
                "role": "Full Stack Developer",
                "technologies": ["React", "D3.js", "WebSocket", "Redis"],
                "description": "Created real-time analytics dashboard for business intelligence",
            },
        ],
        "certifications": [
            {
                "name": "AWS Solutions Architect Professional",
                "issuer": "Amazon Web Services",
                "date": "2022-06",
                "expiry_date": "2025-06",
                "credential_id": "AWS-PSA-12345",
            },
            {
                "name": "Certified Kubernetes Administrator",
                "issuer": "Cloud Native Computing Foundation",
                "date": "2021-03",
                "expiry_date": "2024-03",
                "credential_id": "CKA-67890",
            },
        ],
    }


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


def get_education_service(
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


def get_projects_service(
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


def get_certifications_service(
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


def get_digital_me_summary_service() -> Dict[str, Any]:
    """Get Digital Me summary."""
    data = _load_digital_me_data()
    
    profile = data.get("profile", {})
    work_exp = data.get("work_experience", [])
    skills = data.get("skills", [])
    
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
        "total_projects": len(data.get("projects", [])),
        "total_certifications": len(data.get("certifications", [])),
    }
