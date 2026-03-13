"""
digital_me_rag — SkillManifest

Digital Me RAG engine. Extends kk_utils RAGEngine with user-scoped security
filtering, document-type auto-detection, and resume/project search helpers.
Used internally by the digital_me skill tools.
"""
from kk_utils.skill_manifest import SkillManifest

SKILL = SkillManifest(
    name="digital_me_rag",
    display_name="Digital Me RAG Engine",
    description=(
        "RAG engine for the Digital Me knowledge base. Provides get_digital_me_rag() "
        "singleton used by the digital_me skill tools. Not a tool provider — "
        "consumed internally by other skills."
    ),
    version="1.0.0",
    tags=["digital_me_rag", "rag", "chromadb"],
    collection="digital_me",
    capabilities=["rag_engine"],
    min_access_level="demo",
)
