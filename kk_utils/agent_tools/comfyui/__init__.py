"""
kk_utils.agent_tools.comfyui — ComfyUI Conductor Skills

Skills for interacting with ComfyUI Conductor API:
- comfyui_submit: Submit workflows to ComfyUI
- comfyui_health_check: Check server and workflow health
- comfyui_workflow_lookup: Get workflow metadata

All skills delegate to ComfyUI Conductor endpoints:
- /api/conductor/available-services - Check server online status
- /api/conductor/metadata - Get workflow metadata
- /api/conductor/execute - Submit workflow (submit & forget)
"""

from kk_utils.agent_tools import _auto_register
from .submit_tool import submit_comfyui_workflow
from .health_check_tool import comfyui_health_check
from .workflow_lookup_tool import comfyui_workflow_lookup

__all__ = [
    "submit_comfyui_workflow",
    "comfyui_health_check",
    "comfyui_workflow_lookup",
]

# Auto-register tools when module is imported
_auto_register()
