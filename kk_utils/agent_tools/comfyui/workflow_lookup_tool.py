"""
ComfyUI Workflow Lookup Skill - Get workflow metadata

Endpoint:
- GET /api/conductor/metadata - Get all workflow metadata

Usage:
    result = await comfyui_workflow_lookup(
        server_url="http://server:8188",
        workflow_name="image_to_csv"
    )
"""

from typing import Dict, Any, Optional, List
import logging
from kk_utils.agent_tools import agent_tool

logger = logging.getLogger(__name__)


@agent_tool(
    name="Lookup ComfyUI Workflow",
    description="Get workflow metadata from ComfyUI Conductor. Returns workflow schema, inputs, and configuration.",
    tags=["comfyui", "workflow", "metadata"],
    access_level="user",
    sensitivity="low",
    input_schema={
        "type": "object",
        "properties": {
            "server_url": {
                "type": "string",
                "description": "ComfyUI server URL (e.g., http://server:8188)"
            },
            "workflow_name": {
                "type": "string",
                "description": "Workflow name to lookup"
            },
            "category": {
                "type": "string",
                "description": "Workflow category (optional)"
            }
        },
        "required": ["server_url"]
    },
)
async def comfyui_workflow_lookup(
    server_url: str,
    workflow_name: Optional[str] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get workflow metadata from ComfyUI Conductor.
    
    Args:
        server_url: ComfyUI server URL
        workflow_name: Specific workflow to lookup (optional, returns all if not specified)
        category: Workflow category (optional)
    
    Returns:
        {
            "success": bool,
            "workflows": list (all workflows if no workflow_name specified),
            "workflow": dict (specific workflow if workflow_name specified),
            "error": str (if failed)
        }
    """
    try:
        metadata_endpoint = f"{server_url}/api/conductor/metadata"
        
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(metadata_endpoint, timeout=10) as response:
                if response.status == 200:
                    all_metadata = await response.json()
                    
                    # If specific workflow requested
                    if workflow_name:
                        workflow = _find_workflow_metadata(
                            all_metadata,
                            workflow_name,
                            category,
                        )
                        
                        if workflow:
                            return {
                                "success": True,
                                "workflow": workflow,
                            }
                        else:
                            return {
                                "success": False,
                                "error": f"Workflow '{workflow_name}' not found",
                            }
                    
                    # Return all workflows
                    workflows = _extract_workflows_from_metadata(all_metadata)
                    return {
                        "success": True,
                        "workflows": workflows,
                        "count": len(workflows),
                    }
    
    except Exception as e:
        logger.error(f"comfyui_workflow_lookup failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


def _find_workflow_metadata(
    all_metadata: Dict,
    workflow_name: str,
    category: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Find specific workflow metadata."""
    for key, metadata in all_metadata.items():
        # Match workflow name
        if workflow_name in key or key.endswith(workflow_name):
            # If category specified, check it
            if category:
                wf_category = metadata.get("category", "")
                if wf_category != category:
                    continue
            
            # Add task_key to metadata
            metadata["task_key"] = key
            return metadata
    
    return None


def _extract_workflows_from_metadata(all_metadata: Dict) -> List[Dict[str, Any]]:
    """Extract workflow list from metadata response."""
    workflows = []
    
    for key, metadata in all_metadata.items():
        workflows.append({
            "task_key": key,
            "name": metadata.get("name", key),
            "category": metadata.get("category", ""),
            "description": metadata.get("description", ""),
            "schema": metadata.get("schema", {}),
            "tags": metadata.get("tags", []),
        })
    
    return workflows
