"""
ComfyUI Submit Skill - Submit workflows to ComfyUI Conductor

Endpoint: POST /api/conductor/execute

Usage:
    result = await submit_comfyui_workflow(
        server_url="http://server:8188",
        workflow_name="image_to_csv",
        inputs={"prompt": "generate image", "steps": 20},
        user_id="user123"
    )
"""

from typing import Dict, Any, Optional
import logging
from kk_utils.agent_tools import agent_tool

logger = logging.getLogger(__name__)


@agent_tool(
    name="Submit ComfyUI Workflow",
    description="Submit a workflow to ComfyUI Conductor for execution (submit & forget). Results are saved to Google Drive.",
    tags=["comfyui", "execution", "submit"],
    access_level="user",
    sensitivity="medium",
    input_schema={
        "type": "object",
        "properties": {
            "server_url": {
                "type": "string",
                "description": "ComfyUI server URL (e.g., http://server:8188)"
            },
            "workflow_name": {
                "type": "string",
                "description": "Workflow name (e.g., image_to_csv, video_generator)"
            },
            "inputs": {
                "type": "object",
                "description": "Workflow input parameters"
            },
            "user_id": {
                "type": "string",
                "description": "User ID for tracking"
            }
        },
        "required": ["server_url", "workflow_name", "inputs"]
    },
)
async def submit_comfyui_workflow(
    server_url: str,
    workflow_name: str,
    inputs: Dict[str, Any],
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Submit a workflow to ComfyUI Conductor.
    
    Args:
        server_url: ComfyUI server URL
        workflow_name: Name of workflow to execute
        inputs: Workflow input parameters
        user_id: User ID for tracking
    
    Returns:
        {
            "success": bool,
            "task_id": str (prompt_id),
            "server_url": str,
            "workflow_name": str,
            "message": str,
            "error": str (if failed)
        }
    """
    import aiohttp
    
    try:
        # Step 1: Get workflow metadata
        metadata = await _get_workflow_metadata(server_url, workflow_name)
        if not metadata:
            return {
                "success": False,
                "error": f"Workflow metadata not found: {workflow_name}",
            }
        
        # Step 2: Build submission payload
        task_key = metadata.get("task_key", workflow_name)
        submission_data = {
            "task_key": task_key,
            "inputs": inputs,
            "metadata": metadata,
            "client_id": user_id or "anonymous",
        }
        
        # Step 3: Submit to ComfyUI
        execute_endpoint = f"{server_url}/api/conductor/execute"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                execute_endpoint,
                json=submission_data,
                timeout=30,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "task_id": result.get("prompt_id"),
                        "server_url": server_url,
                        "workflow_name": workflow_name,
                        "message": (
                            f"Workflow submitted successfully. "
                            f"Task ID: {result.get('prompt_id')}. "
                            f"Results will be saved to Google Drive."
                        ),
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}",
                    }
    
    except Exception as e:
        logger.error(f"submit_comfyui_workflow failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


async def _get_workflow_metadata(
    server_url: str,
    workflow_name: str,
) -> Optional[Dict[str, Any]]:
    """Get workflow metadata from /api/conductor/metadata."""
    import aiohttp
    
    try:
        metadata_endpoint = f"{server_url}/api/conductor/metadata"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(metadata_endpoint, timeout=10) as response:
                if response.status == 200:
                    all_metadata = await response.json()
                    
                    # Find specific workflow metadata
                    for key, metadata in all_metadata.items():
                        if workflow_name in key or key.endswith(workflow_name):
                            return metadata
                    
                    # Try exact match
                    return all_metadata.get(workflow_name)
    
    except Exception as e:
        logger.error(f"Failed to get workflow metadata: {e}")
    
    return None
