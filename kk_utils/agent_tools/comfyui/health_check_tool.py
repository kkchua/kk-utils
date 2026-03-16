"""
ComfyUI Health Check Skill - Check server and workflow health

Endpoints:
- GET /api/conductor/available-services - Check server online + list workflows
- GET /api/conductor/workflow/{category}/{name}/health - Check specific workflow health

Usage:
    result = await comfyui_health_check(
        server_url="http://server:8188",
        check_type="full"
    )
"""

from typing import Dict, Any, Optional, List
import logging
from kk_utils.agent_tools import agent_tool

logger = logging.getLogger(__name__)


@agent_tool(
    name="Check ComfyUI Health",
    description="Check ComfyUI server health and available workflows. Returns online status and list of available workflows.",
    tags=["comfyui", "health", "infrastructure"],
    access_level="user",
    sensitivity="low",
    input_schema={
        "type": "object",
        "properties": {
            "server_url": {
                "type": "string",
                "description": "ComfyUI server URL (e.g., http://server:8188)"
            },
            "check_type": {
                "type": "string",
                "enum": ["server", "workflows", "full"],
                "description": "Type of health check"
            },
            "workflow_name": {
                "type": "string",
                "description": "Specific workflow to check (optional)"
            }
        },
        "required": ["server_url"]
    },
)
async def comfyui_health_check(
    server_url: str,
    check_type: str = "full",
    workflow_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Check ComfyUI server and workflow health.
    
    Args:
        server_url: ComfyUI server URL
        check_type: Type of check (server, workflows, full)
        workflow_name: Specific workflow to check (optional)
    
    Returns:
        {
            "online": bool,
            "server_url": str,
            "latency_ms": float,
            "available_workflows": list,
            "workflow_status": dict (if workflow_name specified),
            "error": str (if failed)
        }
    """
    result = {
        "online": False,
        "server_url": server_url,
        "latency_ms": None,
        "available_workflows": [],
        "error": None,
    }
    
    try:
        # Check server health via /api/conductor/available-services
        if check_type in ("server", "full"):
            server_health = await _check_server_online(server_url)
            result.update(server_health)
            
            # If server is offline, return early
            if not result["online"]:
                return result
        
        # Get available workflows
        if check_type in ("workflows", "full"):
            workflows = await _get_available_workflows(server_url)
            result["available_workflows"] = workflows
        
        # Check specific workflow health
        if workflow_name:
            workflow_status = await _check_workflow_health(server_url, workflow_name)
            result["workflow_status"] = workflow_status
        
        return result
    
    except Exception as e:
        logger.error(f"comfyui_health_check failed: {e}", exc_info=True)
        result["error"] = str(e)
        return result


async def _check_server_online(server_url: str) -> Dict[str, Any]:
    """Check if server is online via /api/conductor/available-services."""
    import aiohttp
    from datetime import datetime
    
    try:
        health_endpoint = f"{server_url}/api/conductor/available-services"
        
        async with aiohttp.ClientSession() as session:
            start = datetime.now()
            async with session.get(health_endpoint, timeout=5) as response:
                elapsed = (datetime.now() - start).total_seconds()
                
                if response.status == 200:
                    return {
                        "online": True,
                        "latency_ms": round(elapsed * 1000, 2),
                    }
                else:
                    return {
                        "online": False,
                        "error": f"HTTP {response.status}",
                    }
    
    except Exception as e:
        return {
            "online": False,
            "error": str(e),
        }


async def _get_available_workflows(server_url: str) -> List[Dict[str, Any]]:
    """Get available workflows from /api/conductor/available-services."""
    import aiohttp
    
    try:
        endpoint = f"{server_url}/api/conductor/available-services"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract workflows from response
                    # Response format: {services: [{name, workflows: [...]}]}
                    workflows = []
                    services = data.get("services", [])
                    
                    for service in services:
                        service_workflows = service.get("workflows", [])
                        for wf in service_workflows:
                            workflows.append({
                                "name": wf.get("name"),
                                "category": wf.get("category"),
                                "description": wf.get("description", ""),
                                "tags": wf.get("tags", []),
                            })
                    
                    return workflows
    
    except Exception as e:
        logger.error(f"Failed to get workflows: {e}")
    
    return []


async def _check_workflow_health(
    server_url: str,
    workflow_name: str,
) -> Dict[str, Any]:
    """Check specific workflow health."""
    import aiohttp
    
    try:
        # Try to get workflow metadata as health indicator
        metadata_endpoint = f"{server_url}/api/conductor/metadata"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(metadata_endpoint, timeout=10) as response:
                if response.status == 200:
                    all_metadata = await response.json()
                    
                    # Find workflow metadata
                    for key, metadata in all_metadata.items():
                        if workflow_name in key or key.endswith(workflow_name):
                            return {
                                "name": workflow_name,
                                "status": "available",
                                "metadata": metadata,
                            }
                    
                    return {
                        "name": workflow_name,
                        "status": "not_found",
                    }
    
    except Exception as e:
        return {
            "name": workflow_name,
            "status": "error",
            "error": str(e),
        }
