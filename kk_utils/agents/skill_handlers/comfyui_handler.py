"""
kk_utils.agents.skill_handlers.comfyui_handler — ComfyUI submission handler

Handles ComfyUI skills: submit & forget with Google Drive servers.json

Flow:
1. Load servers.json from Google Drive
2. Filter online servers with available workflows
3. User selects workflow → Get server URL
4. Check server health
5. Submit to ComfyUI /api/conductor/execute
6. Return task_id (submit & forget)
7. Results saved to Google Drive by ComfyUI

Usage:
    handler = ComfyUISkillHandler()
    result = await handler.handle("comfyui_submit", tool_call, context)
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_handler import BaseSkillHandler, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class ComfyUISkillHandler(BaseSkillHandler):
    """
    ComfyUI skill handler - submit & forget.
    
    Features:
    - Google Drive servers.json integration
    - Server health checking
    - Workflow metadata lookup
    - Submit & forget (results to Google Drive)
    """
    
    handler_type = "comfyui"
    
    def __init__(self):
        self._servers_cache: Optional[Dict] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds = 300  # Cache for 5 minutes
    
    async def handle(
        self,
        skill_name: str,
        tool_call: Dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        """
        Execute ComfyUI skill submission.
        
        Args:
            skill_name: Name of skill (e.g., "comfyui_submit")
            tool_call: Tool call from AI {name, arguments}
            context: Execution context
        
        Returns:
            SkillResult with task_id and submission status
        """
        try:
            # 1. Get workflow name from tool call
            workflow_name = tool_call.get("arguments", {}).get("workflow_name")
            if not workflow_name:
                return SkillResult(
                    output={"error": "workflow_name is required"},
                    success=False,
                    error="workflow_name is required",
                )
            
            # 2. Find online server with this workflow
            server = await self._find_server_for_workflow(workflow_name)
            if not server:
                return SkillResult(
                    output={
                        "error": f"No online server found with workflow: {workflow_name}",
                        "available_servers": await self._get_online_servers_with_workflows(),
                    },
                    success=False,
                    error=f"No online server found with workflow: {workflow_name}",
                )
            
            logger.info(f"Found server for workflow '{workflow_name}': {server['name']} → {server['url']}")
            
            # 3. Check server health
            health = await self._check_server_health(server["url"])
            if not health.get("online"):
                return SkillResult(
                    output={
                        "error": f"Server {server['name']} is offline",
                        "health": health,
                    },
                    success=False,
                    error=f"Server {server['name']} is offline",
                )
            
            # 4. Get workflow metadata
            metadata = await self._get_workflow_metadata(server["url"], workflow_name)
            if not metadata:
                return SkillResult(
                    output={"error": f"Workflow metadata not found: {workflow_name}"},
                    success=False,
                    error=f"Workflow metadata not found: {workflow_name}",
                )
            
            # 5. Build submission payload
            submission_data = {
                "task_key": workflow_name,
                "inputs": tool_call.get("arguments", {}),
                "metadata": metadata,
                "client_id": context.user_id,
            }
            
            # 6. Submit to ComfyUI
            submit_result = await self._submit_to_comfyui(server["url"], submission_data)
            
            if not submit_result.get("success"):
                return SkillResult(
                    output={
                        "error": "Submission failed",
                        "details": submit_result,
                    },
                    success=False,
                    error="Submission failed",
                )
            
            # 7. Return task_id (submit & forget)
            return SkillResult(
                output={
                    "task_id": submit_result.get("prompt_id"),
                    "status": "submitted",
                    "server": server["name"],
                    "server_url": server["url"],
                    "workflow": workflow_name,
                    "message": (
                        f"Job submitted successfully to {server['name']}. "
                        f"Results will be saved to Google Drive. "
                        f"Task ID: {submit_result.get('prompt_id')}"
                    ),
                },
                downloads=[],  # Results saved to Google Drive
                requires_polling=False,  # Submit & forget
                metadata={
                    "skill_name": skill_name,
                    "handler_type": self.handler_type,
                    "server_name": server["name"],
                    "workflow_name": workflow_name,
                },
            )
            
        except Exception as e:
            logger.error(f"ComfyUISkillHandler failed: {e}", exc_info=True)
            return SkillResult(
                output={"error": str(e)},
                success=False,
                error=str(e),
                metadata={
                    "skill_name": skill_name,
                    "handler_type": self.handler_type,
                },
            )
    
    async def handle_batch(
        self,
        skill_name: str,
        tool_calls: list,
        context: SkillContext,
    ) -> SkillResult:
        """
        Execute batch ComfyUI job submissions.
        
        Args:
            skill_name: Name of skill
            tool_calls: List of tool calls
            context: Execution context
        
        Returns:
            SkillResult with batch submission status
        """
        try:
            # Submit each job
            results = []
            for tool_call in tool_calls:
                result = await self.handle(skill_name, tool_call, context)
                results.append(result)
            
            # Aggregate results
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            
            return SkillResult(
                output={
                    "batch_summary": {
                        "total": len(tool_calls),
                        "successful": len(successful),
                        "failed": len(failed),
                    },
                    "results": [r.output for r in results],
                },
                downloads=[],
                metadata={
                    "batch": True,
                    "total": len(tool_calls),
                    "successful": len(successful),
                    "failed": len(failed),
                },
            )
            
        except Exception as e:
            logger.error(f"ComfyUISkillHandler batch failed: {e}", exc_info=True)
            return SkillResult(
                output={"error": str(e)},
                success=False,
                error=str(e),
            )
    
    # -------------------------------------------------------------------------
    # Server Discovery & Health
    # -------------------------------------------------------------------------
    
    async def _load_servers_json(self) -> Dict:
        """
        Load servers.json from Google Drive.
        
        Cached for 5 minutes to avoid repeated API calls.
        """
        now = datetime.now()
        
        # Return cached version if still valid
        if (
            self._servers_cache
            and self._cache_timestamp
            and (now - self._cache_timestamp).total_seconds() < self._cache_ttl_seconds
        ):
            logger.debug("Using cached servers.json")
            return self._servers_cache
        
        # Load from Google Drive
        try:
            import os
            import json
            
            # Try ToolRegistry first (if conductor_server skill is registered)
            try:
                from ..agent_tools import get_registry
                registry = get_registry()
                
                if registry.is_registered("resolve_server_url"):
                    # Get all servers via ToolRegistry
                    servers_file_id = os.environ.get("CONDUCTOR_SERVERS_FILE_ID")
                    service_account_file = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")
                    
                    if servers_file_id and service_account_file:
                        from google.oauth2 import service_account
                        from googleapiclient.discovery import build
                        
                        creds = service_account.Credentials.from_service_account_file(
                            service_account_file,
                            scopes=["https://www.googleapis.com/auth/drive.readonly"],
                        )
                        service = build("drive", "v3", credentials=creds)
                        
                        content = service.files().get_media(fileId=servers_file_id).execute()
                        self._servers_cache = json.loads(content.decode("utf-8") if isinstance(content, bytes) else content)
                        self._cache_timestamp = now
                        logger.info(f"Loaded servers.json from Google Drive")
                        return self._servers_cache
            
            except Exception as e:
                logger.debug(f"ToolRegistry server load failed: {e}")
            
            # Fallback: direct Google Drive API
            servers_file_id = os.environ.get("CONDUCTOR_SERVERS_FILE_ID")
            service_account_file = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")
            
            if servers_file_id and service_account_file:
                from google.oauth2 import service_account
                from googleapiclient.discovery import build
                
                creds = service_account.Credentials.from_service_account_file(
                    service_account_file,
                    scopes=["https://www.googleapis.com/auth/drive.readonly"],
                )
                service = build("drive", "v3", credentials=creds)
                
                content = service.files().get_media(fileId=servers_file_id).execute()
                self._servers_cache = json.loads(content.decode("utf-8") if isinstance(content, bytes) else content)
                self._cache_timestamp = now
                logger.info(f"Loaded servers.json from Google Drive")
                return self._servers_cache
        
        except Exception as e:
            logger.error(f"Failed to load servers.json: {e}")
        
        # Fallback to empty dict
        return {}
    
    async def _find_server_for_workflow(self, workflow_name: str) -> Optional[Dict]:
        """
        Find an online server that has the specified workflow.
        
        Returns:
            Server entry dict or None
        """
        servers = await self._load_servers_json()
        
        for server_name, server_data in servers.items():
            # Check if server is online
            if server_data.get("status") != "online":
                continue
            
            # Check if server has the workflow
            workflows = server_data.get("workflows", [])
            if workflow_name not in workflows:
                continue
            
            # Check if server is actually reachable
            health = await self._check_server_health(server_data.get("url"))
            if not health.get("online"):
                continue
            
            # Found a match!
            return {
                "name": server_name,
                **server_data,
            }
        
        return None
    
    async def _get_online_servers_with_workflows(self) -> List[Dict]:
        """Get list of online servers with available workflows."""
        servers = await self._load_servers_json()
        online_servers = []
        
        for server_name, server_data in servers.items():
            if server_data.get("status") != "online":
                continue
            
            workflows = server_data.get("workflows", [])
            if not workflows:
                continue
            
            # Double-check health
            health = await self._check_server_health(server_data.get("url"))
            if not health.get("online"):
                continue
            
            online_servers.append({
                "name": server_name,
                "url": server_data.get("url"),
                "workflows": workflows,
                "tags": server_data.get("tags", []),
                "colab_runtime": server_data.get("colab_runtime", "Unknown"),
            })
        
        return online_servers
    
    async def _check_server_health(self, server_url: str) -> Dict:
        """Check if server is online via /api/conductor/metadata endpoint."""
        import aiohttp
        
        try:
            health_endpoint = f"{server_url}/api/conductor/metadata"
            
            async with aiohttp.ClientSession() as session:
                start = datetime.now()
                async with session.get(health_endpoint, timeout=5) as response:
                    elapsed = (datetime.now() - start).total_seconds()
                    
                    if response.status == 200:
                        return {
                            "online": True,
                            "url": server_url,
                            "latency_ms": elapsed * 1000,
                        }
                    else:
                        return {
                            "online": False,
                            "url": server_url,
                            "error": f"HTTP {response.status}",
                        }
        
        except Exception as e:
            return {
                "online": False,
                "url": server_url,
                "error": str(e),
            }
    
    async def _get_workflow_metadata(self, server_url: str, workflow_name: str) -> Optional[Dict]:
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
    
    async def _submit_to_comfyui(self, server_url: str, submission_data: dict) -> Dict:
        """Submit job to /api/conductor/execute endpoint."""
        import aiohttp
        
        try:
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
                            "prompt_id": result.get("prompt_id"),
                            "task_key": result.get("task_key"),
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}",
                        }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
    
    def can_handle(self, skill_name: str, tool_call: Dict[str, Any]) -> bool:
        """Check if this handler can process the skill."""
        skill_meta = self._get_skill_metadata(skill_name)
        return skill_meta.get("handler_type") == "comfyui"
