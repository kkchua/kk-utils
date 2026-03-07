"""
KK-Utils - RAG Client Module

Reusable client for Personal Assistant RAG (Retrieval-Augmented Generation) API.
Provides consistent access to:
- Document management (upload, list, delete, stats)
- Subscription status
- RAG queries

Usage:
    from kk_utils import RAGClient

    client = RAGClient(base_url="http://localhost:8000", user_id="demo_user")
    
    # List documents
    docs = client.list_documents()
    
    # Get subscription
    sub = client.get_subscription()
    
    # Upload document
    result = client.upload_document("/path/to/file.pdf")
"""
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RAGClient:
    """
    Client for Personal Assistant RAG API.

    Provides unified access to document management and subscription APIs.
    All HTTP calls are logged at DEBUG level.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        user_id: str = "demo_user",
        timeout: int = 30,
        access_token: Optional[str] = None,
    ):
        """
        Initialize RAG client.

        Args:
            base_url: Backend API base URL
            user_id: User ID for requests
            timeout: Request timeout in seconds
            access_token: JWT access token for authentication (optional)
        """
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self.timeout = timeout
        self.access_token = access_token
        self.session = requests.Session()
        self.session.headers.update({
            "X-User-ID": self.user_id,
            # Don't set Content-Type here - it will be set automatically per request
            # For multipart uploads, requests will set "multipart/form-data"
            # For JSON requests, requests will set "application/json"
        })
        
        # Add authorization header if token provided
        if self.access_token:
            self.session.headers.update({
                "Authorization": f"Bearer {self.access_token}",
            })

        logger.debug(f"RAGClient initialized: base_url={self.base_url}, user_id={self.user_id}")
        logger.debug(f"Authentication: {'Enabled' if access_token else 'Disabled'}")

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make GET request with logging."""
        url = f"{self.base_url}{path}"
        logger.debug(f"GET {url} params={params}")
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Response: {result}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"GET {path} failed: {e}")
            raise

    def _post(
        self,
        path: str,
        json: Optional[Dict] = None,
        files: Optional[Dict] = None,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make POST request with logging."""
        url = f"{self.base_url}{path}"
        logger.debug(f"POST {url} json={json} params={params}")

        try:
            response = self.session.post(url, json=json, files=files, data=data, params=params, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Response: {result}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"POST {path} failed: {e}")
            raise

    def _delete(self, path: str) -> Dict[str, Any]:
        """Make DELETE request with logging."""
        url = f"{self.base_url}{path}"
        logger.debug(f"DELETE {url}")
        
        try:
            response = self.session.delete(url, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Response: {result}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"DELETE {path} failed: {e}")
            raise

    # =========================================================================
    # Document Management
    # =========================================================================

    def upload_document(self, file_path: str, collection: str = "digital_me") -> Dict[str, Any]:
        """
        Upload a document.

        Args:
            file_path: Path to file to upload

        Returns:
            Upload result with document info
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Uploading document: {file_path.name} -> collection: {collection}")

        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f)}
            data = {"metadata": "", "collection": collection}

            return self._post(
                "/api/v1/documents/upload",
                files=files,
                data=data,
            )

    def list_collections(self) -> List[Dict[str, Any]]:
        """List all RAG collections."""
        logger.debug("Listing RAG collections")
        return self._get("/api/v1/rag/collections")

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a specific collection."""
        logger.debug(f"Getting stats for collection: {collection_name}")
        return self._get(f"/api/v1/rag/collections/{collection_name}/stats")

    def list_collection_documents(
        self,
        collection_name: str,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List documents in a specific collection. Returns dict with 'documents' and 'total'."""
        logger.debug(f"Listing documents in collection: {collection_name}")
        return self._get(
            f"/api/v1/rag/collections/{collection_name}/documents",
            params={"limit": limit, "offset": offset},
        )

    def search_collection(
        self,
        collection_name: str,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Search within a specific collection. Returns dict with 'results' list."""
        logger.debug(f"Searching collection {collection_name}: {query[:50]}")
        # Backend takes query/top_k as URL query params on this POST endpoint
        return self._post(
            f"/api/v1/rag/collections/{collection_name}/search",
            params={"query": query, "top_k": top_k},
        )

    def delete_collection(self, collection_name: str) -> Dict[str, Any]:
        """Delete an entire collection."""
        logger.info(f"Deleting collection: {collection_name}")
        return self._delete(f"/api/v1/rag/collections/{collection_name}")

    def delete_collection_document(self, collection_name: str, doc_id: str) -> Dict[str, Any]:
        """Delete a document/chunk from a specific collection."""
        logger.info(f"Deleting document {doc_id} from collection: {collection_name}")
        return self._delete(f"/api/v1/rag/collections/{collection_name}/documents/{doc_id}")

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List user's documents.

        Returns:
            List of document metadata
        """
        logger.debug("Listing documents")
        result = self._get("/api/v1/documents")
        return result.get("documents", [])

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Get document details.

        Args:
            doc_id: Document ID

        Returns:
            Document metadata and text preview
        """
        logger.debug(f"Getting document: {doc_id}")
        return self._get(f"/api/v1/documents/{doc_id}")

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Delete a document.

        Args:
            doc_id: Document ID

        Returns:
            Delete result
        """
        logger.info(f"Deleting document: {doc_id}")
        return self._delete(f"/api/v1/documents/{doc_id}")

    def get_document_text(self, doc_id: str) -> str:
        """
        Get full document text.

        Args:
            doc_id: Document ID

        Returns:
            Full extracted text
        """
        logger.debug(f"Getting document text: {doc_id}")
        result = self._get(f"/api/v1/documents/{doc_id}/text")
        return result.get("text", "")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get document statistics.

        Returns:
            Statistics including total documents, size, chunks, and subscription info
        """
        logger.debug("Getting document stats")
        return self._get("/api/v1/documents/stats/summary")

    # =========================================================================
    # Subscription Management
    # =========================================================================

    def get_subscription(self) -> Dict[str, Any]:
        """
        Get user's subscription status.

        Returns:
            Subscription info with plan, features, and usage
        """
        logger.debug(f"Getting subscription for user: {self.user_id}")
        result = self._get(f"/api/v1/admin/subscription/{self.user_id}")
        return result

    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive usage summary.

        Returns:
            Usage summary with all tracked features
        """
        logger.debug("Getting usage summary")
        result = self.get_subscription()
        return result.get("usage", {})

    def check_feature_limit(
        self,
        feature: str,
        amount: int = 1,
    ) -> Dict[str, Any]:
        """
        Check if user can use a feature.

        Args:
            feature: Feature name (e.g., "document_uploads", "rag_queries_per_day")
            amount: Amount to check

        Returns:
            Dict with allowed, current, limit, remaining, percentage
        """
        logger.debug(f"Checking feature limit: {feature} (amount={amount})")
        return self._post(
            f"/api/v1/admin/subscription/{self.user_id}/check",
            json={"feature": feature, "amount": amount},
        )

    # =========================================================================
    # RAG Queries
    # =========================================================================

    def query(
        self,
        question: str,
        top_k: int = 3,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        Query the RAG system.

        Args:
            question: User question
            top_k: Number of results to return
            include_sources: Include source documents in response

        Returns:
            RAG response with answer and sources
        """
        logger.info(f"RAG query: {question[:50]}...")
        
        payload = {
            "query": question,
            "top_k": top_k,
            "include_sources": include_sources,
            "user_id": self.user_id,
        }
        
        return self._post("/api/v1/rag/query", json=payload)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search documents.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of matching document chunks
        """
        logger.debug(f"Search: {query[:50]}...")
        
        payload = {
            "query": query,
            "top_k": top_k,
            "user_id": self.user_id,
        }
        
        result = self._post("/api/v1/rag/search", json=payload)
        return result.get("results", [])

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def get_subscription_display(self) -> str:
        """
        Get formatted subscription status for UI display.

        Returns:
            Formatted markdown string
        """
        try:
            status = self.get_subscription()
            
            logger.debug(f"Subscription response: {status}")

            if not status.get("success"):
                logger.warning(f"Subscription API returned failure: {status.get('error', 'unknown')}")
                return "⚠️ Unable to load subscription"

            plan = status.get("plan_name", "Unknown")
            is_trial = status.get("is_trial", False)
            usage = status.get("usage", {})
            
            logger.debug(f"Subscription usage: {usage}")

            lines = [f"### 📊 Subscription Status\n", f"**Plan:** {plan}"]

            if is_trial:
                lines.append("🧪 TRIAL")

            lines.append("\n**Usage:**")

            feature_names = {
                "chat_messages_per_session": "Chat messages",
                "tool_calls_per_message": "Tool calls",
                "document_uploads": "Document uploads",
                "rag_queries_per_day": "RAG queries",
                "ai_processing_minutes": "AI processing",
            }

            features = usage.get("features", {})
            logger.debug(f"Subscription features: {features}")
            
            for feature_key, data in features.items():
                display_name = feature_names.get(feature_key, feature_key.replace("_", " ").title())
                current = data.get("current", 0)
                limit = data.get("limit", 999999)
                unlimited = data.get("unlimited", False)

                if unlimited:
                    lines.append(f"✓ {display_name}: Unlimited")
                else:
                    pct = data.get("percentage", 0)
                    lines.append(f"◉ {display_name}: {current}/{limit} ({pct:.0f}%)")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Get subscription display failed: {e}", exc_info=True)
            return "❌ Error loading subscription"

    def get_documents_display(self, limit: int = 5) -> str:
        """
        Get formatted document list for UI display.

        Args:
            limit: Max documents to show

        Returns:
            Formatted markdown string
        """
        try:
            docs = self.list_documents()
            
            if not docs:
                return "### 📄 Uploaded Documents\n\n**No documents uploaded yet**"
            
            lines = [
                "### 📄 Uploaded Documents\n",
                f"**Total:** {len(docs)} document(s)",
                "",
            ]
            
            for doc in docs[:limit]:
                status_icon = "✅" if doc.get("is_processed", False) else "❌"
                chunks = f"{doc.get('chunk_count', 0)} chunks" if doc.get("chunk_count", 0) > 0 else "Processing..."
                filename = doc.get("filename", "Unknown")
                file_type = doc.get("file_type", "unknown").upper()
                
                lines.append(f"- {status_icon} **{filename}** ({file_type}) - {chunks}")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Get documents display failed: {e}")
            return "❌ Error loading documents"
