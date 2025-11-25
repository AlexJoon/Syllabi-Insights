"""
Vector Store - OpenAI's hosted Vector Store API.

Uses OpenAI's Vector Stores (from the Assistants API) for document
storage and retrieval, allowing you to upload files via the dashboard.
"""

import os
from openai import OpenAI


class VectorStore:
    """
    Vector store using OpenAI's hosted Vector Store API.

    This allows you to:
    1. Create a vector store in the OpenAI dashboard
    2. Upload files directly in the dashboard
    3. Query them from this agent

    Set OPENAI_VECTOR_STORE_ID in your .env file.
    """

    def __init__(self, vector_store_id: str | None = None):
        """
        Initialize the vector store.

        Args:
            vector_store_id: OpenAI Vector Store ID (vs_xxx) from dashboard
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_store_id = vector_store_id or os.getenv("OPENAI_VECTOR_STORE_ID")

        if not self.vector_store_id:
            print("Warning: OPENAI_VECTOR_STORE_ID not set. Creating a new vector store...")
            self._create_vector_store()

    def _create_vector_store(self):
        """Create a new vector store if none exists."""
        vs = self.client.vector_stores.create(
            name="syllabi-insights-store"
        )
        self.vector_store_id = vs.id
        print(f"Created vector store: {vs.id}")
        print(f"Add this to your .env file: OPENAI_VECTOR_STORE_ID={vs.id}")

    def upload_file(self, file_path: str) -> dict:
        """
        Upload a file to the vector store.

        Args:
            file_path: Path to the file (PDF, DOCX, TXT, etc.)

        Returns:
            Upload result with file ID
        """
        # First upload the file to OpenAI
        with open(file_path, "rb") as f:
            file = self.client.files.create(
                file=f,
                purpose="assistants"
            )

        # Then attach it to the vector store
        vs_file = self.client.vector_stores.files.create(
            vector_store_id=self.vector_store_id,
            file_id=file.id
        )

        return {
            "file_id": file.id,
            "vector_store_file_id": vs_file.id,
            "status": vs_file.status,
        }

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search the vector store using OpenAI's file search.

        Note: OpenAI's vector store search is done via the Assistants API
        or the Responses API with file_search tool. For direct search,
        we'll use the Responses API.

        Args:
            query: Search query
            top_k: Number of results (max_num_results)

        Returns:
            List of matching chunks with content and scores
        """
        # Use the Responses API with file_search tool
        response = self.client.responses.create(
            model="gpt-4o",
            tools=[{
                "type": "file_search",
                "vector_store_ids": [self.vector_store_id],
                "max_num_results": top_k,
            }],
            input=f"Search for: {query}\n\nReturn the relevant excerpts from the documents.",
        )

        # Extract the file search results
        results = []
        for output in response.output:
            if output.type == "file_search_call":
                for result in output.results or []:
                    results.append({
                        "content": result.text,
                        "score": result.score,
                        "file_id": result.file_id,
                        "filename": result.filename,
                    })

        return results

    def search_simple(self, query: str, top_k: int = 10) -> str:
        """
        Simple search that returns formatted context for RAG.

        Uses GPT-4o with file_search to get relevant content.

        Args:
            query: What to search for
            top_k: Max results

        Returns:
            Formatted string of relevant content
        """
        response = self.client.responses.create(
            model="gpt-4o",
            tools=[{
                "type": "file_search",
                "vector_store_ids": [self.vector_store_id],
                "max_num_results": top_k,
            }],
            input=query,
        )

        # Return the model's response which includes file search context
        return response.output_text

    def list_files(self) -> list[dict]:
        """List all files in the vector store."""
        files = self.client.vector_stores.files.list(
            vector_store_id=self.vector_store_id
        )
        return [
            {
                "id": f.id,
                "status": f.status,
            }
            for f in files.data
        ]

    def get_stats(self) -> dict:
        """Get vector store statistics."""
        vs = self.client.vector_stores.retrieve(self.vector_store_id)
        return {
            "id": vs.id,
            "name": vs.name,
            "file_count": vs.file_counts.completed,
            "status": vs.status,
            "usage_bytes": vs.usage_bytes,
        }
