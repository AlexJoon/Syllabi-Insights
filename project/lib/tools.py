"""
Tools - OpenAI function calling tools for the Syllabi Insights Agent.

These tools are exposed to GPT-4 for RAG operations on syllabi
using OpenAI's hosted Vector Store.
"""

import json

from .vectorstore import VectorStore


# OpenAI function definitions for tool calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_syllabi",
            "description": "Search through syllabi documents in the vector store using semantic similarity. Use this to find relevant course information before answering questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query - what you're looking for in the syllabi",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "upload_syllabus",
            "description": "Upload a new syllabus document to the vector store. Supports PDF, DOCX, TXT, and other text files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the syllabus file on the local filesystem",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_index_stats",
            "description": "Get statistics about the vector store - how many files are indexed, storage used, etc.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_indexed_files",
            "description": "List all files currently indexed in the vector store.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


async def execute_tool(tool_name: str, arguments: str, vectorstore: VectorStore) -> str:
    """
    Execute a tool and return the result.

    Args:
        tool_name: Name of the tool to execute
        arguments: JSON string of tool arguments
        vectorstore: VectorStore instance

    Returns:
        Tool result as a string
    """
    args = json.loads(arguments) if arguments else {}

    if tool_name == "search_syllabi":
        return await _search_syllabi(vectorstore, **args)
    elif tool_name == "upload_syllabus":
        return await _upload_syllabus(vectorstore, **args)
    elif tool_name == "get_index_stats":
        return await _get_index_stats(vectorstore)
    elif tool_name == "list_indexed_files":
        return await _list_indexed_files(vectorstore)
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


async def _search_syllabi(
    vectorstore: VectorStore,
    query: str,
    num_results: int = 5,
) -> str:
    """Search the syllabi using OpenAI's vector store."""
    try:
        results = vectorstore.search(query=query, top_k=num_results)

        if not results:
            return json.dumps({
                "results": [],
                "message": "No matching content found. Make sure you have files in your vector store.",
            })

        formatted = []
        for r in results:
            formatted.append({
                "content": r.get("content", "")[:500] + "..." if len(r.get("content", "")) > 500 else r.get("content", ""),
                "filename": r.get("filename", "Unknown"),
                "relevance": round(r.get("score", 0), 3),
            })

        return json.dumps({"results": formatted, "total_found": len(results)})

    except Exception as e:
        return json.dumps({"error": str(e)})


async def _upload_syllabus(vectorstore: VectorStore, file_path: str) -> str:
    """Upload a syllabus file to the vector store."""
    try:
        result = vectorstore.upload_file(file_path)
        return json.dumps({
            "success": True,
            "message": f"File uploaded successfully",
            "file_id": result["file_id"],
            "status": result["status"],
        })
    except FileNotFoundError:
        return json.dumps({"success": False, "error": f"File not found: {file_path}"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


async def _get_index_stats(vectorstore: VectorStore) -> str:
    """Get vector store statistics."""
    try:
        stats = vectorstore.get_stats()
        return json.dumps(stats)
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _list_indexed_files(vectorstore: VectorStore) -> str:
    """List all indexed files."""
    try:
        files = vectorstore.list_files()
        return json.dumps({"files": files, "total": len(files)})
    except Exception as e:
        return json.dumps({"error": str(e)})
