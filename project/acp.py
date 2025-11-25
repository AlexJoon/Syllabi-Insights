"""
Syllabi Insights Agent - Agentex ACP Entry Point

This is the main entry point for the Agentex framework.
The agent provides RAG-powered insights from academic syllabi using:
- OpenAI's hosted Vector Store for document storage and retrieval
- GPT-4 for intelligent responses with tool calling

Run with: agentex agents run --manifest manifest.yaml
"""

import os
from typing import AsyncGenerator

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Import our RAG components
from lib.vectorstore import VectorStore
from lib.tools import TOOLS, execute_tool

# Initialize vector store (uses OPENAI_VECTOR_STORE_ID from .env)
vectorstore = VectorStore()

# System prompt for the syllabi insights agent
SYSTEM_PROMPT = """You are an intelligent assistant specialized in analyzing academic syllabi.
You help users understand course content, requirements, schedules, and learning objectives.

You have access to the following tools:
- search_syllabi: Search through syllabi in the vector store using semantic similarity
- upload_syllabus: Upload a new syllabus file to the vector store
- get_index_stats: Get statistics about the vector store
- list_indexed_files: List all files in the vector store

When answering questions:
1. Use the search_syllabi tool to find relevant content before answering
2. Cite specific courses or sections when relevant
3. Highlight important dates, assignments, and grading policies
4. Compare courses when asked about multiple syllabi
5. Be helpful, accurate, and educational in your responses

If the vector store is empty, let the user know they can upload files via the OpenAI dashboard
or ask you to upload a file from their local filesystem."""


async def stream_response(user_message: str, history: list[dict] | None = None) -> AsyncGenerator[str, None]:
    """
    Stream a response to the user's message using RAG.

    This is the main entry point called by Agentex for streaming responses.

    Args:
        user_message: The user's input message
        history: Optional conversation history

    Yields:
        String chunks of the response
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history if provided
    if history:
        messages.extend(history)

    # Add the current user message
    messages.append({"role": "user", "content": user_message})

    # First, check if we need to use tools
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        stream=False,  # First pass to check for tool calls
    )

    assistant_message = response.choices[0].message

    # If the model wants to use tools, execute them
    if assistant_message.tool_calls:
        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments

            # Execute the tool
            result = await execute_tool(tool_name, tool_args, vectorstore)

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        # Now stream the final response with tool results
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    else:
        # No tools needed, stream directly
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


async def run(user_message: str, history: list[dict] | None = None) -> str:
    """
    Non-streaming response handler.

    This is called by Agentex when streaming is not requested.

    Args:
        user_message: The user's input message
        history: Optional conversation history

    Returns:
        Complete response string
    """
    response_parts = []
    async for chunk in stream_response(user_message, history):
        response_parts.append(chunk)
    return "".join(response_parts)


# Agentex entry point - this is what the framework calls
async def handle_message(message: str, context: dict | None = None) -> AsyncGenerator[str, None]:
    """
    Main Agentex entry point for handling messages.

    Args:
        message: User's message
        context: Optional context including conversation history

    Yields:
        Response chunks for streaming
    """
    history = context.get("history", []) if context else []
    async for chunk in stream_response(message, history):
        yield chunk
