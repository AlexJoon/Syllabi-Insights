"""
Syllabi Insights Agent - Agentex ACP Entry Point

This is the main entry point for the Agentex framework.
The agent provides RAG-powered insights from academic syllabi using:
- OpenAI's hosted Vector Store for document storage and retrieval
- GPT-4 for intelligent responses with tool calling

Run with: agentex agents run --manifest manifest.yaml
"""

import os
from pathlib import Path
from typing import AsyncGenerator, List, Dict, Any

from dotenv import dotenv_values
from openai import OpenAI

from agentex import AsyncAgentex
from agentex.lib.sdk.fastacp import FastACP
from agentex.lib.types.fastacp import SyncACPConfig
from agentex.lib.types.acp import SendMessageParams
from agentex.types.text_delta import TextDelta
from agentex.types.task_message_update import TaskMessageUpdate, StreamTaskMessageDelta

# Load environment variables directly from .env file (not os.environ which may have manifest placeholders)
env_path = Path(__file__).parent.parent / ".env"
env_vars = dotenv_values(env_path)

# Initialize OpenAI client with the actual key from .env
client = OpenAI(api_key=env_vars.get("OPENAI_API_KEY"))

# Import our RAG components (use relative imports for module execution)
from .lib.vectorstore import VectorStore
from .lib.tools import TOOLS, execute_tool

# Initialize vector store (uses OPENAI_VECTOR_STORE_ID from .env)
vectorstore = VectorStore()

# Initialize Agentex client for fetching conversation history
agentex_client = AsyncAgentex()


async def get_conversation_history(task_id: str) -> List[Dict[str, Any]]:
    """
    Fetch conversation history from Agentex and convert to OpenAI message format.

    Args:
        task_id: The task ID to fetch messages for

    Returns:
        List of messages in OpenAI format (role, content)
    """
    try:
        # Fetch messages for this task from Agentex
        # messages.list() returns List[TaskMessage] directly
        messages = await agentex_client.messages.list(task_id=task_id)

        # Convert Agentex messages to OpenAI format
        openai_messages = []
        for msg in messages:
            # Only process text content messages
            if hasattr(msg.content, 'type') and msg.content.type == 'text':
                # Map Agentex author to OpenAI role
                author = msg.content.author
                role = author if author in ['user', 'assistant', 'system'] else 'assistant'

                openai_messages.append({
                    "role": role,
                    "content": msg.content.content
                })

        return openai_messages
    except Exception as e:
        print(f"[ACP] Warning: Could not fetch conversation history: {e}")
        return []

# System prompt for the syllabi insights agent
SYSTEM_PROMPT = """You are an intelligent assistant specialized in analyzing academic syllabi.
You help users understand course content, requirements, schedules, and learning objectives.

You have access to the following tools:
- search_syllabi: Search through syllabi in the vector store using semantic similarity
- upload_syllabus: Upload a new syllabus file to the vector store
- get_index_stats: Get statistics about the vector store
- list_indexed_files: List all files in the vector store

CRITICAL INSTRUCTIONS:
1. ALWAYS call search_syllabi tool FIRST when a user asks about syllabi content - do NOT respond with text saying you will search, just execute the tool directly
2. NEVER say "Let me search" or "I'll search" - instead, immediately call the appropriate tool
3. After getting tool results, provide a comprehensive answer based on the actual data returned
4. Cite specific courses or sections when relevant
5. Highlight important dates, assignments, and grading policies
6. Compare courses when asked about multiple syllabi

If the vector store is empty, let the user know they can upload files via the OpenAI dashboard
or ask you to upload a file from their local filesystem."""


# Create the ACP server using FastACP factory
acp = FastACP.create("sync", config=SyncACPConfig())


@acp.on_message_send
async def handle_message(params: SendMessageParams) -> AsyncGenerator[TaskMessageUpdate, None]:
    """
    Handle incoming messages from the user.

    This is the main entry point called by Agentex for processing messages.
    Uses streaming to send response chunks as they're generated.
    """
    # Extract the user message from the params
    # SendMessageParams has 'content' field which is a TaskMessageContent (TextContent, etc.)
    user_message = ""
    if params.content:
        # TextContent has a 'content' field with the actual text
        if hasattr(params.content, 'content'):
            user_message = params.content.content

    if not user_message:
        yield StreamTaskMessageDelta(
            type="delta",
            index=0,
            delta=TextDelta(type="text", text_delta="I didn't receive a message. Please try again.")
        )
        return

    # Get task ID for fetching conversation history
    task_id = params.task.id
    print(f"[ACP] Processing message for task: {task_id}")

    # Fetch conversation history from Agentex
    conversation_history = await get_conversation_history(task_id)
    print(f"[ACP] Found {len(conversation_history)} previous messages in conversation")

    # Build message history with system prompt first
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history (excludes system messages)
    for msg in conversation_history:
        if msg["role"] != "system":
            messages.append(msg)

    # Add the current user message
    messages.append({"role": "user", "content": user_message})

    print(f"[ACP] Total messages in context: {len(messages)}")

    # First, check if we need to use tools
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        stream=False,
    )

    assistant_message = response.choices[0].message

    # If the model wants to use tools, execute them
    if assistant_message.tool_calls:
        print(f"[ACP] Model requested {len(assistant_message.tool_calls)} tool call(s)")
        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            print(f"[ACP] Executing tool: {tool_name} with args: {tool_args}")

            # Execute the tool
            result = await execute_tool(tool_name, tool_args, vectorstore)
            print(f"[ACP] Tool result: {result[:200]}..." if len(result) > 200 else f"[ACP] Tool result: {result}")

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

        # index represents the content block, not the chunk number
        # All deltas for the same text response should use index=0
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield StreamTaskMessageDelta(
                    type="delta",
                    index=0,
                    delta=TextDelta(type="text", text_delta=chunk.choices[0].delta.content)
                )
    else:
        # No tools needed, stream directly
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True,
        )

        # index represents the content block, not the chunk number
        # All deltas for the same text response should use index=0
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield StreamTaskMessageDelta(
                    type="delta",
                    index=0,
                    delta=TextDelta(type="text", text_delta=chunk.choices[0].delta.content)
                )
