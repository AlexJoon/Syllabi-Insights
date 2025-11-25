"""
Syllabi Insights Agent - Agentex ACP Entry Point

This is the main entry point for the Agentex framework.
The agent provides RAG-powered insights from academic syllabi using:
- OpenAI's hosted Vector Store for document storage and retrieval
- GPT-4 for intelligent responses with tool calling

Run with: agentex agents run --manifest manifest.yaml
"""

import os
import json
import uuid
from pathlib import Path
from typing import AsyncGenerator, List, Dict, Any, Optional
from datetime import datetime, UTC
from contextlib import asynccontextmanager

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

# Initialize Agentex client for fetching conversation history and tracing
agentex_client = AsyncAgentex()


@asynccontextmanager
async def trace_span(
    trace_id: str,
    name: str,
    input_data: Optional[Dict[str, Any]] = None,
    parent_id: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
):
    """
    Context manager for creating and managing observability spans.

    Uses task_id as trace_id so spans appear in the UI traces sidebar.
    Automatically handles span creation, timing, and completion.
    """
    span_id = str(uuid.uuid4())
    start_time = datetime.now(UTC)

    try:
        # Create the span
        span = await agentex_client.spans.create(
            id=span_id,
            name=name,
            trace_id=trace_id,
            start_time=start_time,
            input=input_data,
            parent_id=parent_id,
            data=data,
        )
        yield span
    except Exception as e:
        print(f"[TRACE] Error creating span '{name}': {e}")
        yield None
        return

    # Update span with end time on completion
    try:
        await agentex_client.spans.update(
            span_id=span_id,
            end_time=datetime.now(UTC),
        )
    except Exception as e:
        print(f"[TRACE] Error updating span '{name}': {e}")


async def update_span_output(span_id: str, output: Dict[str, Any]):
    """Update a span with output data."""
    try:
        await agentex_client.spans.update(
            span_id=span_id,
            output=output,
            end_time=datetime.now(UTC),
        )
    except Exception as e:
        print(f"[TRACE] Error updating span output: {e}")


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
    Includes observability tracing for all major operations.
    """
    # Extract the user message from the params
    user_message = ""
    if params.content:
        if hasattr(params.content, 'content'):
            user_message = params.content.content

    if not user_message:
        yield StreamTaskMessageDelta(
            type="delta",
            index=0,
            delta=TextDelta(type="text", text_delta="I didn't receive a message. Please try again.")
        )
        return

    # Get task ID - used as trace_id for observability
    task_id = params.task.id
    print(f"[ACP] Processing message for task: {task_id}")

    # Create root span for the entire message handling
    async with trace_span(
        trace_id=task_id,
        name="handle_message",
        input_data={"user_message": user_message[:500]},  # Truncate for readability
        data={"task_id": task_id}
    ) as root_span:
        root_span_id = root_span.id if root_span else None

        # Trace: Fetch conversation history
        async with trace_span(
            trace_id=task_id,
            name="fetch_conversation_history",
            parent_id=root_span_id,
            data={"task_id": task_id}
        ) as history_span:
            conversation_history = await get_conversation_history(task_id)
            if history_span:
                await update_span_output(history_span.id, {
                    "message_count": len(conversation_history)
                })
        print(f"[ACP] Found {len(conversation_history)} previous messages in conversation")

        # Build message history with system prompt first
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in conversation_history:
            if msg["role"] != "system":
                messages.append(msg)
        messages.append({"role": "user", "content": user_message})
        print(f"[ACP] Total messages in context: {len(messages)}")

        # Trace: Initial LLM call (tool decision)
        async with trace_span(
            trace_id=task_id,
            name="llm_tool_decision",
            parent_id=root_span_id,
            input_data={"model": "gpt-4o", "message_count": len(messages)},
            data={"has_tools": True}
        ) as llm_span:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                stream=False,
            )
            assistant_message = response.choices[0].message
            has_tool_calls = bool(assistant_message.tool_calls)
            if llm_span:
                await update_span_output(llm_span.id, {
                    "has_tool_calls": has_tool_calls,
                    "tool_count": len(assistant_message.tool_calls) if has_tool_calls else 0,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0
                    }
                })

        # If the model wants to use tools, execute them
        if assistant_message.tool_calls:
            print(f"[ACP] Model requested {len(assistant_message.tool_calls)} tool call(s)")
            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                print(f"[ACP] Executing tool: {tool_name} with args: {tool_args}")

                # Trace: Tool execution
                async with trace_span(
                    trace_id=task_id,
                    name=f"tool:{tool_name}",
                    parent_id=root_span_id,
                    input_data={"tool": tool_name, "arguments": json.loads(tool_args) if tool_args else {}},
                    data={"tool_call_id": tool_call.id}
                ) as tool_span:
                    result = await execute_tool(tool_name, tool_args, vectorstore)
                    if tool_span:
                        # Parse result and truncate if needed for display
                        try:
                            result_preview = json.loads(result)
                            if isinstance(result_preview, dict) and "results" in result_preview:
                                result_preview = {"result_count": len(result_preview.get("results", []))}
                        except:
                            result_preview = {"result_length": len(result)}
                        await update_span_output(tool_span.id, result_preview)

                print(f"[ACP] Tool result: {result[:200]}..." if len(result) > 200 else f"[ACP] Tool result: {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            # Trace: Final LLM response generation
            async with trace_span(
                trace_id=task_id,
                name="llm_generate_response",
                parent_id=root_span_id,
                input_data={"model": "gpt-4o", "message_count": len(messages)},
                data={"streaming": True, "with_tool_results": True}
            ) as response_span:
                stream = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    stream=True,
                )
                response_text = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        response_text += content
                        yield StreamTaskMessageDelta(
                            type="delta",
                            index=0,
                            delta=TextDelta(type="text", text_delta=content)
                        )
                if response_span:
                    await update_span_output(response_span.id, {
                        "response_length": len(response_text)
                    })
        else:
            # No tools needed - Trace: Direct LLM response
            async with trace_span(
                trace_id=task_id,
                name="llm_generate_response",
                parent_id=root_span_id,
                input_data={"model": "gpt-4o", "message_count": len(messages)},
                data={"streaming": True, "with_tool_results": False}
            ) as response_span:
                stream = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    stream=True,
                )
                response_text = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        response_text += content
                        yield StreamTaskMessageDelta(
                            type="delta",
                            index=0,
                            delta=TextDelta(type="text", text_delta=content)
                        )
                if response_span:
                    await update_span_output(response_span.id, {
                        "response_length": len(response_text)
                    })
