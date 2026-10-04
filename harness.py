import sys

sys.stdout.reconfigure(encoding="utf-8")  # so emoji don't crash the Windows console

import asyncio
import json
import os
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from pydantic import BaseModel

ROOT = Path(__file__).parent
WORKSPACE = ROOT / "workspace"
MEMORY_FILE = ROOT / "memory.md"
MCP_CONFIG = ROOT / "mcp_servers.json"


# ---------------------------------------------------------------------------
# Pillar 3: the model registry. Every entry is just an OpenAI-compatible
# endpoint. Cloud or local, frontier or free - same client, same harness.
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    base_url: str
    api_key: str = "none"
    model: str


MODELS: dict[str, ModelConfig] = {
    "local": ModelConfig(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen3.5:4b",
    ),
    "local-small": ModelConfig(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen3.5:2b",
    ),
    "gpt": ModelConfig(
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model="gpt-5.2",
    ),
    "claude": ModelConfig(
        # Anthropic's OpenAI-compatible endpoint - same client works.
        base_url="https://api.anthropic.com/v1/",
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        model="claude-sonnet-5",
    ),
}

model_name = "local"  # switch at runtime with /model


# ---------------------------------------------------------------------------
# Pillar 2: memory. A markdown file in the prompt + a tool to append to it.
# ---------------------------------------------------------------------------

def load_memory() -> str:
    if MEMORY_FILE.is_file():
        return MEMORY_FILE.read_text(encoding="utf-8")
    return "(nothing saved yet)"


def save_memory(fact: str) -> str:
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- {fact}\n")
    return f"saved: {fact}"


# ---------------------------------------------------------------------------
# Local tools (the ones we wrote by hand back in v1)
# ---------------------------------------------------------------------------

def list_files() -> str:
    return "\n".join(p.name for p in WORKSPACE.iterdir()) or "(empty)"


def read_file(filename: str) -> str:
    path = WORKSPACE / filename
    return path.read_text(encoding="utf-8") if path.is_file() else f"error: no {filename}"


def write_file(filename: str, content: str) -> str:
    (WORKSPACE / filename).write_text(content, encoding="utf-8")
    return f"wrote {filename}"


LOCAL_TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "save_memory": save_memory,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files in the user's workspace folder.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one file from the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file in the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "Save one short fact about the user to long-term memory. "
                "Use whenever you learn something lasting: their name, "
                "preferences, projects, recurring tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"],
            },
        },
    },
]


SYSTEM_PROMPT = """You are a helpful personal assistant running inside a \
custom harness. Be concise.

The user's workspace folder holds their personal files: notes, todo lists, \
ideas. You have tools to list, read, and write those files. Never claim you \
lack access to the user's files or tasks - use your tools to look.

Here is what you remember about the user from previous sessions:
{memory}

When you learn a new lasting fact about the user, save it with save_memory."""

# the running conversation - shared by every model we switch to
messages = [{"role": "system", "content": SYSTEM_PROMPT.format(memory=load_memory())}]


# ---------------------------------------------------------------------------
# Pillar 1: MCP (same component we built in v3)
# ---------------------------------------------------------------------------

mcp_sessions: dict[str, ClientSession] = {}  # tool name -> its server session


async def connect_mcp(stack: AsyncExitStack):
    """Launch every server in mcp_servers.json and merge in its tools."""
    config = json.loads(MCP_CONFIG.read_text(encoding="utf-8"))
    for name, spec in config["mcpServers"].items():
        params = StdioServerParameters(command=spec["command"], args=spec["args"])
        try:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except Exception as e:
            print(f"  [mcp] '{name}' failed to start: {e}")
            continue

        tools = (await session.list_tools()).tools
        for tool in tools:
            mcp_sessions[tool.name] = session
            TOOL_SCHEMAS.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
            )
        print(f"  [mcp] connected '{name}': {[t.name for t in tools]}")


async def call_tool(name: str, args: dict) -> str:
    if name in LOCAL_TOOLS:
        return LOCAL_TOOLS[name](**args)
    if name in mcp_sessions:
        result = await mcp_sessions[name].call_tool(name, args)
        return "\n".join(c.text for c in result.content if hasattr(c, "text"))
    return f"error: unknown tool {name}"


# ---------------------------------------------------------------------------
# The agent loop (unchanged since v2)
# ---------------------------------------------------------------------------

async def run_agent(user_message: str) -> str:
    cfg = MODELS[model_name]
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    messages.append({"role": "user", "content": user_message})
    while True:
        response = await client.chat.completions.create(
            model=cfg.model, messages=messages, tools=TOOL_SCHEMAS
        )
        message = response.choices[0].message
        if not message.tool_calls:
            messages.append({"role": "assistant", "content": message.content})
            return message.content or ""

        messages.append(message)
        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            print(f"  [tool] {call.function.name}({args})")
            result = await call_tool(call.function.name, args)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

def handle_command(line: str) -> bool:
    """Returns True if the line was a command."""
    global model_name
    if not line.startswith("/"):
        return False
    cmd, _, arg = line.partition(" ")
    if cmd == "/models":
        for name, cfg in MODELS.items():
            marker = "*" if name == model_name else " "
            print(f" {marker} {name:<12} {cfg.model}  ({cfg.base_url})")
    elif cmd == "/model":
        if arg in MODELS:
            model_name = arg
            print(f"  switched to {arg} ({MODELS[arg].model})")
        else:
            print(f"  unknown model '{arg}' - try /models")
    elif cmd == "/tools":
        for schema in TOOL_SCHEMAS:
            fn = schema["function"]
            origin = "local" if fn["name"] in LOCAL_TOOLS else "mcp"
            print(f"  [{origin}] {fn['name']}: {fn['description'][:60]}")
    elif cmd == "/memory":
        print(load_memory())
    elif cmd == "/quit":
        raise SystemExit
    else:
        print("  commands: /models /model <name> /tools /memory /quit")
    return True


async def main():
    async with AsyncExitStack() as stack:
        print("starting harness...")
        await connect_mcp(stack)
        print(
            f"\nharness ready - model: {model_name} "
            f"({MODELS[model_name].model}), "
            f"{len(TOOL_SCHEMAS)} tools, memory loaded"
        )
        print("type /models, /model <name>, /tools, /memory, or just talk\n")
        try:
            while True:
                line = (await asyncio.to_thread(input, "you: ")).strip()
                if not line or handle_command(line):
                    continue
                print("\nassistant:", await run_agent(line), "\n")
        except (EOFError, KeyboardInterrupt, SystemExit):
            print("\nbye!")


if __name__ == "__main__":
    asyncio.run(main())
