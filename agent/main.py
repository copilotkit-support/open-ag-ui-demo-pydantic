from fastapi import FastAPI
from fastapi.responses import StreamingResponse  # For streaming responses
from typing import Any
import os
import uvicorn
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.ag_ui import run_ag_ui, SSE_CONTENT_TYPE, StateDeps, handle_ag_ui_request
from fastapi.requests import Request
from http import HTTPStatus
from fastapi.responses import Response, StreamingResponse
from stock import AgentState, agent, pydantic_agent
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.mount(
    "/pydantic-agent",
    pydantic_agent,
    "pydantic agent",
)


def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )


if __name__ == "__main__":
    main()
