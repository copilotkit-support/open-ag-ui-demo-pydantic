from fastapi import FastAPI
from fastapi.responses import StreamingResponse  # For streaming responses
from typing import Any
import os
import uvicorn
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.ag_ui import run_ag_ui, SSE_CONTENT_TYPE, StateDeps, handle_ag_ui_request
# from fastapi.requests import Request
from http import HTTPStatus
from fastapi.responses import Response, StreamingResponse
from stock import AgentState, agent
from dotenv import load_dotenv
from starlette.requests import Request
from starlette.responses import Response


from pydantic_ai.ag_ui import handle_ag_ui_request

load_dotenv()
app = FastAPI()

# app.mount(
#     "/pydantic-agent",
#     pydantic_agent,
#     "pydantic agent",
# )

@app.post('/pydantic-agent')
async def run_agent(request: Request) -> Response:
    return await handle_ag_ui_request(agent = agent, deps = StateDeps(AgentState()), request=request)

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
