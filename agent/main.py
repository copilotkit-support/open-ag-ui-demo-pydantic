from fastapi import FastAPI
from fastapi.responses import StreamingResponse  # For streaming responses
import uuid
from typing import Any
import os
import uvicorn
import asyncio
from pydantic import ValidationError
import json
from ag_ui.core import (
    RunAgentInput,
    StateSnapshotEvent,
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    TextMessageStartEvent,
    TextMessageEndEvent,
    TextMessageContentEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolCallArgsEvent,
    StateDeltaEvent,
)
from pydantic_ai import Agent
from pydantic_ai.ag_ui import run_ag_ui, SSE_CONTENT_TYPE, StateDeps, handle_ag_ui_request
from fastapi.requests import Request
from http import HTTPStatus
from fastapi.responses import Response, StreamingResponse
from stock import AgentState, agent, pydantic_agent
from dotenv import load_dotenv
from starlette.types import Scope, Receive, Send

load_dotenv()
app = FastAPI()

# app.mount(
#     "/pydantic-agent",
#     pydantic_agent,
#     "pydantic-agent",
# )

@app.post('/pydantic-agent')
async def run_agent(request: Request) -> Response:
    res = await request.json()
    print("Running agent")
    return await handle_ag_ui_request(agent, request, deps=StateDeps(AgentState(
        available_cash=float(res['state']['available_cash']),
        investment_summary=res['state']['investment_summary'],
        investment_portfolio=res['state']['investment_portfolio'],
        tool_logs=res['state']['tool_logs'],
        tools=res['state']['tools'],
        be_stock_data=res['state']['be_stock_data'],
        be_arguments=res['state']['be_arguments'],
        # render_standard_charts_and_table_args=res['state']['render_standard_charts_and_table_args'],
    )))


# @app.post("/pydantic-agent")
# async def pydantic_agent(request: Request) -> Response:
#     accept = request.headers.get('accept', SSE_CONTENT_TYPE)
#     try:
#         run_input = RunAgentInput.model_validate(await request.json())
#     except ValidationError as e:  # pragma: no cover
#         return Response(
#             content=json.dumps({"error": str(e)}),
#             media_type='application/json',
#             status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
#         )

#     event_stream = run_ag_ui(agent, run_input, accept=accept, deps=StateDeps(AgentState(
#         tools=run_input.tools,
#         messages=run_input.messages,
#         be_stock_data=None,
#         be_arguments={},
#         available_cash=run_input.state['available_cash'],
#         investment_summary=run_input.state['investment_summary'],
#         investment_portfolio=run_input.state['investment_portfolio'],
#         render_standard_charts_and_table_args=run_input.state['render_standard_charts_and_table_args'],
#         tool_logs=[]
#     )))

    # return StreamingResponse(event_stream, media_type=accept)


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
