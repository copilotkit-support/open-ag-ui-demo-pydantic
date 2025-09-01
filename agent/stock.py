from pydantic import BaseModel, Field
from typing import Any, Callable, Literal
from ag_ui.core import (
    StateSnapshotEvent,
    EventType,
    StateDeltaEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolCallArgsEvent,
    TextMessageStartEvent,
    TextMessageEndEvent,
    TextMessageContentEvent,
    CustomEvent,
)
from pydantic_ai import Agent, RunContext
from pydantic_ai.ag_ui import StateDeps
from dotenv import load_dotenv
import uuid
import json
import asyncio
from datetime import datetime

# import datetime
import yfinance as yf
import numpy as np
import pandas as pd

load_dotenv()


class AgentState(BaseModel):
    tools: list = []
    be_stock_data: Any = None
    be_arguments: dict = {}
    available_cash: float = 0.0
    investment_summary: dict = {}
    investment_portfolio: list = []
    tool_logs: list = []
    # render_standard_charts_and_table_args: dict = {}


class JSONPatchOp(BaseModel):
    """A class representing a JSON Patch operation (RFC 6902)."""

    op: Literal["add", "remove", "replace", "move", "copy", "test"] = Field(
        description="The operation to perform: add, remove, replace, move, copy, or test",
    )
    path: str = Field(description="JSON Pointer (RFC 6901) to the target location")
    value: Any = Field(
        default=None,
        description="The value to apply (for add, replace operations)",
    )
    from_: str | None = Field(
        default=None,
        alias="from",
        description="Source path (for move, copy operations)",
    )


class bull_insights(BaseModel):
    title: str = Field(description="Short title for the positive insight.")
    description: str = Field(
        description="Detailed description of the positive insight."
    )
    emoji: str = Field(description="Emoji representing the positive insight.")


class bear_insights(BaseModel):
    title: str = Field(description="Short title for the negative insight.")
    description: str = Field(
        description="Detailed description of the negative insight."
    )
    emoji: str = Field(description="Emoji representing the negative insight.")


class insights(BaseModel):
    bullInsights: list[bull_insights]
    bearInsights: list[bear_insights]


agent = Agent(
    "openai:gpt-4o-mini",
    # instructions='You are a stock portfolio analysis agent. For every single task, you should use tools to gather information from the internet. You have four tools at your disposal: extract_relevant_data_from_user_prompt, generate_insights, generate_stock_report, analyze_stock_performance. For every question, you need to break it down into smaller steps, and use the tools to gather the information. For 1st step, you need to call extract_relevant_data_from_user_prompt tool and the output will be added to portfolio data context. Then you call generate_insights tool and the output will be added to insights context. Then you call generate_stock_report tool and the output will be added to stock report context. Then you call analyze_stock_performance tool and the output will be added to stock performance context. After that you can call the chat tool with the portfolio data, insights, stock report and stock performance to answer the question.',
    instructions="You are a stock portfolio analysis agent. Use the tools provided effectively to answer the user query. When user asks for something related to investments in stock portfolio, you should always call the render_standard_charts_and_table tool with render_standard_charts_and_table_args as the tool argument to the frontend after running the generate_insights tool",
    deps_type=StateDeps[AgentState],
)

# @agent.tool
# async def update_state(ctx: RunContext[StateDeps[AgentState]]) -> StateSnapshotEvent:
#     return StateSnapshotEvent(
#         type=EventType.STATE_SNAPSHOT,
#         snapshot=ctx.deps.state,
#     )


@agent.tool
async def chat_tool(
    ctx: RunContext[StateDeps[AgentState]], message: str
) -> list[TextMessageStartEvent, TextMessageContentEvent, TextMessageEndEvent]:
    """
        When user asks for something unrelated to the stock portfolio, you should use this tool to answer the question. The answers should be generic and should be relevant to the user query. Also when the message role is a tool, you should trigger this tool to provide a summary of the tool call provided. 

    Args:
        ctx (RunContext[StateDeps[AgentState]]): _description_
        message (str): _description_

    Returns:
        list[TextMessageStartEvent, TextMessageContentEvent, TextMessageEndEvent]: _description_
    """
    message_id = str(uuid.uuid4())
    return [
        TextMessageStartEvent(
            type=EventType.TEXT_MESSAGE_START,
            message_id=message_id,
            role="assistant",
        ),
        TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT,
            message_id=message_id,
            delta=message,
        ),
        TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END,
            message_id=str(uuid.uuid4()),
        ),
    ]


@agent.tool
async def gather_stock_data(
    ctx: RunContext[StateDeps[AgentState]],
    stock_tickers_list: list[str],
    investment_date: str,
    interval_of_investment: str,
    amount_of_dollars_to_be_invested: list[float],
    operation: Literal["add", "update", "delete", "replace"],
) -> list[StateSnapshotEvent, StateDeltaEvent]:
    """
    This tool is used for the chat purposes. If the user query is not related to the stock portfolio, you should use this tool to answer the question. The answers should be generic and should be relevant to the user query.

    Args:
        ctx (RunContext[StateDeps[AgentState]]): _description_

    Returns:
        StateSnapshotEvent: _description_
    """
    print(
        stock_tickers_list,
        investment_date,
        interval_of_investment,
        amount_of_dollars_to_be_invested,
        operation,
    )
    changes = []

    changes.append(
        JSONPatchOp(
            op="replace",
            path="/investment_portfolio",
            value=[
                {
                    "ticker": ticker,
                    "amount": amount_of_dollars_to_be_invested[index],
                }
                for index, ticker in enumerate(stock_tickers_list)
            ],
        )
    )
    ctx.deps.state.investment_portfolio = [
        {
            "ticker": ticker,
            "amount": amount_of_dollars_to_be_invested[index],
        }
        for index, ticker in enumerate(stock_tickers_list)
    ]
    tickers = stock_tickers_list
    investment_date = investment_date
    current_year = datetime.now().year
    if current_year - int(investment_date[:4]) > 4:
        print("investment date is more than 4 years ago")
        investment_date = f"{current_year - 4}-01-01"
    if current_year - int(investment_date[:4]) == 0:
        history_period = "1y"
    else:
        history_period = f"{current_year - int(investment_date[:4])}y"

    data = yf.download(
        tickers,
        period=history_period,
        interval="3mo",
        start=investment_date,
        end=datetime.today().strftime("%Y-%m-%d"),
    )
    changes.append(
        JSONPatchOp(
            op="replace",
            path="/be_stock_data",
            value=data["Close"].to_dict(),
        )
    )
    ctx.deps.state.be_stock_data = data["Close"].to_dict()
    changes.append(
        JSONPatchOp(
            op="replace",
            path="/be_arguments",
            value={
                "ticker_symbols": stock_tickers_list,
                "investment_date": investment_date,
                "amount_of_dollars_to_be_invested": amount_of_dollars_to_be_invested,
                "interval_of_investment": interval_of_investment,
            },
        )
    )
    tool_log_id = str(uuid.uuid4())
    # changes.append(
    #     JSONPatchOp(
    #         op="add",
    #         path="/tool_logs/-",
    #         value={
    #             "message": "Gathering stock data",
    #             "status": "completed",
    #             "id": tool_log_id,
    #         },
    #     )
    # )
    ctx.deps.state.be_arguments = {
        "ticker_symbols": stock_tickers_list,
        "investment_date": investment_date,
        "amount_of_dollars_to_be_invested": amount_of_dollars_to_be_invested,
        "interval_of_investment": interval_of_investment,
    }
    # message_id = str(uuid.uuid4())

    return [
        StateSnapshotEvent(
            type=EventType.STATE_SNAPSHOT,
            snapshot=ctx.deps.state,
        ),
        StateDeltaEvent(type=EventType.STATE_DELTA, delta=changes),
    ]


@agent.tool
async def allocate_cash(
    ctx: RunContext[StateDeps[AgentState]],
) -> list[StateDeltaEvent]:
    """
    This tool should be called after gather_stock_data so as to allocate cash to repective stocks extracted from previous stock
    """

    stock_data_dict = (
        ctx.deps.state.be_stock_data
    )  # Dictionary: {ticker: {date: price}}
    # Convert back to DataFrame for processing
    stock_data = pd.DataFrame(stock_data_dict)
    args = ctx.deps.state.be_arguments
    tickers = args["ticker_symbols"]
    investment_date = args["investment_date"]
    amounts = args["amount_of_dollars_to_be_invested"]  # list, one per ticker
    interval = "single_shot"

    if ctx.deps.state.available_cash is not None:
        total_cash = ctx.deps.state.available_cash
    else:
        total_cash = sum(amounts)
    holdings = {ticker: 0.0 for ticker in tickers}
    investment_log = []
    add_funds_needed = False
    add_funds_dates = []

    # Ensure DataFrame is sorted by date
    stock_data = stock_data.sort_index()

    if interval == "single_shot":
        # Buy all shares at the first available date using allocated money for each ticker
        first_date = stock_data.index[0]
        row = stock_data.loc[first_date]
        for idx, ticker in enumerate(tickers):
            price = row[ticker]
            if np.isnan(price):
                investment_log.append(
                    f"{first_date.date()}: No price data for {ticker}, could not invest."
                )
                add_funds_needed = True
                add_funds_dates.append(
                    (str(first_date.date()), ticker, price, amounts[idx])
                )
                continue
            allocated = amounts[idx]
            if total_cash >= allocated and allocated >= price:
                shares_to_buy = allocated // price
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    holdings[ticker] += shares_to_buy
                    total_cash -= cost
                    investment_log.append(
                        f"{first_date.date()}: Bought {shares_to_buy:.2f} shares of {ticker} at ${price:.2f} (cost: ${cost:.2f})"
                    )
                else:
                    investment_log.append(
                        f"{first_date.date()}: Not enough allocated cash to buy {ticker} at ${price:.2f}. Allocated: ${allocated:.2f}"
                    )
                    add_funds_needed = True
                    add_funds_dates.append(
                        (str(first_date.date()), ticker, price, allocated)
                    )
            else:
                investment_log.append(
                    f"{first_date.date()}: Not enough total cash to buy {ticker} at ${price:.2f}. Allocated: ${allocated:.2f}, Available: ${total_cash:.2f}"
                )
                add_funds_needed = True
                add_funds_dates.append(
                    (str(first_date.date()), ticker, price, total_cash)
                )
        # No further purchases on subsequent dates
    else:
        # DCA or other interval logic (previous logic)
        for date, row in stock_data.iterrows():
            for i, ticker in enumerate(tickers):
                price = row[ticker]
                if np.isnan(price):
                    continue  # skip if price is NaN
                # Invest as much as possible for this ticker at this date
                if total_cash >= price:
                    shares_to_buy = total_cash // price
                    if shares_to_buy > 0:
                        cost = shares_to_buy * price
                        holdings[ticker] += shares_to_buy
                        total_cash -= cost
                        investment_log.append(
                            f"{date.date()}: Bought {shares_to_buy:.2f} shares of {ticker} at ${price:.2f} (cost: ${cost:.2f})"
                        )
                else:
                    add_funds_needed = True
                    add_funds_dates.append(
                        (str(date.date()), ticker, price, total_cash)
                    )
                    investment_log.append(
                        f"{date.date()}: Not enough cash to buy {ticker} at ${price:.2f}. Available: ${total_cash:.2f}. Please add more funds."
                    )

    # Calculate final value and new summary fields
    final_prices = stock_data.iloc[-1]
    total_value = 0.0
    returns = {}
    total_invested_per_stock = {}
    percent_allocation_per_stock = {}
    percent_return_per_stock = {}
    total_invested = 0.0
    for idx, ticker in enumerate(tickers):
        # Calculate how much was actually invested in this stock
        if interval == "single_shot":
            # Only one purchase at first date
            first_date = stock_data.index[0]
            price = stock_data.loc[first_date][ticker]
            shares_bought = holdings[ticker]
            invested = shares_bought * price
        else:
            # Sum all purchases from the log
            invested = 0.0
            for log in investment_log:
                if f"shares of {ticker}" in log and "Bought" in log:
                    # Extract cost from log string
                    try:
                        cost_str = log.split("(cost: $")[-1].split(")")[0]
                        invested += float(cost_str)
                    except Exception:
                        pass
        total_invested_per_stock[ticker] = invested
        total_invested += invested
    # Now calculate percent allocation and percent return
    for ticker in tickers:
        invested = total_invested_per_stock[ticker]
        holding_value = holdings[ticker] * final_prices[ticker]
        returns[ticker] = holding_value - invested
        total_value += holding_value
        percent_allocation_per_stock[ticker] = (
            (invested / total_invested * 100) if total_invested > 0 else 0.0
        )
        percent_return_per_stock[ticker] = (
            ((holding_value - invested) / invested * 100) if invested > 0 else 0.0
        )
    total_value += total_cash  # Add remaining cash to total value

    # Store results in state
    ctx.deps.state.investment_summary = {
        "holdings": holdings,
        "final_prices": final_prices.to_dict(),
        "cash": total_cash,
        "returns": returns,
        "total_value": total_value,
        "investment_log": investment_log,
        "add_funds_needed": add_funds_needed,
        "add_funds_dates": add_funds_dates,
        "total_invested_per_stock": total_invested_per_stock,
        "percent_allocation_per_stock": percent_allocation_per_stock,
        "percent_return_per_stock": percent_return_per_stock,
    }
    ctx.deps.state.available_cash = float(total_cash)  # Update available cash in state

    # --- Portfolio vs SPY performanceData logic ---
    # Get SPY prices for the same dates
    spy_ticker = "SPY"
    spy_prices = None
    try:
        spy_prices = yf.download(
            spy_ticker,
            period=f"{len(stock_data)//4}y" if len(stock_data) > 4 else "1y",
            interval="3mo",
            start=stock_data.index[0],
            end=stock_data.index[-1],
        )["Close"]
        # Align SPY prices to stock_data dates
        spy_prices = spy_prices.reindex(stock_data.index, method="ffill")
    except Exception as e:
        print("Error fetching SPY data:", e)
        spy_prices = pd.Series([None] * len(stock_data), index=stock_data.index)

    # Simulate investing the same total_invested in SPY
    spy_shares = 0.0
    spy_cash = total_invested
    spy_invested = 0.0
    spy_investment_log = []
    if interval == "single_shot":
        first_date = stock_data.index[0]
        spy_price = spy_prices.loc[first_date]
        if isinstance(spy_price, pd.Series):
            spy_price = spy_price.iloc[0]
        if not pd.isna(spy_price):
            spy_shares = spy_cash // spy_price
            spy_invested = spy_shares * spy_price
            spy_cash -= spy_invested
            spy_investment_log.append(
                f"{first_date.date()}: Bought {spy_shares:.2f} shares of SPY at ${spy_price:.2f} (cost: ${spy_invested:.2f})"
            )
    else:
        # DCA: invest equal portions at each date
        dca_amount = total_invested / len(stock_data)
        for date in stock_data.index:
            spy_price = spy_prices.loc[date]
            if isinstance(spy_price, pd.Series):
                spy_price = spy_price.iloc[0]
            if not pd.isna(spy_price):
                shares = dca_amount // spy_price
                cost = shares * spy_price
                spy_shares += shares
                spy_cash -= cost
                spy_invested += cost
                spy_investment_log.append(
                    f"{date.date()}: Bought {shares:.2f} shares of SPY at ${spy_price:.2f} (cost: ${cost:.2f})"
                )

    # Build performanceData array
    performanceData = []
    running_holdings = holdings.copy()
    running_cash = total_cash
    for date in stock_data.index:
        # Portfolio value: sum of shares * price at this date + cash
        port_value = (
            sum(
                running_holdings[t] * stock_data.loc[date][t]
                for t in tickers
                if not pd.isna(stock_data.loc[date][t])
            )
            # + running_cash
        )
        # SPY value: shares * price + cash
        spy_price = spy_prices.loc[date]
        if isinstance(spy_price, pd.Series):
            spy_price = spy_price.iloc[0]
        spy_val = spy_shares * spy_price + spy_cash if not pd.isna(spy_price) else None
        performanceData.append(
            {
                "date": str(date.date()),
                "portfolio": float(port_value) if port_value is not None else None,
                "spy": float(spy_val) if spy_val is not None else None,
            }
        )

    ctx.deps.state.investment_summary["performanceData"] = performanceData
    # --- End performanceData logic ---

    # Compose summary message
    if add_funds_needed:
        msg = "Some investments could not be made due to insufficient funds. Please add more funds to your wallet.\n"
        for d, t, p, c in add_funds_dates:
            msg += (
                f"On {d}, not enough cash for {t}: price ${p:.2f}, available ${c:.2f}\n"
            )
    else:
        msg = "All investments were made successfully.\n"
    msg += f"\nFinal portfolio value: ${total_value:.2f}\n"
    msg += "Returns by ticker (percent and $):\n"
    for ticker in tickers:
        percent = percent_return_per_stock[ticker]
        abs_return = returns[ticker]
        msg += f"{ticker}: {percent:.2f}% (${abs_return:.2f})\n"

    # ctx.state.messages.append(
    #     ToolMessage(
    #         role="tool",
    #         id=str(uuid.uuid4()),
    #         content="The relevant details had been extracted",
    #         tool_call_id=ctx.state.messages[-1].tool_calls[0].id,
    #     )
    # )
    tool_log_id = str(uuid.uuid4())
    changes = []
    # changes.append(
    #     JSONPatchOp(
    #         op="add",
    #         path="/tool_logs/-",
    #         value={
    #             "message": "Allocating cash",
    #             "status": "completed",
    #             "id": tool_log_id,
    #         },
    #     )
    # )
    return [StateDeltaEvent(type=EventType.STATE_DELTA, delta=changes)]


@agent.tool
async def generate_insights(
    ctx: RunContext[StateDeps[AgentState]],
    bullInsights: list[bull_insights],
    bearInsights: list[bear_insights],
    tickers: list[str],
) -> list[StateDeltaEvent]:
    """
    This tool should be called after allocate_cash so as to generate insights based on the stock tickers present in ctx.deps.state.investment_summary. Make sure that each insight is unique and not repeated. For each company stocks in the list provided, you should generate 2 positive insights and 2 negative insights. This tool should be called only once after allocating cash. At that time itself insights for all stocks tickers need to be generated.
    """
    print(bullInsights, bearInsights, tickers)
    tool_call_id = str(uuid.uuid4())
    # ctx.deps.state.render_standard_charts_and_table_args = (
    #     {
    #         "investment_summary": ctx.deps.state.investment_summary,
    #         "insights": {
    #             "bullInsights": [insight.model_dump() for insight in bullInsights],
    #             "bearInsights": [insight.model_dump() for insight in bearInsights],
    #         },
    #     }
    #     # default=str,  # Convert non-serializable objects to strings
    # )
    return [
        # StateDeltaEvent(
        #     type=EventType.STATE_DELTA,
        #     delta=[
        #         {"op": "replace", "path": "/tool_logs", "value": []},
        #         {
        #             "op": "replace",
        #             "path": "/render_standard_charts_and_table_args",
        #             "value": ctx.deps.state.render_standard_charts_and_table_args,
        #         },
        #     ],
        # ),
        # CustomEvent(
        #     type=EventType.CUSTOM,
        #     name='render_standard_charts_and_table',
        #     value=[
        #         {
        #             'tool': 'render_standard_charts_and_table',
        #             'tool_argument': json.dumps(
        #                 {
        #                     "investment_summary": ctx.deps.state.investment_summary,
        #                     "insights": {
        #                         "bullInsights": [insight.model_dump() for insight in bullInsights],
        #                         "bearInsights": [insight.model_dump() for insight in bearInsights],
        #                     },
        #                 },
        #                 default=str,  # Convert non-serializable objects to strings
        #             ),
        #         },
        #     ],
        # )
        ToolCallStartEvent(
            type=EventType.TOOL_CALL_START,
            tool_call_id=tool_call_id,
            tool_call_name="render_standard_charts_and_table",
        ),
        ToolCallArgsEvent(
            type=EventType.TOOL_CALL_ARGS,
            tool_call_id=tool_call_id,
            delta=json.dumps(
                {
                    "investment_summary": ctx.deps.state.investment_summary,
                    "insights": {
                        "bullInsights": [
                            insight.model_dump() for insight in bullInsights
                        ],
                        "bearInsights": [
                            insight.model_dump() for insight in bearInsights
                        ],
                    },
                },
                default=str,  # Convert non-serializable objects to strings
            ),
        ),
        # ToolCallEndEvent(
        #     type=EventType.TOOL_CALL_END,
        #     tool_call_id=tool_call_id,
        # )
    ]


pydantic_agent = agent.to_ag_ui(deps=StateDeps(AgentState()))