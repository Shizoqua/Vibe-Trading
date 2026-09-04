"""Regression coverage for replaying compacted non-repeatable tool results."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.agent.context import ContextBuilder
from src.agent.grounding import ToolAuthorization
from src.agent.loop import AgentLoop, _microcompact
from src.agent.tools import BaseTool, ToolRegistry
from src.agent.trace import TraceWriter


class _DataTool(BaseTool):
    """Small data tool whose execution count is observable."""

    name = "nonrepeatable_data"
    description = "test data tool"
    parameters: dict = {"type": "object", "properties": {"symbol": {"type": "string"}}}
    is_readonly = True
    deterministic = False

    def __init__(self, *, repeatable: bool = False, status: str = "ok") -> None:
        self.repeatable = repeatable
        self.status = status
        self.calls: list[dict] = []

    def execute(self, **kwargs: object) -> str:
        self.calls.append(dict(kwargs))
        return json.dumps(
            {
                "status": self.status,
                "symbol": kwargs.get("symbol"),
                "payload": "x" * 256,
                "execution": len(self.calls),
            }
        )


def _build_agent(tmp_path: Path, tool: BaseTool):
    registry = ToolRegistry()
    registry.register(tool)
    captured: list[tuple[str, dict]] = []
    agent = AgentLoop(
        registry=registry,
        llm=SimpleNamespace(),
        max_iterations=4,
        event_callback=lambda name, data: captured.append((name, data)),
    )
    run_dir = tmp_path / tool.name
    run_dir.mkdir()
    agent.memory.run_dir = str(run_dir)
    return agent, run_dir, captured


def _call(
    agent: AgentLoop,
    run_dir: Path,
    messages: list[dict],
    react_trace: list[dict],
    *,
    index: int,
    symbol: str,
) -> None:
    trace = TraceWriter(run_dir)
    agent._process_tool_calls(
        [SimpleNamespace(id=f"call_{index}", name="nonrepeatable_data", arguments={"symbol": symbol})],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        index,
    )
    trace.close()


def test_compacted_nonrepeatable_result_is_replayed_without_reexecution(tmp_path: Path) -> None:
    tool = _DataTool()
    agent, run_dir, captured = _build_agent(tmp_path, tool)
    messages: list[dict] = []
    react_trace: list[dict] = []

    _call(agent, run_dir, messages, react_trace, index=1, symbol="600584.SH")
    original = messages[0]["content"]
    messages.extend(
        {"role": "tool", "tool_call_id": f"dummy_{i}", "content": "d" * 200}
        for i in range(3)
    )
    _microcompact(messages)
    assert messages[0]["content"] == "[cleared]"

    _call(agent, run_dir, messages, react_trace, index=2, symbol="600584.SH")

    assert len(tool.calls) == 1
    assert messages[-1]["content"] == original
    assert any(event["type"] == "tool_result_cached" for event in react_trace)
    cached_events = [data for _, data in captured if data.get("cached")]
    assert len(cached_events) == 1
    assert cached_events[0]["tool"] == tool.name


def test_different_arguments_stay_blocked_instead_of_replaying_stale_data(tmp_path: Path) -> None:
    tool = _DataTool()
    agent, run_dir, _ = _build_agent(tmp_path, tool)
    messages: list[dict] = []
    react_trace: list[dict] = []

    _call(agent, run_dir, messages, react_trace, index=1, symbol="600584.SH")
    _call(agent, run_dir, messages, react_trace, index=2, symbol="AAPL")

    assert len(tool.calls) == 1
    assert json.loads(messages[-1]["content"])["skipped"] is True
    assert not any(event["type"] == "tool_result_cached" for event in react_trace)


def test_failed_nonrepeatable_result_is_not_cached(tmp_path: Path) -> None:
    tool = _DataTool(status="error")
    agent, run_dir, _ = _build_agent(tmp_path, tool)
    messages: list[dict] = []
    react_trace: list[dict] = []

    _call(agent, run_dir, messages, react_trace, index=1, symbol="600584.SH")
    _call(agent, run_dir, messages, react_trace, index=2, symbol="600584.SH")

    assert len(tool.calls) == 2
    assert not any(event["type"] == "tool_result_cached" for event in react_trace)


def test_repeatable_tool_still_executes_again(tmp_path: Path) -> None:
    tool = _DataTool(repeatable=True)
    agent, run_dir, _ = _build_agent(tmp_path, tool)
    messages: list[dict] = []
    react_trace: list[dict] = []

    _call(agent, run_dir, messages, react_trace, index=1, symbol="600584.SH")
    _call(agent, run_dir, messages, react_trace, index=2, symbol="600584.SH")

    assert len(tool.calls) == 2
    assert not any(event["type"] == "tool_result_cached" for event in react_trace)


def test_cached_nonrepeatable_replay_still_passes_grounding_gate(tmp_path: Path) -> None:
    tool = _DataTool()
    agent, run_dir, _ = _build_agent(tmp_path, tool)
    messages: list[dict] = []
    react_trace: list[dict] = []
    seen: list[str] = []

    class _Grounding:
        authorized_symbols: set[str] = set()
        identity_status = "locked"

        def authorize_tool_call(self, tool_name, arguments, **kwargs):
            seen.append(kwargs["call_id"])
            return ToolAuthorization(allowed=True)

        def identity_summary(self):
            return {}

        def ingest_tool_result(self, **kwargs):
            return None

    agent._grounding = _Grounding()

    _call(agent, run_dir, messages, react_trace, index=1, symbol="600584.SH")
    _call(agent, run_dir, messages, react_trace, index=2, symbol="600584.SH")

    assert seen == ["call_1", "call_2"]
    assert len(tool.calls) == 1
    assert any(event["type"] == "tool_result_cached" for event in react_trace)
