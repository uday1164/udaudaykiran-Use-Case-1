import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from langchain_core.messages import HumanMessage

from agents.retry_utils import (
    classify_error_type,
    estimate_input_tokens,
    extract_failed_generation,
    invoke_agent_with_retry,
)


class DummyAgent:
    def __init__(self):
        self.called = False

    def invoke(self, input_dict, config=None):
        self.called = True
        return {"messages": []}


def test_preflight_input_budget_exceeded_blocks_before_invoke():
    agent = DummyAgent()
    input_dict = {"messages": [HumanMessage(content="x" * 200)]}

    with pytest.raises(RuntimeError, match="preflight_input_budget_exceeded"):
        invoke_agent_with_retry(agent, input_dict, agent_name="TEST", max_input_tokens=10)

    assert agent.called is False


def test_classify_error_type_for_tool_schema_failure():
    error = RuntimeError("tool call validation failed: attempted to call tool 'return_json' which was not in request.tools")
    assert classify_error_type(error) == "tool_schema_hard_block"


def test_classify_error_type_for_request_too_large():
    error = RuntimeError("Error code: 413 - Request too large for model on tokens per minute")
    assert classify_error_type(error) == "request_too_large_shrink"


def test_extract_failed_generation_snippet_when_present():
    error = RuntimeError(
        "Error code: 400 - {'error': {'message': 'tool call validation failed', "
        "'failed_generation': '<function=return_json>{\"answer\": 1}</function>'}}"
    )
    snippet = extract_failed_generation(error)
    assert "return_json" in snippet


def test_estimate_input_tokens_uses_message_content():
    input_dict = {"messages": [HumanMessage(content="abcd" * 25)]}
    assert estimate_input_tokens(input_dict) >= 25