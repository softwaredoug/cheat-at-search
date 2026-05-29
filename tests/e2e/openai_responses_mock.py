import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock


def function_call(name: str, arguments: dict, call_id: str = "call_1"):
    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=json.dumps(arguments),
        call_id=call_id,
    )


def reasoning(text: str = "mock reasoning"):
    return SimpleNamespace(
        type="reasoning",
        summary=[SimpleNamespace(text=text)],
    )


def message(text: str = "mock message"):
    return SimpleNamespace(type="message", content=text)


def responses_api_response(
    *,
    output=None,
    output_parsed=None,
    input_tokens: int = 10,
    output_tokens: int = 3,
    response_id: str = "resp_test_123",
):
    return SimpleNamespace(
        id=response_id,
        output=output or [],
        output_parsed=output_parsed,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


class MockOpenAIResponses:
    """Queue-backed OpenAI Responses API mock for e2e tests.

    Tests enqueue create/parse responses in the order the code should receive
    them. Parsed responses can be supplied as dictionaries; when the application
    calls ``responses.parse(..., text_format=SomeModel)``, the mock validates the
    dictionary into that Pydantic model and exposes it as ``output_parsed``.
    """

    def __init__(self):
        self.create_queue = deque()
        self.parse_queue = deque()
        self.constructor = Mock(side_effect=self._build_client)
        self.client = Mock()
        self.client.responses.create = Mock(side_effect=self._create)
        self.client.responses.parse = Mock(side_effect=self._parse)

    def queue_create_response(self, *, output=None, output_parsed=None, **kwargs):
        self.create_queue.append(
            responses_api_response(
                output=output,
                output_parsed=output_parsed,
                **kwargs,
            )
        )

    def queue_parse_response(self, *, output=None, output_parsed=None, payload=None, **kwargs):
        self.parse_queue.append(
            {
                "output": output,
                "output_parsed": output_parsed,
                "payload": payload,
                "kwargs": kwargs,
            }
        )

    def _build_client(self, **_kwargs):
        return self.client

    def _create(self, **_kwargs):
        if not self.create_queue:
            raise AssertionError("No mocked OpenAI responses.create response queued")
        return self.create_queue.popleft()

    def _parse(self, **kwargs):
        if not self.parse_queue:
            raise AssertionError("No mocked OpenAI responses.parse response queued")
        queued = self.parse_queue.popleft()
        output_parsed = queued["output_parsed"]
        if output_parsed is None and queued["payload"] is not None:
            response_model = kwargs.get("text_format")
            if response_model is None:
                raise AssertionError("payload queued but responses.parse was called without text_format")
            output_parsed = response_model.model_validate(queued["payload"])
        return responses_api_response(
            output=queued["output"],
            output_parsed=output_parsed,
            **queued["kwargs"],
        )
