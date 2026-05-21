from __future__ import annotations

from typing import Optional

from cheat_at_search.agent.openai_agent import OpenAIAgent
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.codegen.models import GuardrailResponse


def _resolve_logger(logger=None, logger_name: str = "validators"):
    if logger is not None:
        return logger
    return log_to_stdout(logger_name=logger_name)


def make_length_validator(max_lines: int = 10, max_cols: int = 120):
    guardrail_desc = (
        f"Edits longer than {max_lines} and wider than {max_cols} "
        "characters will be rejected."
    )

    def length_validation(code: str) -> Optional[str]:
        if code.count("\n") > max_lines:
            return f"Code exceeds maximum length of {max_lines} lines."

        for line in code.split("\n"):
            if len(line) > max_cols + 20:
                return f"Line exceeds maximum length of {max_cols} characters: {line}"
        return None

    length_validation.__doc__ = guardrail_desc
    return length_validation


def make_guardrail_checker(
    prompt: str,
    model: str = "openai/gpt-5-mini",
    reasoning: str = "medium",
    logger=None,
):
    agent = OpenAIAgent(
        tools=[],
        model=model,
        response_model=GuardrailResponse,
        reasoning_level=reasoning,
    )
    logger = _resolve_logger(logger)

    def code_guardrails(code: str) -> Optional[str]:
        """Edits where the code appears to be overfit to training queries will be rejected."""
        inputs = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    "Please evaluate the following code for compliance:\n"
                    f"```python\n{code}\n```"
                ),
            },
        ]
        resp = agent.loop(inputs=inputs)
        if resp is None:
            return "Guardrail check failed: no response from model."
        if not resp.compliant:
            issues = (
                "\n".join(resp.issues)
                if resp.issues
                else "No specific issues provided."
            )
            return f"Code does not comply with guardrails:\n{issues}"
        logger.debug("Guardrail check passed.")

    return code_guardrails
