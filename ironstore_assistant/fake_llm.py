"""A scripted stand-in for a chat model.

Used by the tests and by the notebooks when no OPENAI_API_KEY is set, so every step of the
pipeline can be run (and understood) offline. It never "thinks": it returns what you script.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda


class FakeLLM(BaseChatModel):
    """Returns scripted replies. `structured` is what with_structured_output() returns,
    `tool_script` is a list of AIMessages returned one by one by the tool-bound model."""

    text: str = "rewritten question"
    structured: Any = None
    tool_script: list = []
    seen_prompts: list = []

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen_prompts.append(messages)
        if self.tool_script:
            return ChatResult(generations=[ChatGeneration(message=self.tool_script.pop(0))])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(self.text))])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def run(prompt_value):
            self.seen_prompts.append(prompt_value)
            return self.structured
        return RunnableLambda(run)
