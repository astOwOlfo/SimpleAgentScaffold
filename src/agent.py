from openai import AsyncOpenAI
from typing import Any
import inspect
from inspect import signature
import json
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from openai.types.chat import ChatCompletionMessage
from beartype import beartype

from src.sandbox import DockerSandbox


@beartype
class Tool(ABC):
    @abstractmethod
    def description_for_openai_api(self) -> dict:
        pass

    @abstractmethod
    async def _call(self, **kwargs) -> Any:
        pass

    async def call(self, json_arguments: str) -> str | None:
        try:
            arguments = json.loads(json_arguments)
        except json.JSONDecodeError:
            return None

        if not isinstance(arguments, dict):
            return None

        required_arguments = [
            arg
            for arg, param in signature(self._call).parameters.items()
            if param.default == inspect.Parameter.empty
        ]
        optional_arguments = [
            arg
            for arg, param in signature(self._call).parameters.items()
            if param.default != inspect.Parameter.empty
        ]

        if not (set(required_arguments) <= set(arguments.keys())):
            return None

        if not (
            set(arguments.keys()) <= set(required_arguments) | set(optional_arguments)
        ):
            return None

        return await self._call(**arguments)


class FinishTool(Tool):
    def description_for_openai_api(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Call this tool when you are done. Do not call it unless you have checked that you have completed the task successfully.",
                "parameters": None,
            },
        }

    async def _call(self) -> None:
        pass


@beartype
@dataclass
class BashTool(Tool):
    sandbox: DockerSandbox
    verbose: bool = False
    max_output_length: int = 1024

    def description_for_openai_api(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "run_bash_command",
                "description": "Run a bash command and return the output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        }

    async def _call(self, command: str) -> str:
        if self.verbose:
            print("\033[1;31mRUNNING COMMAND:\033[0m", command)

        result = await self.sandbox.run_command(command)

        if self.verbose:
            print(f"\033[1;31mEXIT CODE: {result.returncode}\033[0m")
            print("\033[1;31mSTDOUT:\033[0m", result.stdout)
            print("\033[1;31mSTDOUT:\033[0m", result.stderr)

        return json.dumps(
            {
                "exit_code": result.returncode,
                "stdout": self._truncate(result.stdout),
                "stderr": self._truncate(result.stderr),
            }
        )

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_length:
            return text
        return (
            text[: self.max_output_length // 2]
            + "[TRUNCATED]"
            + text[-self.max_output_length // 2 :]
        )


@beartype
@dataclass
class Agent:
    model: str
    openai_client: AsyncOpenAI
    tools: list[Tool] = field(default_factory=lambda: [])
    max_turns: int = 15
    system_message: str | None = (
        "You are an agent that can call tools to complete a task. You are very thorough and do never give up. If something doesn't work, you make the changes that make are most likely to make it work and keep trying until it works. There is no human in the loop, so you cannot ask for help or ask for clarification. Before you call the finish tool, you must checked whether you have completed the task successfully. If it turns out that what you did failed complete the task, you should retry completing the task and not call the finish tool. When you are sure that you have completed the task successfully, call the finish tool."
    )

    def __post_init__(self):
        if not any(isinstance(tool, FinishTool) for tool in self.tools):
            self.tools.append(FinishTool())

    async def run(self, prompt: str) -> None:
        conversation = [{"role": "user", "content": prompt}]

        if self.system_message is not None:
            conversation.append({"role": "system", "content": self.system_message})

        for _ in range(self.max_turns):
            response = await self._chatbot_response(conversation)

            conversation.append(response)

            if response.tool_calls is None:
                assert (
                    response.content is not None
                )  # this assert cannot be triggered, right?
                conversation.append({"role": "assistant", "content": response.content})
                conversation.append(
                    {
                        "role": "user",
                        "content": "You should call tools instead of giving chat responses.",
                    }
                )
                continue

            finished = False

            for tool_call in response.tool_calls:
                tool = next(
                    (
                        tool
                        for tool in self.tools
                        if tool.description_for_openai_api()["function"]["name"]
                        == tool_call.function.name
                    ),
                    None,
                )
                assert tool is not None  # this assert cannot be triggered, right?

                if isinstance(tool, FinishTool):
                    # we cannot just return here because there might be multiple tool calls and the FinishTool call might not be the last one
                    finished = True
                    break

                returned = await tool.call(tool_call.function.arguments)
                conversation.append(
                    {
                        "role": "tool",
                        "content": json.dumps(returned),
                        "tool_call_id": tool_call.id,
                    }
                )

            if finished:
                return

    async def _chatbot_response(
        self, conversation: list[dict | ChatCompletionMessage]
    ) -> ChatCompletionMessage:
        response = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=conversation,
            tools=[tool.description_for_openai_api() for tool in self.tools],
        )
        return response.choices[0].message
