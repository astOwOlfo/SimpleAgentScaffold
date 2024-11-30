import asyncio
from openai import AsyncOpenAI
from tabulate import tabulate
from pathlib2 import Path
from os import getenv
import json
from dataclasses import dataclass
from beartype import beartype
from tqdm.asyncio import tqdm
from src.sandbox import DockerSandbox
from src.agent import Agent, BashTool


@beartype
@dataclass(frozen=True)
class Task:
    category: str
    description: str
    public_tests: str
    private_tests: str


@beartype
@dataclass(frozen=True)
class EvaluationResult:
    public_tests_passed: bool
    private_tests_passed: bool


async def evaluate_agent(
    task: Task, model: str = "gpt-4o-mini", max_turns: int = 15
) -> EvaluationResult:
    async with DockerSandbox() as sandbox:
        agent = Agent(
            model=model,
            openai_client=AsyncOpenAI(api_key=getenv("OPENAI_API_KEY")),
            tools=[BashTool(sandbox)],
            max_turns=max_turns,
        )

        await agent.run(task.description)

        await sandbox.run_command(
            f"cat << EOF > public_tests.py\n{task.public_tests}\nEOF"
        )
        public_tests_completed_process = await sandbox.run_command(
            "pytest public_tests.py"
        )
        public_tests_passed = public_tests_completed_process.returncode == 0

        await sandbox.run_command(
            f"cat << EOF > private_tests.py\n{task.private_tests}\nEOF"
        )
        private_tests_completed_process = await sandbox.run_command(
            "pytest private_tests.py"
        )
        private_tests_passed = private_tests_completed_process.returncode == 0

        return EvaluationResult(
            public_tests_passed=public_tests_passed,
            private_tests_passed=private_tests_passed,
        )


@beartype
async def evaluate_agent_multiple_tasks(
    tasks: list[Task],
    model: str = "gpt-4o-mini",
    max_turns: int = 15,
    max_run_in_parallel: int = 64,
) -> dict[Task, EvaluationResult]:
    semaphore = asyncio.Semaphore(max_run_in_parallel)

    async def evaluate_one(*args, **kwargs):
        async with semaphore:
            return await evaluate_agent(*args, **kwargs)

    results = await tqdm.gather(
        *[evaluate_one(task=task, model=model, max_turns=max_turns) for task in tasks],
        desc="running evals",
    )

    return dict(zip(tasks, results))


@beartype
def print_evaluation_results(evaluation_results: dict[Task, EvaluationResult]) -> None:
    headers = [
        "category",
        "both failed",
        "public passed",
        "private passed",
        "both passed",
    ]
    table = []
    task_categories = list(set(task.category for task in evaluation_results.keys()))
    for task_category in task_categories:
        table.append(
            [task_category]
            + [
                len(
                    [
                        task
                        for task, result in evaluation_results.items()
                        if task.category == task_category
                        and result.private_tests_passed == private_passed
                        and result.public_tests_passed == public_passed
                    ]
                )
                for private_passed, public_passed in [
                    (False, False),
                    (False, True),
                    (True, False),
                    (True, True),
                ]
            ]
        )
    print(tabulate(table, headers=headers))


@beartype
def load_tasks() -> list[Task]:
    tasks_dir = Path("tasks")
    tasks = []

    for task_file in tasks_dir.glob("*.json"):
        with open(task_file) as f:
            task_data = json.load(f)
            for task in task_data:
                tasks.append(
                    Task(
                        category=task_file.stem,
                        description=task["task_description"],
                        public_tests=task["task_verifiable"],
                        private_tests=task["private_tests"],
                    )
                )

    return tasks
