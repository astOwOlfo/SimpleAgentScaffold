import asyncio
from src.evaluate import (
    load_tasks,
    evaluate_agent_multiple_tasks,
    Task,
    print_evaluation_results,
)


def main():
    tasks: list[Task] = load_tasks()
    evaluation_results = asyncio.run(
        evaluate_agent_multiple_tasks(tasks, max_run_in_parallel=256)
    )
    print_evaluation_results(evaluation_results)


if __name__ == "__main__":
    main()
