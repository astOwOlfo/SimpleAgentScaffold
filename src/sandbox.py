import asyncio
from asyncio import Lock
from itertools import count
from pathlib import Path
from beartype import beartype
from dataclasses import dataclass
from typing import List

_container_creation_lock = Lock()


@dataclass
class CompletedProcess:
    returncode: int
    stdout: str
    stderr: str


@beartype
class DockerSandbox:
    container_name: str

    def __init__(self):
        self.container_name = ""

    async def __aenter__(self):
        async with _container_creation_lock:
            self.container_name = await self.make_unique_container_name()
            await self.build_image()
            await self.start_container()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.cleanup()

    async def make_unique_container_name(self) -> str:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "ps",
            "-a",
            "--format",
            "{{.Names}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise Exception(f"Error getting existing containers: {stderr}")
        existing_containers = stdout.splitlines()
        for i in count():
            name = f"bash-sandbox-instance-{i}"
            if name not in existing_containers:
                return name

    async def build_image(self) -> None:
        sandbox_path = Path("./sandbox")
        if not sandbox_path.is_dir():
            raise FileNotFoundError(f"Sandbox directory '{sandbox_path}' not found.")
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "build",
            "-t",
            "bash-sandbox",
            str(sandbox_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise Exception(f"Error building image: {stderr}")

    async def start_container(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "run",
            "-d",
            "--name",
            self.container_name,
            "--tty",
            "bash-sandbox",
            "/bin/sh",
            "-c",
            "while true; do sleep 1; done",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise Exception(f"Error starting container: {stderr}")

    async def run_command(
        self, command: str, timeout_seconds: int = 30
    ) -> CompletedProcess:
        args = [
            "docker",
            "exec",
            self.container_name,
            "/bin/bash",
            "-c",
            command,
        ]
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout_seconds
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return CompletedProcess(returncode=1, stdout="", stderr="Timed out.")

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        return CompletedProcess(
            returncode=proc.returncode, stdout=stdout, stderr=stderr
        )

    async def cleanup(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "stop",
            self.container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            self.container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()


"""
import subprocess
from threading import Lock
from itertools import count
from pathlib import Path
from beartype import beartype


_container_creation_lock = Lock()


@beartype
class DockerSandbox:
    container_name: str

    def __init__(self):
        with _container_creation_lock:
            self.container_name = self.make_unique_container_name()
            self.build_image()
            self.start_container()

    def make_unique_container_name(self) -> str:
        existing_containers = subprocess.check_output(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            text=True
        ).splitlines()
        for i in count():
            name = f'bash-sandbox-instance-{i}'
            if name not in existing_containers:
                return name

    def build_image(self) -> None:
        sandbox_path = Path('./sandbox')
        if not sandbox_path.is_dir():
            raise FileNotFoundError(f"Sandbox directory '{sandbox_path}' not found.")
        subprocess.run(
            ["docker", "build", "-t", "bash-sandbox", str(sandbox_path)],
            check=True
        )

    def start_container(self) -> None:
        subprocess.run(
            [
                "docker", "run", "-d", "--name", self.container_name,
                "--tty", "bash-sandbox", "/bin/sh", "-c", "while true; do sleep 1; done"
            ],
            check=True
        )

    def run_command(self, command: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["docker", "exec", self.container_name, "/bin/bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    def cleanup(self) -> None:
        subprocess.run(["docker", "stop", self.container_name], check=True)
        subprocess.run(["docker", "rm", self.container_name], check=True)

    def __enter__(self) -> 'DockerSandbox':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.cleanup()


# Example usage:
if __name__ == "__main__":
    with DockerSandbox() as sandbox:
        result = sandbox.run_command("echo Hello, World!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
"""
