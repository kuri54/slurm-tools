from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class NodeInfo:
    node: str
    state: str
    cpu_alloc: int
    cpu_tot: int
    mem_alloc: int
    mem_tot: int


@dataclass
class GpuInfo:
    index: int
    name: str
    mem_used: int
    mem_total: int


GPU_QUERY = (
    "nvidia-smi "
    "--query-gpu=index,name,memory.used,memory.total "
    "--format=csv,noheader,nounits"
)


def run_command(cmd: list[str]) -> str:
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except FileNotFoundError:
        raise RuntimeError(f"command not found: {cmd[0]}")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else ""
        stdout = e.stdout.strip() if e.stdout else ""
        detail = stderr or stdout or f"exit status {e.returncode}"
        raise RuntimeError(f"failed to run {' '.join(cmd)}: {detail}")


def default_node_name() -> str:
    env_node = os.environ.get("SLTOP_NODE")
    if env_node:
        return env_node

    return local_short_hostname()


def local_short_hostname() -> str:
    try:
        out = run_command(["hostname", "-s"]).strip()
        if out:
            return out
    except RuntimeError:
        pass

    return socket.gethostname().split(".")[0]


def parse_node_info(raw: str, node: str) -> NodeInfo:
    def extract_int(key: str) -> int:
        m = re.search(rf"\b{re.escape(key)}=(\d+)\b", raw)
        if not m:
            raise RuntimeError(f"failed to parse {key} from scontrol output")
        return int(m.group(1))

    def extract_str(key: str) -> str:
        m = re.search(rf"\b{re.escape(key)}=([^\s]+)", raw)
        if not m:
            raise RuntimeError(f"failed to parse {key} from scontrol output")
        return m.group(1)

    return NodeInfo(
        node=node,
        state=extract_str("State"),
        cpu_alloc=extract_int("CPUAlloc"),
        cpu_tot=extract_int("CPUTot"),
        mem_alloc=extract_int("AllocMem"),
        mem_tot=extract_int("RealMemory"),
    )


def get_node_info(node: str) -> NodeInfo:
    raw = run_command(["scontrol", "show", "node", node])
    return parse_node_info(raw, node)


def parse_gpu_info(raw: str) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue

        try:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    mem_used=int(parts[2]),
                    mem_total=int(parts[3]),
                )
            )
        except ValueError:
            continue

    return gpus


def is_local_node(node: str) -> bool:
    local = local_short_hostname().split(".")[0].lower()
    target = node.split(".")[0].lower()
    return local == target


def get_gpu_info(node: str) -> list[GpuInfo]:
    if is_local_node(node):
        if shutil.which("nvidia-smi") is None:
            return []
        raw = run_command(GPU_QUERY.split())
        return parse_gpu_info(raw)

    if shutil.which("ssh") is None:
        raise RuntimeError(
            "ssh command not found; cannot query GPU metrics on remote node"
        )

    raw = run_command(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            node,
            GPU_QUERY,
        ]
    )
    return parse_gpu_info(raw)


def print_report(node_info: NodeInfo, gpu_info: list[GpuInfo]) -> None:
    cpu_free = node_info.cpu_tot - node_info.cpu_alloc
    mem_free = node_info.mem_tot - node_info.mem_alloc

    print(f"NODE  : {node_info.node}")
    print(f"STATE : {node_info.state}")
    print(
        f"CPU   : {node_info.cpu_alloc} / {node_info.cpu_tot} used  "
        f"({cpu_free} free)"
    )
    print(
        f"MEM   : {node_info.mem_alloc} MB / {node_info.mem_tot} MB used  "
        f"({mem_free} MB free)"
    )
    print("GPU   :")

    for gpu in gpu_info:
        print(f"  GPU {gpu.index} ({gpu.name}): mem {gpu.mem_used} / {gpu.mem_total} MB")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sltop",
        description="Simple Slurm resource viewer",
    )
    parser.add_argument(
        "node",
        nargs="?",
        default=None,
        help="node name (default: SLTOP_NODE or `hostname -s`)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    node = args.node or default_node_name()

    try:
        node_info = get_node_info(node)
        gpu_info = get_gpu_info(node)
        print_report(node_info, gpu_info)
        return 0
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
