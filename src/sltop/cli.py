from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field


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


@dataclass
class JobResource:
    job_id: int
    user: str
    state: str
    cpu: int
    gpu: int
    mem_mb: int


@dataclass
class StateAggregate:
    cpu_total: int = 0
    gpu_total: int = 0
    mem_mb_total: int = 0
    job_ids: list[int] = field(default_factory=list)


@dataclass
class UserAggregate:
    run: StateAggregate = field(default_factory=StateAggregate)
    pend: StateAggregate = field(default_factory=StateAggregate)


GPU_QUERY = (
    "nvidia-smi "
    "--query-gpu=index,name,memory.used,memory.total "
    "--format=csv,noheader,nounits"
)
SQUEUE_FORMAT = "%A|%u|%T"
JOB_STATE_MAP = {
    "RUNNING": "RUN",
    "PENDING": "PEND",
}


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


def get_jobs_for_node(node: str) -> list[tuple[int, str, str]]:
    raw = run_command(
        [
            "squeue",
            "-h",
            "-w",
            node,
            "-t",
            "RUNNING,PENDING",
            "-o",
            SQUEUE_FORMAT,
        ]
    )

    jobs: list[tuple[int, str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split("|", 2)]
        if len(parts) != 3:
            continue

        job_id_raw, user, state_raw = parts
        state = JOB_STATE_MAP.get(state_raw)
        if state is None or not user:
            continue

        try:
            job_id = int(job_id_raw)
        except ValueError:
            continue

        jobs.append((job_id, user, state))

    return jobs


def parse_mem_to_mb(mem_str: str) -> int:
    value = mem_str.strip()
    if not value:
        return 0

    m = re.match(
        r"^([0-9]+(?:\.[0-9]+)?)([KMGTPEkmgtpe]?)(?:[iI]?[bB])?(?:[cCnN])?$",
        value,
    )
    if not m:
        return 0

    amount = float(m.group(1))
    unit = m.group(2).upper()
    factor = {
        "": 1,
        "K": 1 / 1024,
        "M": 1,
        "G": 1024,
        "T": 1024 * 1024,
        "P": 1024**3,
        "E": 1024**4,
    }[unit]
    return max(0, int(round(amount * factor)))


def parse_tres_count(value: str) -> int:
    m = re.match(r"^(\d+)", value.strip())
    if not m:
        return 0
    return int(m.group(1))


def parse_req_tres(raw: str) -> tuple[int, int, int]:
    m = re.search(r"\bReqTRES=([^\s]+)", raw)
    if not m:
        return 0, 0, 0

    cpu = 0
    mem_mb = 0
    gpu_from_gres = 0
    gpu_fallback = 0

    for part in m.group(1).split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key == "cpu":
            cpu = parse_tres_count(value)
        elif key == "mem":
            mem_mb = parse_mem_to_mb(value)
        elif key.startswith("gres/gpu"):
            gpu_from_gres += parse_tres_count(value)
        elif key == "gpu":
            gpu_fallback += parse_tres_count(value)

    gpu = gpu_from_gres if gpu_from_gres > 0 else gpu_fallback
    return cpu, gpu, mem_mb


def get_job_resource(job_id: int, user: str, state: str) -> JobResource:
    raw = run_command(["scontrol", "show", "job", "-o", str(job_id)])
    cpu, gpu, mem_mb = parse_req_tres(raw)
    return JobResource(
        job_id=job_id,
        user=user,
        state=state,
        cpu=cpu,
        gpu=gpu,
        mem_mb=mem_mb,
    )


def is_missing_job_error(error: RuntimeError) -> bool:
    detail = str(error).lower()
    if "scontrol show job" not in detail:
        return False

    known_patterns = (
        "invalid job id",
        "unknown job id",
        "invalid job/step id",
        "no jobs in the system",
    )
    return any(p in detail for p in known_patterns)


def get_job_resources(node: str) -> list[JobResource]:
    jobs = get_jobs_for_node(node)

    resources: list[JobResource] = []
    for job_id, user, state in jobs:
        try:
            resource = get_job_resource(job_id, user, state)
        except RuntimeError as e:
            if is_missing_job_error(e):
                continue
            raise
        resources.append(resource)

    return resources


def build_user_aggregates(resources: list[JobResource]) -> dict[str, UserAggregate]:
    aggregates: dict[str, UserAggregate] = {}

    for resource in resources:
        aggregate = aggregates.setdefault(resource.user, UserAggregate())
        bucket = aggregate.run if resource.state == "RUN" else aggregate.pend
        bucket.cpu_total += resource.cpu
        bucket.gpu_total += resource.gpu
        bucket.mem_mb_total += resource.mem_mb
        bucket.job_ids.append(resource.job_id)

    for aggregate in aggregates.values():
        aggregate.run.job_ids.sort()
        aggregate.pend.job_ids.sort()

    return aggregates


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


def format_job_ids(job_ids: list[int]) -> str:
    if not job_ids:
        return "-"
    return ",".join(str(job_id) for job_id in job_ids)


def print_users_report(aggregates: dict[str, UserAggregate]) -> None:
    print("USERS :")

    if not aggregates:
        return

    users = sorted(aggregates.keys())
    user_width = max(len(user) for user in users)

    cpu_width = max(
        1,
        max(
            len(str(state.cpu_total))
            for user in users
            for state in (aggregates[user].run, aggregates[user].pend)
        ),
    )
    gpu_width = max(
        1,
        max(
            len(str(state.gpu_total))
            for user in users
            for state in (aggregates[user].run, aggregates[user].pend)
        ),
    )
    mem_width = max(
        1,
        max(
            len(str(state.mem_mb_total))
            for user in users
            for state in (aggregates[user].run, aggregates[user].pend)
        ),
    )

    for user in users:
        run_state = aggregates[user].run
        pend_state = aggregates[user].pend

        print(
            f"  {user:<{user_width}}  RUN  "
            f"cpu {run_state.cpu_total:>{cpu_width}}  "
            f"gpu {run_state.gpu_total:>{gpu_width}}  "
            f"mem {run_state.mem_mb_total:>{mem_width}}MB   "
            f"jobs: {format_job_ids(run_state.job_ids)}"
        )
        print(
            f"  {'':<{user_width}}  PEND "
            f"cpu {pend_state.cpu_total:>{cpu_width}}  "
            f"gpu {pend_state.gpu_total:>{gpu_width}}  "
            f"mem {pend_state.mem_mb_total:>{mem_width}}MB   "
            f"jobs: {format_job_ids(pend_state.job_ids)}"
        )


def print_report(
    node_info: NodeInfo,
    gpu_info: list[GpuInfo],
    user_aggregates: dict[str, UserAggregate],
) -> None:
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

    print_users_report(user_aggregates)


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
        job_resources = get_job_resources(node)
        user_aggregates = build_user_aggregates(job_resources)
        print_report(node_info, gpu_info, user_aggregates)
        return 0
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
