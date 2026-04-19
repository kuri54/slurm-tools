# sltop

`sltop` is a small CLI tool to quickly show CPU, memory, and GPU usage for a Slurm node.

## Purpose

- Check Slurm node status at a glance
- Combine `scontrol` and `nvidia-smi` output in one view
- Run directly with `uvx` (no manual install needed)

## Installation

One-off run (no permanent install):

```bash
uvx --from git+https://github.com/kuri54/slurm-tools.git sltop
```

Persistent install (recommended if you use it often):

```bash
uv tool install git+https://github.com/kuri54/slurm-tools.git
```

After installing, run:

```bash
sltop
```

To test from a local clone:

```bash
uvx --from . sltop
```

## Usage

By default, `sltop` uses the node name from `hostname -s`.

```bash
sltop
```

Example output:

```text
NODE  : <node-name>
STATE : MIXED
CPU   : 30 / 48 used  (18 free)
MEM   : 30720 MB / 250000 MB used  (219280 MB free)
GPU   :
  GPU 0 (NVIDIA RTX A6000): mem 18432 / 49140 MB
  GPU 1 (NVIDIA RTX A6000): mem 18390 / 49140 MB
  GPU 2 (NVIDIA RTX A6000): mem 210 / 49140 MB
```

## How to override the node

You can choose the target node in two ways.

1. Pass the node name as an argument:

```bash
sltop <node-name>
```

2. Set the `SLTOP_NODE` environment variable:

```bash
export SLTOP_NODE=<node-name>
sltop
```

Note: if the target node is not the local host, GPU metrics are fetched via `ssh <node> ...`.
