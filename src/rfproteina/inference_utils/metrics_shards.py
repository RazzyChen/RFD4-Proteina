"""Durable, rank-local inference metrics; merging requires no process group."""

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_write(path, lambda name: Path(name).write_text(json.dumps(value, indent=2) + "\n"))


def _atomic_write(path: Path, write: Callable[[str], Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        write(name)
        with open(name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(row: dict) -> tuple[str, str]:
    if not row.get("example_id") or not row.get("dataset_name"):
        raise ValueError("Metric shards require example_id and dataset_name on every row")
    return str(row["dataset_name"]), str(row["example_id"])


class MetricsShardWriter:
    """One writer per rank. Manifests commit immutable Parquet parts atomically.

    Reopening an interrupted writer with ``resume=True`` appends parts; completed
    runs are immutable. The caller remains responsible for retrying missing work.
    """

    def __init__(
        self, directory: Path, rank: int, world_size: int, *, expected_batches: int, resume: bool = False
    ) -> None:
        if world_size < 1 or not 0 <= rank < world_size or expected_batches < 0:
            raise ValueError("Invalid rank, world size, or expected batch count")
        self.directory = Path(directory) / f"rank{rank:06d}"
        self.path = self.directory / "manifest.json"
        if self.path.exists():
            if not resume or (self.directory / "DONE.json").exists():
                raise FileExistsError(f"Preserving existing metrics: {self.directory}")
            self.manifest = json.loads(self.path.read_text())
            if (self.manifest["rank"], self.manifest["world_size"], self.manifest["expected_batches"]) != (
                rank,
                world_size,
                expected_batches,
            ):
                raise ValueError("Cannot resume metrics with a different rank or batch allocation")
        else:
            self.manifest = {
                "version": 1,
                "rank": rank,
                "world_size": world_size,
                "expected_batches": expected_batches,
                "completed_batches": [],
                "parts": [],
            }
            _atomic_json(self.path, self.manifest)

    def write(self, phase: str, rows: list[dict]) -> None:
        if not rows:
            return
        if phase not in {"design", "refold", "final"}:
            raise ValueError(f"Unknown metrics phase: {phase}")
        if (self.directory / "DONE.json").exists():
            raise RuntimeError("Completed metrics are immutable")
        keys = [_key(row) for row in rows]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate candidate identities in one metrics part")
        name = f"{len(self.manifest['parts']):08d}_{phase}.parquet"
        path = self.directory / name
        _atomic_write(path, lambda tmp: pd.DataFrame(rows).to_parquet(tmp, engine="pyarrow", index=False))
        self.manifest["parts"].append({"file": name, "phase": phase, "rows": len(rows), "sha256": _digest(path)})
        _atomic_json(self.path, self.manifest)

    def batch_complete(self, batch_id: str) -> None:
        if batch_id in self.manifest["completed_batches"]:
            raise ValueError(f"Batch already committed: {batch_id}")
        self.manifest["completed_batches"].append(batch_id)
        _atomic_json(self.path, self.manifest)

    def complete(self) -> None:
        if len(self.manifest["completed_batches"]) != self.manifest["expected_batches"]:
            raise ValueError("Cannot complete metrics with missing generation batches")
        _atomic_json(self.directory / "DONE.json", {"manifest_sha256": _digest(self.path)})


def read_metrics_shards(directory: Path, *, allow_partial: bool = False) -> tuple[pd.DataFrame, dict]:
    """Read committed parts in rank/part order, joining phases by candidate identity.

    Accept one run directory or a tree of run directories. A later invocation
    replaces an earlier candidate row as a whole, so stale refold columns cannot
    leak into a retry that has only produced design metrics.
    """
    directory = Path(directory)
    runs = sorted(directory.rglob("run.json"), key=lambda p: (p.stat().st_mtime_ns, str(p)))
    if not runs:
        raise ValueError(f"No metrics shard run manifests found in {directory}")
    merged: dict[tuple[str, str], dict] = {}
    report: dict[str, Any] = {"complete": True, "missing": [], "runs": len(runs), "repeated_candidates": 0}
    for run_path in runs:
        run = json.loads(run_path.read_text())
        if run.get("version") != 1 or run["world_size"] < 1:
            raise ValueError(f"Unsupported run manifest: {run_path}")
        seen: set[tuple[str, str]] = set()
        for rank in range(run["world_size"]):
            folder = run_path.parent / f"rank{rank:06d}"
            path, done = folder / "manifest.json", folder / "DONE.json"
            if not path.exists():
                report["missing"].append(f"{folder}: missing rank manifest")
                continue
            manifest = json.loads(path.read_text())
            if manifest["version"] != 1 or (manifest["rank"], manifest["world_size"]) != (rank, run["world_size"]):
                raise ValueError(f"Inconsistent rank manifest: {path}")
            if not done.exists():
                report["missing"].append(
                    f"{folder}: incomplete rank "
                    f"({len(manifest['completed_batches'])}/{manifest['expected_batches']} batches)"
                )
            elif json.loads(done.read_text())["manifest_sha256"] != _digest(path):
                raise ValueError(f"Completion record does not match manifest: {folder}")
            elif len(manifest["completed_batches"]) != manifest["expected_batches"]:
                raise ValueError(f"Completed rank has missing batches: {folder}")
            stages: dict[str, dict] = {phase: {} for phase in ("design", "refold", "final")}
            for part in manifest["parts"]:
                name = part["file"]
                if Path(name).name != name or part["phase"] not in stages:
                    raise ValueError(f"Invalid part in {path}")
                part_path = folder / name
                if not part_path.is_file() or _digest(part_path) != part["sha256"]:
                    raise ValueError(f"Missing or corrupted metrics part: {part_path}")
                import pyarrow.parquet as pq

                # Read exactly this file: output paths such as seed=42 must not
                # be interpreted as Hive partitions and override metric columns.
                rows = pq.ParquetFile(part_path).read().to_pylist()
                if len(rows) != part["rows"]:
                    raise ValueError(f"Wrong row count: {part_path}")
                for row in rows:
                    stages[part["phase"]][_key(row)] = row
            keys = dict.fromkeys(k for phase in stages.values() for k in phase)
            for key in keys:
                if key in seen:
                    raise ValueError(f"Candidate assigned to multiple ranks in one run: {key}")
                seen.add(key)
                row = {**stages["design"].get(key, {}), **stages["refold"].get(key, {})}
                row = stages["final"].get(key, row)
                report["repeated_candidates"] += int(key in merged)
                merged[key] = row
    report["complete"] = not report["missing"]
    report["candidate_count"] = len(merged)
    if not report["complete"] and not allow_partial:
        raise ValueError(
            "Incomplete inference metrics; use allow_partial=True to inspect committed results:\n"
            + "\n".join(report["missing"])
        )
    return pd.DataFrame(merged.values()), report


def create_metrics_run(directory: Path, world_size: int, metadata: dict) -> None:
    _atomic_json(Path(directory) / "run.json", dict(version=1, world_size=world_size, **metadata))


def try_aggregate_metrics_run(directory: Path, output_dir: Path) -> bool:
    """The last finisher publishes this invocation's CSVs, without rank waiting.

    Requires a shared filesystem supporting atomic rename and advisory file locks.
    The lock is released on process exit; failed merges can be retried. Completed
    shards remain immutable and available for the standalone merger.
    """
    import fcntl

    directory, output_dir = Path(directory), Path(output_dir)
    run = json.loads((directory / "run.json").read_text())
    if not all((directory / f"rank{rank:06d}" / "DONE.json").is_file() for rank in range(run["world_size"])):
        return False
    with (directory / "merge.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        marker = directory / "MERGED.json"
        if marker.exists():
            return False
        from rfproteina.inference_utils.aggregate_metrics import concatenate_results

        output_dir.mkdir(parents=True, exist_ok=True)
        # Publish only after all aggregation succeeds. Never clean up other jobs'
        # CSVs; their shards are combined separately by the offline merger.
        with tempfile.TemporaryDirectory(prefix=".merge-", dir=output_dir) as staging:
            concatenate_results(str(directory), output_dir=staging, cleanup=False, sequence_probability_workers=1)
            files = sorted(Path(staging).iterdir())
            for path in files:
                os.replace(path, output_dir / path.name)
            _atomic_json(marker, {"output_dir": str(output_dir.resolve()), "files": [p.name for p in files]})
        return True
