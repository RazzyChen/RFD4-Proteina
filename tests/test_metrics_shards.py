"""Inference metrics survive skew and interruption without cross-rank gathers."""

import json
import multiprocessing as mp
from concurrent.futures import Future
from pathlib import Path

import pandas as pd
import pytest

from rfproteina.inference_utils.aggregate_metrics import concatenate_results
from rfproteina.inference_utils.metrics_shards import MetricsShardWriter, create_metrics_run, read_metrics_shards
from rfproteina.metrics.utils.metrics_utils import MetricsPool


def _row(index, **values):
    return dict(
        dataset_name="target", example_id=f"rep{index}", task="binder", output_base_dir="/test/samples", **values
    )


def test_rank_and_phase_merge_preserves_precision_and_optional_columns(tmp_path):
    create_metrics_run(tmp_path, 2, {})
    for rank in range(2):
        writer = MetricsShardWriter(tmp_path, rank, 2, expected_batches=2)
        for batch in range(2):
            i = rank + 2 * batch
            writer.write("design", [_row(i, **{"search.reward": i + 0.123456789})])
            writer.batch_complete(str(batch))
        # Refolds arrive in reverse order; one candidate has a missing optional metric.
        for i in (rank + 2, rank):
            writer.write("refold", [_row(i, **{"ppi.pass": i % 2, "rmsd": None if i == 2 else i + 0.25})])
        writer.complete()
    frame, report = read_metrics_shards(tmp_path)
    assert frame.example_id.tolist() == ["rep0", "rep2", "rep1", "rep3"]
    assert report["complete"] and report["candidate_count"] == 4
    assert frame.set_index("example_id").loc["rep2", "search.reward"] == 2.123456789
    assert pd.isna(frame.set_index("example_id").loc["rep2", "rmsd"])
    pd.testing.assert_frame_equal(frame, read_metrics_shards(tmp_path)[0])
    concatenate_results(str(tmp_path), sequence_probability_workers=1)
    result = pd.read_csv(tmp_path / "per_example_metrics.csv")
    assert set(result.example_id) == set(frame.example_id)
    assert len(result) == 4
    concatenate_results(str(tmp_path), sequence_probability_workers=1)
    assert len(pd.read_csv(tmp_path / "per_example_metrics.csv")) == 4


def test_interrupted_rank_can_be_inspected_and_resumed_without_duplicates(tmp_path):
    create_metrics_run(tmp_path, 2, {})
    first = MetricsShardWriter(tmp_path, 0, 2, expected_batches=1)
    first.write("final", [_row(0, value=3.0)])
    first.batch_complete("0")
    first.complete()
    late = MetricsShardWriter(tmp_path, 1, 2, expected_batches=2)
    late.write("design", [_row(1, value=4.0)])
    late.batch_complete("0")
    # No DONE: equivalent to interruption after a committed batch.
    with pytest.raises(ValueError, match="Incomplete"):
        read_metrics_shards(tmp_path)
    partial, report = read_metrics_shards(tmp_path, allow_partial=True)
    assert len(partial) == 2 and not report["complete"]
    with pytest.raises(ValueError, match="missing generation batches"):
        late.complete()
    resumed = MetricsShardWriter(tmp_path, 1, 2, expected_batches=2, resume=True)
    resumed.write("design", [_row(1, value=4.0), _row(3, value=6.0)])
    resumed.batch_complete("1")
    resumed.complete()
    final, report = read_metrics_shards(tmp_path)
    assert report["complete"]
    assert final.set_index("example_id").value.to_dict() == {"rep0": 3.0, "rep1": 4.0, "rep3": 6.0}


def test_corrupt_part_is_not_silently_skipped_even_in_partial_mode(tmp_path):
    create_metrics_run(tmp_path, 1, {})
    writer = MetricsShardWriter(tmp_path, 0, 1, expected_batches=0)
    writer.write("design", [_row(0, value=1.0)])
    next(writer.directory.glob("*.parquet")).write_bytes(b"truncated")
    with pytest.raises(ValueError, match="corrupted"):
        read_metrics_shards(tmp_path, allow_partial=True)


def test_uncommitted_parts_and_failed_writes_do_not_become_results(tmp_path, monkeypatch):
    create_metrics_run(tmp_path, 1, {})
    writer = MetricsShardWriter(tmp_path, 0, 1, expected_batches=0)

    def fail(self, path, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    with pytest.raises(OSError, match="disk full"):
        writer.write("design", [_row(0, value=1.0)])
    assert not list(writer.directory.glob("*.tmp"))
    assert json.loads(writer.path.read_text())["parts"] == []
    # An unreferenced file cannot be mistaken for a committed part.
    (writer.directory / "99999999_design.parquet").write_bytes(b"uncommitted")
    assert read_metrics_shards(tmp_path, allow_partial=True)[0].empty


def test_missing_rank_and_duplicate_cross_rank_assignment_are_reported(tmp_path):
    create_metrics_run(tmp_path, 2, {})
    writer = MetricsShardWriter(tmp_path, 0, 2, expected_batches=0)
    writer.write("final", [_row(0, value=1.0)])
    writer.complete()
    assert "missing rank manifest" in read_metrics_shards(tmp_path, allow_partial=True)[1]["missing"][0]
    other = MetricsShardWriter(tmp_path, 1, 2, expected_batches=0)
    other.write("final", [_row(0, value=1.0)])
    other.complete()
    with pytest.raises(ValueError, match="multiple ranks"):
        read_metrics_shards(tmp_path)


def test_same_example_id_in_different_targets_is_preserved(tmp_path):
    create_metrics_run(tmp_path, 1, {})
    writer = MetricsShardWriter(tmp_path, 0, 1, expected_batches=0)
    row = _row(0, value=1.0)
    writer.write("final", [row, {**row, "dataset_name": "other_target"}])
    writer.complete()
    assert len(read_metrics_shards(tmp_path)[0]) == 2


def test_output_directory_is_not_interpreted_as_a_parquet_partition(tmp_path):
    directory = tmp_path / "seed=42"
    create_metrics_run(directory, 1, {})
    writer = MetricsShardWriter(directory, 0, 1, expected_batches=0)
    writer.write("final", [_row(0, seed=42.0, value=1.0)])
    writer.complete()
    frame, report = read_metrics_shards(directory)
    assert report["complete"] and frame.seed.tolist() == [42.0]


def test_new_attempt_does_not_inherit_old_refold_values(tmp_path):
    for name, phase, row in [("first", "final", _row(0, confidence=90.0)), ("retry", "design", _row(0, value=2.0))]:
        directory = tmp_path / name
        create_metrics_run(directory, 1, {})
        writer = MetricsShardWriter(directory, 0, 1, expected_batches=0)
        writer.write(phase, [row])
        writer.complete()
    frame, report = read_metrics_shards(tmp_path)
    assert "confidence" not in frame
    assert frame.value.iloc[0] == 2.0
    assert report["repeated_candidates"] == 1


def test_poll_keeps_pending_cpu_metrics_and_final_drain_gets_them():
    pool = MetricsPool(0, {})
    ready, pending = Future(), Future()
    ready.set_result({"value": 1.0})
    pool._futures = [("first", ready), ("second", pending)]
    assert pool.drain_generated_structure(wait=False) == {"first": {"value": 1.0}}
    assert pool.drain_generated_structure(wait=False) == {}
    pending.set_result({"value": 2.0})
    assert pool.drain_generated_structure() == {"second": {"value": 2.0}}


def _interrupted_worker(directory, ready, release):
    writer = MetricsShardWriter(Path(directory), 0, 1, expected_batches=2)
    writer.write("design", [_row(0, value=1.0)])
    writer.batch_complete("0")
    ready.set()
    release.wait()


def test_process_termination_preserves_committed_batch_and_resume_matches_control(tmp_path):
    create_metrics_run(tmp_path, 1, {})
    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    process = ctx.Process(target=_interrupted_worker, args=(str(tmp_path), ready, release))
    process.start()
    try:
        assert ready.wait(45), "Worker never committed its first batch"
        process.terminate()
        process.join(10)
        assert not process.is_alive()
        partial, report = read_metrics_shards(tmp_path, allow_partial=True)
        assert not report["complete"] and partial.example_id.tolist() == ["rep0"]
        resumed = MetricsShardWriter(tmp_path, 0, 1, expected_batches=2, resume=True)
        resumed.write("design", [_row(1, value=2.0)])
        resumed.batch_complete("1")
        resumed.complete()
        final, report = read_metrics_shards(tmp_path)
        assert report["complete"]
        pd.testing.assert_frame_equal(final, pd.DataFrame([_row(0, value=1.0), _row(1, value=2.0)]))
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
