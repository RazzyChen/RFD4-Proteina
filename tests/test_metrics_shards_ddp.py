"""Real Lightning lifecycle: a late rank must not time out metrics aggregation."""

import json
import os
import time
from datetime import timedelta

import lightning as L
import pytest
import torch
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import rfproteina.proteina as module
from rfproteina.inference_utils.metrics_shards import read_metrics_shards
from rfproteina.metrics.utils.metrics_utils import MetricsPool
from rfproteina.proteina import Proteina, _design_key


def _forbidden(*args, **kwargs):
    raise AssertionError("File metrics must not use cross-rank gather or distributed Foldseek")


def _post_fold(entry, oracle, refold, criteria, prior_design_metrics_by_id):
    prior = prior_design_metrics_by_id[entry["example_id"]]
    assert len(prior_design_metrics_by_id) == 1
    return {**prior, "oracle.score": prior["design.score"] * 2, "ppi.pass": float(prior["design.score"] > 2)}


class ShardedMetricsProbe(L.LightningModule):
    """Use production batch/epoch hooks, with small deterministic metric rows."""

    on_validation_batch_end = Proteina.on_validation_batch_end
    on_validation_epoch_end = Proteina.on_validation_epoch_end
    _collect_design_metrics = Proteina._collect_design_metrics
    _gather_metrics = _forbidden

    def __init__(self, directory, delay, mixed=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.store_dir = str(directory)
        self.delay = delay
        self.mixed = mixed
        self.is_inference = True
        self.metrics_aggregation = "files"
        self.job_id = self.global_prefix = self.ckpt_path = self.ckpt_name = None
        self.cfg_exp = OmegaConf.create({})
        self.search_strategy = None
        self.val_dataset_tags = None
        self.design_metric_manager = None
        self.oracle_metric_manager = None
        self.refolding_metric_manager = None
        self.pass_criteria_metric_manager = None
        self.pre_oracle_filter_metric_manager = None
        self._metrics_pool = MetricsPool(0, {})
        self._metrics_shards = None

    def on_validation_epoch_start(self):
        Proteina.on_validation_epoch_start(self)
        self.originals = module.compute_oracle_metrics_for_val_output, module.run_foldseek_distributed
        module.compute_oracle_metrics_for_val_output = _post_fold
        module.run_foldseek_distributed = _forbidden

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.global_rank == 1 and dataloader_idx == 0 and batch_idx == (1 if self.mixed else 2):
            time.sleep(self.delay)
        for value in batch.tolist():
            row = dict(
                dataset_name="probe",
                task="probe",
                example_id=f"sample{value}",
                output_base_dir=self.store_dir,
                **{"design.score": float(value)},
            )
            self.validation_output_generation.append(row)
            self._shard_design_entries[_design_key(row)] = row

    def on_validation_end(self):
        module.compute_oracle_metrics_for_val_output, module.run_foldseek_distributed = self.originals
        (self._metrics_shards.directory / "finished_at.json").write_text(json.dumps(time.time()))


@pytest.mark.parametrize("devices,mixed", [(1, False), (2, False), (2, True)])
def test_lightning_late_rank_finishes_without_end_collectives(tmp_path, devices, mixed):
    # Set RFD4_SHARD_TEST_CUDA=1 to run the same check with two real CUDA devices/NCCL.
    cuda = os.environ.get("RFD4_SHARD_TEST_CUDA") == "1"
    if cuda and torch.cuda.device_count() < devices:
        pytest.skip("Not enough CUDA devices")
    timeout, delay = 5, 9
    strategy = (
        DDPStrategy(
            start_method="spawn", process_group_backend="nccl" if cuda else "gloo", timeout=timedelta(seconds=timeout)
        )
        if devices > 1
        else "auto"
    )
    trainer = L.Trainer(
        accelerator="gpu" if cuda else "cpu",
        devices=devices,
        strategy=strategy,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    loaders = (
        [DataLoader(list(range(4)), batch_size=1), DataLoader(list(range(4, 8)), batch_size=1)]
        if mixed
        else DataLoader(list(range(6)), batch_size=1)
    )
    trainer.validate(ShardedMetricsProbe(tmp_path, delay, mixed), loaders, verbose=False)
    frame, report = read_metrics_shards(tmp_path / "inference_metrics" / "metrics_shards")
    count = 8 if mixed else 6
    assert report["complete"] and len(frame) == count
    indexed = frame.set_index("example_id")
    for index in range(count):
        assert indexed.loc[f"sample{index}", "oracle.score"] == index * 2
        assert indexed.loc[f"sample{index}", "ppi.pass"] == float(index > 2)
    if devices == 2:
        root = next((tmp_path / "inference_metrics" / "metrics_shards").iterdir())
        times = [json.loads((root / f"rank{rank:06d}" / "finished_at.json").read_text()) for rank in range(2)]
        assert times[1] - times[0] > timeout
