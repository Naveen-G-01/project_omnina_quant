"""
Regression test for lib/simulator/lookup_tables.py's LatencyEstimator.

Guards two fixes made to the original version:
1. Real per-layer output spatial size (via forward hooks) replaced a flat
   28x28-for-every-QConv2d assumption that could be off by 1-2 orders of
   magnitude and invert which layer looked more expensive.
2. The 'bottleneck' field now compares the actual accumulated
   compute/memory cycle totals, instead of re-deriving an approximate
   figure with a single flat INT8 rate that ignored per-layer INT4/INT8
   mixing already applied earlier in the same function.

Run: pytest tests/test_latency_estimator.py -v
"""
import torch
import torch.nn as nn
import pytest

from lib.simulator.lookup_tables import LatencyEstimator
from lib.utils.quantize_utils import QConv2d, QLinear


class _SpatiallyVaryingNet(nn.Module):
    """One layer that stays at a large spatial resolution, one that's been
    downsampled to a small one -- same weight count, very different real
    compute cost. A flat spatial-area assumption can't tell them apart."""
    def __init__(self):
        super().__init__()
        self.early = QConv2d(3, 16, 3, stride=1, padding=1)   # 224x224 out
        self.down = nn.MaxPool2d(16)                            # -> 14x14
        self.late = QConv2d(16, 16, 3, stride=1, padding=1)    # 14x14 out
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = QLinear(16, 10)

    def forward(self, x):
        x = self.early(x)
        x = self.down(x)
        x = self.late(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _quantizable_idx(model):
    return [i for i, m in enumerate(model.modules()) if type(m) in [QConv2d, QLinear]]


def test_every_estimate_is_labeled_unmeasured():
    """The paper's own Section IV-E rule: never report a modeled number as
    measured. Enforce it at the data-structure level, not just in prose."""
    model = _SpatiallyVaryingNet().eval()
    idx = _quantizable_idx(model)
    est = LatencyEstimator()
    result = est.estimate(model, {i: (8, 8) for i in idx})
    assert result['is_measured'] is False


def test_quantizing_the_expensive_layer_saves_more_than_the_cheap_one():
    """This is the core correctness property the flat-28x28 version got
    backwards: pushing the layer with the larger real output map to INT4
    must save more cycles than pushing the smaller one, because it's
    doing more actual work."""
    model = _SpatiallyVaryingNet().eval()
    idx = _quantizable_idx(model)
    early_idx, late_idx = idx[0], idx[1]
    est = LatencyEstimator()

    baseline = est.estimate(model, {i: (8, 8) for i in idx})

    strat_early_int4 = {i: (8, 8) for i in idx}
    strat_early_int4[early_idx] = (4, 4)
    early_int4 = est.estimate(model, strat_early_int4)

    strat_late_int4 = {i: (8, 8) for i in idx}
    strat_late_int4[late_idx] = (4, 4)
    late_int4 = est.estimate(model, strat_late_int4)

    saved_early = baseline['total_cycles'] - early_int4['total_cycles']
    saved_late = baseline['total_cycles'] - late_int4['total_cycles']
    assert saved_early > saved_late


def test_bottleneck_label_is_consistent_with_reported_cycles():
    """The bottleneck field must reflect the same compute/memory split
    that actually produced total_cycles -- not an independently
    recomputed approximation using a single flat rate."""
    model = _SpatiallyVaryingNet().eval()
    idx = _quantizable_idx(model)
    # Force a memory-bound regime: tiny compute rate, huge bandwidth,
    # so memory_cycles should exceed compute_cycles for every layer.
    est_memory_bound = LatencyEstimator(peak_macs_per_cycle_int8=10**9,
                                         peak_macs_per_cycle_int4=10**9,
                                         bandwidth_bytes_per_cycle=1)
    r = est_memory_bound.estimate(model, {i: (8, 8) for i in idx})
    assert r['bottleneck'] == 'Memory'

    # And the reverse: tiny bandwidth denominator flip -- huge compute
    # rate denominator, tiny bandwidth -- forces compute-bound.
    est_compute_bound = LatencyEstimator(peak_macs_per_cycle_int8=1,
                                          peak_macs_per_cycle_int4=1,
                                          bandwidth_bytes_per_cycle=10**9)
    r2 = est_compute_bound.estimate(model, {i: (8, 8) for i in idx})
    assert r2['bottleneck'] == 'Compute'


def test_missing_strategy_entries_are_skipped_not_errored():
    model = _SpatiallyVaryingNet().eval()
    idx = _quantizable_idx(model)
    est = LatencyEstimator()
    partial = {idx[0]: (8, 8)}  # only one of the layers specified
    r = est.estimate(model, partial)
    assert r['total_cycles'] > 0
