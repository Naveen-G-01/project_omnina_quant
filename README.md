# Project Omnia: Entropy-Driven Mixed-Precision Quantization

Entropy-guided mixed-precision post-training quantization (PTQ) for CNNs. Built as a
drop-in replacement for [HAQ](https://github.com/mit-han-lab/haq)'s RL-based
bit-width search: instead of training a DDPG agent (`lib/rl/ddpg.py`,
`lib/env/*.py`) to pick per-layer bit-widths, Project Omnia computes each
layer's activation Shannon entropy `H(X_l)` from a single calibration pass and
assigns bit-widths with a deterministic threshold rule:

```
H(X_l) >= tau  ->  high_bits   (e.g. INT8 weight/activation)
H(X_l) <  tau  ->  low_bits    (e.g. INT4 weight/activation)
```

`tau` is chosen by sweeping a set of candidate values and picking the most
aggressive one (highest fraction of low-bit layers) that still satisfies the
accuracy and bit-width constraints described below.

## What's reused from HAQ vs. new in this repo

| Reused unchanged from HAQ | New in Project Omnia |
|---|---|
| `lib/utils/quantize_utils.py` (`QConv2d`, `QLinear`, `calibrate`) | `entropy_quantize.py` (entry point) |
| `lib/utils/data_utils.py` (`get_dataset`) | `lib/utils/entropy_utils.py` (entropy collection, tau sweep) |
| `lib/utils/utils.py` (`Logger`, `AverageMeter`, `accuracy`) | |
| `models/*` (`mobilenet.py`, `mobilenetv2.py`, `mobilenetv3.py`) | |

`lib/rl/ddpg.py` and `lib/env/*.py` are **not** used by `entropy_quantize.py`;
`rl_quantise.py` (HAQ's original RL search) and `pretrain.py` (FP32 training)
are kept in the repo for reference/comparison but are separate pipelines.

## Repo layout

```
entropy_quantize.py          # entry point for entropy-driven PTQ
rl_quantise.py                # original HAQ RL search (unchanged, for comparison)
pretrain.py                   # FP32 training/finetuning (unchanged)
models/
  __init__.py
  mobilenet.py                 # mobilenet / qmobilenet
  mobilenetv2.py                # mobilenetv2 / qmobilenetv2
  mobilenetv3.py                 # mobilenet_v3 (see limitation below)
lib/
  utils/
    quantize_utils.py         # QConv2d, QLinear, calibrate (HAQ, unchanged)
    data_utils.py               # get_dataset (HAQ, unchanged)
    utils.py                      # Logger, AverageMeter, accuracy (HAQ, unchanged)
    entropy_utils.py               # entropy collection + tau sweep (new)
```

`lib/__init__.py` and `lib/utils/__init__.py` must exist (can be empty) for
the package imports to resolve.

## Requirements

- Python 3, PyTorch + torchvision (with CUDA for realistic run times)
- `numpy`, `scikit-learn` (used by `quantize_utils.py`'s k-means path),
  `progress` (`from progress.bar import Bar`)
- An ImageNet-format dataset (`train/` and `val/` subfolders of class directories)
- A FP32 checkpoint for the architecture you want to quantize

## Usage

```bash
python entropy_quantize.py \
    --arch qmobilenetv2 \
    --dataset imagenet --dataset_root data/imagenet \
    --resume checkpoints/mobilenetv2/mobilenetv2-150.pth.tar \
    --calib_size 100 --tau_steps 25 \
    --max_acc_drop 0.8 --min_low_bit_frac 0.6 \
    --output save/omnia_mobilenetv2
```

For faster iteration while tuning `tau_steps` or the accuracy/bit-width
constraints, add `--eval_subset 5000` (or similar) so each tau candidate is
evaluated on a subset during the sweep; the final chosen `tau` is always
re-evaluated on the full validation set before the report is written.

### Key arguments

| Argument | Default | Meaning |
|---|---|---|
| `--arch` | `qmobilenetv2` | Model architecture (see supported architectures below) |
| `--resume` | *(required)* | Path to the FP32 checkpoint to quantize |
| `--calib_size` | 100 | Number of images in the calibration stream |
| `--calib_batch` | 25 | Batch size for the calibration stream |
| `--tau_steps` | 25 | Number of tau candidates swept between the observed min/max entropy |
| `--w_bit_low` / `--a_bit_low` | 4 / 4 | Weight/activation bits for layers below tau |
| `--w_bit_high` / `--a_bit_high` | 8 / 8 | Weight/activation bits for layers at/above tau |
| `--max_acc_drop` | 0.8 | Max acceptable Top-1 drop (percentage points) vs. FP32 |
| `--min_low_bit_frac` | 0.6 | Min fraction of layers required at the low-bit setting |
| `--eval_subset` | `None` | If set, subsample validation during the tau sweep for speed |

Run `python entropy_quantize.py --help` for the full list.

## Supported architectures

Only architectures with a quantized (`QConv2d`/`QLinear`-based) variant work
with `entropy_quantize.py`, since the entropy/bit-assignment pipeline needs
`w_bit`/`a_bit` attributes to write to:

- `qmobilenet` (`models/mobilenet.py`)
- `qmobilenetv2` (`models/mobilenetv2.py`)

**`mobilenetv3.py` does not currently have a quantized variant** — its
`MobileBottleneck` block hardcodes `conv_layer = nn.Conv2d` and never imports
`QConv2d`/`QLinear`, so there is no `qmobilenet_v3` entry point. `--arch
qmobilenetv3` will not resolve. Adding one would mean threading a
`conv_layer` argument through `MobileNetV3`/`MobileBottleneck` the way
`mobilenet.py`/`mobilenetv2.py` already do.

## Output

Each run writes to `<output>/`:

- `omnia_report.json` — FP32/quantized Top-1, accuracy drop, chosen tau,
  per-layer entropy and bit-width strategy, compression estimate, and
  PASS/FAIL against the `--max_acc_drop` / `--min_low_bit_frac` values used
  for that run (plus the fixed 2.2x compression / 180s calibration-time
  targets from the project brief).
- `omnia_quantized.pth.tar` — `state_dict`, chosen `strategy`, and `tau`.
- `log.txt` — per-tau sweep log (tau, acc, acc_drop, low_bit_frac, feasible).

## Known limitations

- **Compression numbers are a bit-width estimate, not an on-disk size
  reduction.** `QConv2d`/`QLinear` fake-quantize on the fly (straight-through
  estimator) and always store full FP32-shaped weight tensors — the
  `compression_vs_int8`/`compression_vs_fp32` figures in the report describe
  the theoretical footprint *if* the chosen bit-widths were packed, not the
  actual size of `omnia_quantized.pth.tar`. Real latency/memory numbers
  require wiring in `lib/simulator/lookup_tables` (`HAS_HW_SIMULATOR`); the
  weight-size estimate is a fallback.
- **`calibrate()` in `quantize_utils.py` assumes a CUDA device** (it moves
  the calibration batch to `cuda:0` unconditionally). CPU-only runs of
  `entropy_quantize.py` will fail at the calibration step even though the
  rest of the script checks `torch.cuda.is_available()`. This function is
  unchanged from HAQ; patch it locally if you need CPU support.
- The tau sweep re-calibrates and re-evaluates the model once per candidate
  (`--tau_steps`, default 25); use `--eval_subset` to keep this affordable
  on the full ImageNet validation set.
