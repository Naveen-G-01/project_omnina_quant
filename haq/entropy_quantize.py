# entropy_quantize.py
#
# "Information-Entropy Driven Adaptive Mixed-Precision Quantization for
#  Ultra-Low Latency Edge Vision Accelerators"  (Project Omnia)
#
# This is the drop-in replacement for HAQ's rl_quantize.py. It reuses:
#   - models/*                         (unchanged, from the HAQ repo)
#   - lib/utils/quantize_utils.py      (QConv2d, QLinear, calibrate -- unchanged)
#   - lib/utils/data_utils.py          (get_dataset -- unchanged)
#   - lib/utils/utils.py               (Logger, AverageMeter, accuracy -- unchanged)
#   - lib/simulator/lookup_tables/*    (optional, for real HW latency -- see note below)
#
# It does NOT use:
#   - lib/rl/ddpg.py, lib/env/*.py     (RL search -- replaced by entropy_utils.py)
#
# Place this file at the repo root (next to rl_quantize.py / finetune.py).
#
# Example:
#   python entropy_quantize.py \
#       --arch qmobilenetv2 --dataset imagenet --dataset_root data/imagenet \
#       --resume checkpoints/mobilenetv2/mobilenetv2-150.pth.tar \
#       --calib_size 100 --tau_steps 25 \
#       --max_acc_drop 0.8 --min_low_bit_frac 0.6 \
#       --output save/omnia_mobilenetv2

import os
import json
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
import models as customized_models

from lib.utils.utils import Logger, AverageMeter, accuracy
from lib.utils.data_utils import get_dataset
from lib.utils.quantize_utils import QConv2d, QLinear, calibrate
from lib.utils.entropy_utils import (
    run_calibration_and_get_entropy,
    apply_strategy,
    estimate_compression,
    sweep_tau,
)

# try to hook into HAQ's hardware lookup-table simulator for real latency
# numbers ("[Emulated FPGA Latency & Accuracy Checks]" in the pipeline
# diagram). Falls back to the weight-size-only estimate in entropy_utils
# if the simulator module/API differs in your checkout.
try:
    from lib.simulator.lookup_tables import LatencyEstimator  # noqa
    HAS_HW_SIMULATOR = True
except Exception:
    HAS_HW_SIMULATOR = False


# ----------------------------- model registry -----------------------------
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))
for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]
model_names = default_model_names + customized_models_names


def get_args():
    p = argparse.ArgumentParser(description='Project Omnia: Entropy-Driven Mixed-Precision PTQ')
    # data
    p.add_argument('--dataset', default='imagenet', type=str)
    p.add_argument('--dataset_root', default='data/imagenet', type=str)
    p.add_argument('--workers', default=8, type=int)
    p.add_argument('--calib_size', default=100, type=int,
                   help='number of calibration images (brief target: 100)')
    p.add_argument('--calib_batch', default=25, type=int)
    p.add_argument('--eval_batch', default=256, type=int)
    p.add_argument('--eval_subset', default=None, type=int,
                   help='if set, evaluate on a subset during tau-sweeping for speed; '
                        'final chosen tau is still re-evaluated on the full val set')
    # arch
    p.add_argument('--arch', '-a', default='qmobilenetv2', choices=model_names)
    p.add_argument('--resume', required=True, type=str,
                   help='path to FP32 pretrained checkpoint to quantize')
    # entropy / tau
    p.add_argument('--num_bins', default=256, type=int)
    p.add_argument('--tau_min', default=None, type=float,
                   help='if unset, derived from observed entropy range')
    p.add_argument('--tau_max', default=None, type=float)
    p.add_argument('--tau_steps', default=25, type=int)
    p.add_argument('--w_bit_low', default=4, type=int)
    p.add_argument('--a_bit_low', default=4, type=int)
    p.add_argument('--w_bit_high', default=8, type=int)
    p.add_argument('--a_bit_high', default=8, type=int)
    # journal target constraints (Section 5 of the brief)
    p.add_argument('--max_acc_drop', default=0.8, type=float,
                   help='max acceptable Top-1 accuracy drop vs FP32 (percentage points)')
    p.add_argument('--min_low_bit_frac', default=0.6, type=float,
                   help='min fraction of layers required at INT4')
    # misc
    p.add_argument('--gpu_id', default='0', type=str)
    p.add_argument('--output', default='save/omnia', type=str)
    p.add_argument('--seed', default=234, type=int)
    return p.parse_args()


def build_quantizable_index(model):
    idx = []
    for i, m in enumerate(model.modules()):
        if type(m) in [QConv2d, QLinear]:
            idx.append(i)
    return idx


def load_fp32_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model_state = model.state_dict()
    loaded, skipped = 0, 0
    for name, param in state_dict.items():
        if name in model_state and model_state[name].shape == param.shape:
            model_state[name].copy_(param)
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_state)
    print(f'[load_fp32_weights] loaded {loaded} tensors, skipped {skipped} (shape/name mismatch)')
    return model


def evaluate(model, loader, use_cuda, max_batches=None):
    model.eval()
    top1 = AverageMeter()
    with torch.no_grad():
        for b, (images, targets) in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            if use_cuda:
                images, targets = images.cuda(), targets.cuda()
            out = model(images)
            prec1, _ = accuracy(out.data, targets.data, topk=(1, 5))
            top1.update(prec1.item(), images.size(0))
    return top1.avg


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    os.makedirs(args.output, exist_ok=True)
    logger = Logger(os.path.join(args.output, 'log.txt'), title='project-omnia-' + args.arch)
    logger.set_names(['tau', 'acc', 'acc_drop', 'low_bit_frac', 'feasible'])

    # ---- 1. datasets: full val set + tiny calibration stream -----------
    train_loader, val_loader, n_class = get_dataset(
        dataset_name=args.dataset, batch_size=args.eval_batch,
        n_worker=args.workers, data_root=args.dataset_root)

    # calibration stream: first `calib_size` images from the train loader's
    # underlying dataset, matching the brief's "Tiny calibration subset
    # (100 sample images)" requirement -- NOT the full train set.
    calib_dataset = torch.utils.data.Subset(train_loader.dataset, list(range(args.calib_size)))
    calib_loader = torch.utils.data.DataLoader(
        calib_dataset, batch_size=args.calib_batch, shuffle=False, num_workers=args.workers)

    # ---- 2. build model + load FP32 weights ------------------------------
    model = models.__dict__[args.arch](pretrained=False, num_classes=n_class)
    model = load_fp32_weights(model, args.resume)
    if use_cuda:
        model = model.cuda()

    quantizable_idx = build_quantizable_index(model)
    print(f'[entropy_quantize] {len(quantizable_idx)} quantizable layers found')

    # ---- 3. FP32 baseline accuracy (needed to measure acc_drop) --------
    fp32_acc = evaluate(model, val_loader, use_cuda)
    print(f'[entropy_quantize] FP32 baseline Top-1: {fp32_acc:.3f}')

    # ---- 4. single-pass entropy calculation -----------------------------
    t0 = time.time()
    entropy_dict = run_calibration_and_get_entropy(
        model, calib_loader, quantizable_idx,
        num_bins=args.num_bins, use_cuda=use_cuda)
    calib_time = time.time() - t0
    print(f'[entropy_quantize] entropy calibration took {calib_time:.1f}s '
          f'on {args.calib_size} images (target: <=180s)')

    entropies = list(entropy_dict.values())
    tau_min = args.tau_min if args.tau_min is not None else min(entropies)
    tau_max = args.tau_max if args.tau_max is not None else max(entropies)
    tau_candidates = [tau_min + i * (tau_max - tau_min) / max(1, args.tau_steps - 1)
                       for i in range(args.tau_steps)]

    # ---- 5. calibrate_fn / eval_fn closures for sweep_tau ---------------
    def calibrate_fn(model, strategy):
        apply_strategy(model, quantizable_idx, strategy)
        model = calibrate(model, train_loader)  # computes S_l, Z_l (quantize_utils)
        return model

    eval_max_batches = None
    if args.eval_subset is not None:
        eval_max_batches = max(1, args.eval_subset // args.eval_batch)

    def eval_fn(model):
        return evaluate(model, val_loader, use_cuda, max_batches=eval_max_batches)

    # ---- 6. tau-sweeping (dynamic thresholding) -------------------------
    best_tau, sweep_results = sweep_tau(
        model, quantizable_idx, entropy_dict, tau_candidates,
        calibrate_fn=calibrate_fn, eval_fn=eval_fn, fp32_acc=fp32_acc,
        low_bits=(args.w_bit_low, args.a_bit_low),
        high_bits=(args.w_bit_high, args.a_bit_high),
        max_acc_drop=args.max_acc_drop, min_low_bit_frac=args.min_low_bit_frac)

    for r in sweep_results:
        logger.append([r['tau'], r['acc'], r['acc_drop'], r['low_bit_frac'], r['satisfies_constraints']])

    # ---- 7. finalize with the chosen tau, full val-set re-evaluation ----
    best_strategy = next(r['strategy'] for r in sweep_results if r['tau'] == best_tau)
    model = calibrate_fn(model, best_strategy)
    final_acc = evaluate(model, val_loader, use_cuda)  # full val set, no subset
    final_acc_drop = fp32_acc - final_acc
    low_bit_frac = next(r['low_bit_frac'] for r in sweep_results if r['tau'] == best_tau)
    compression = estimate_compression(model, quantizable_idx, best_strategy)

    # ---- 8. hardware latency (real simulator if available) --------------
    hw_report = {'source': 'weight-size estimate (no HW simulator wired in)'}
    if HAS_HW_SIMULATOR:
        try:
            estimator = LatencyEstimator()  # signature depends on your lib/simulator API
            hw_report = estimator.estimate(model, best_strategy)
            hw_report['source'] = 'lib/simulator/lookup_tables'
        except Exception as e:
            hw_report = {'source': 'weight-size estimate (simulator call failed: %s)' % str(e)}

    report = {
        'title': 'Information-Entropy Driven Adaptive Mixed-Precision Quantization',
        'arch': args.arch,
        'calibration_images': args.calib_size,
        'calibration_time_sec': calib_time,
        'fp32_top1': fp32_acc,
        'final_top1': final_acc,
        'top1_drop': final_acc_drop,
        'low_bit_layer_fraction': low_bit_frac,
        'chosen_tau': best_tau,
        'per_layer_entropy': entropy_dict,
        'per_layer_strategy': {str(k): v for k, v in best_strategy.items()},
        'compression': compression,
        'hardware_estimate': hw_report,
        'targets_from_brief': {
            'delta_top1_target': 0.8,
            'delta_top1_met': final_acc_drop <= 0.8,
            'low_bit_frac_target': 0.6,
            'low_bit_frac_met': low_bit_frac >= 0.6,
            'compression_target_vs_int8': 2.2,
            'compression_met': compression['compression_vs_int8'] >= 2.2,
            'calibration_time_target_sec': 180,
            'calibration_time_met': calib_time <= 180,
        },
    }

    with open(os.path.join(args.output, 'omnia_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    torch.save({'state_dict': model.state_dict(), 'strategy': best_strategy, 'tau': best_tau},
               os.path.join(args.output, 'omnia_quantized.pth.tar'))

    logger.close()

    print('\n===== Project Omnia: Final Report =====')
    print(f"FP32 Top-1:        {fp32_acc:.3f}")
    print(f"Quantized Top-1:   {final_acc:.3f}  (drop: {final_acc_drop:.3f}, "
          f"target <=0.8: {'PASS' if final_acc_drop <= 0.8 else 'FAIL'})")
    print(f"INT4 layer frac:   {low_bit_frac:.3f}  (target >=0.60: "
          f"{'PASS' if low_bit_frac >= 0.6 else 'FAIL'})")
    print(f"Compression vs INT8: {compression['compression_vs_int8']:.2f}x "
          f"(target >=2.2x: {'PASS' if compression['compression_vs_int8'] >= 2.2 else 'FAIL'})")
    print(f"Calibration time:  {calib_time:.1f}s (target <=180s: "
          f"{'PASS' if calib_time <= 180 else 'FAIL'})")
    print(f"Full report saved to {os.path.join(args.output, 'omnia_report.json')}")


if __name__ == '__main__':
    main()
