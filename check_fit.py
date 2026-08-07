#!/usr/bin/env python3
"""
Check whether the fine-tuned action head actually learned the demonstrations.

Runs the trained head on cached TRAINING embeddings and compares its output
against the ground-truth actions from the same frames.

    predictions track ground truth  ->  the model learned; the problem is in
                                        deploy.py (state encoding, camera, etc.)
    predictions are ~0 everywhere   ->  the model collapsed to the mean, which
                                        happens when most training frames are
                                        near-stationary

Usage:
    python check_fit.py
    python check_fit.py --episode ep0003 --n 12
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/radtesting/MiniCPM-Robot/MiniCPM-RobotManip")
sys.path.insert(0, "/home/radtesting/franka_ws")
from vla_infer import MiniCPMVLAInference  # noqa: E402
from train import (DEMO_ROOT, OUT_DIR, CKPT, load_episode,  # noqa: E402
                   make_vectors, chunk, denorm_dpos, build_transitions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default=None, help="default: first episode")
    ap.add_argument("--n", type=int, default=10, help="frames to sample")
    args = ap.parse_args()

    stats = json.load(open(os.path.join(OUT_DIR, "norm_stats.json")))
    blob = torch.load(os.path.join(OUT_DIR, "action_head.pt"), map_location="cpu")
    eid = blob["embodiment_id"]

    # ---- how much of the training data is actually motion? -----------------
    dirs = sorted(glob.glob(os.path.join(DEMO_ROOT, "ep*")))
    all_mag = []
    for d in dirs:
        _, pos, aa, grip = load_episode(d)
        dpos, _, _ = build_transitions(pos, aa, grip)
        all_mag.append(np.linalg.norm(dpos, axis=1))
    mag = np.concatenate(all_mag)
    print("=== training data motion ===")
    print(f"  frames:                 {len(mag)}")
    print(f"  median |delta|:         {np.median(mag)*1000:.2f} mm")
    print(f"  mean   |delta|:         {mag.mean()*1000:.2f} mm")
    print(f"  frames under 0.5 mm:    {(mag < 0.0005).mean()*100:.0f}%")
    print(f"  frames under 1.0 mm:    {(mag < 0.001).mean()*100:.0f}%")
    print(f"  frames over  5.0 mm:    {(mag > 0.005).mean()*100:.0f}%")

    # ---- run the trained head on cached training embeddings ----------------
    ep = args.episode or os.path.basename(dirs[0])
    d = os.path.join(DEMO_ROOT, ep)
    cache = torch.load(os.path.join(OUT_DIR, "cache", f"{ep}.pt"))
    embs = cache["embs"]

    task, pos, aa, grip = load_episode(d)
    act, state = make_vectors(pos, aa, grip, stats)

    print(f"\nloading model ...")
    m = MiniCPMVLAInference(checkpoint_path=CKPT, device="cuda")
    head = m.model.action_head
    head.load_state_dict(blob["state_dict"], strict=False)
    head.eval()
    dev = next(head.parameters()).device
    dtype = next(head.parameters()).dtype

    # Sample frames biased toward ones where the arm was actually moving.
    dpos_gt, _, _ = build_transitions(pos, aa, grip)
    moving = np.where(np.linalg.norm(dpos_gt, axis=1) > 0.002)[0]
    if len(moving) < args.n:
        moving = np.arange(len(pos) - 1)
    idx = moving[np.linspace(0, len(moving) - 1, args.n).astype(int)]

    print(f"\n=== {ep}: predicted vs ground-truth first-step delta (mm) ===")
    print(f"{'frame':>6}  {'ground truth':>22}  {'predicted':>22}  {'err':>6}")
    errs, gts, prs = [], [], []
    with torch.no_grad():
        for i in idx:
            e = embs[i:i + 1].to(dev, dtype)
            s = torch.from_numpy(state[i]).to(dev, dtype).view(1, 1, 80)
            eid_t = torch.tensor([eid], dtype=torch.long, device=dev)
            out = head.predict_action(e, s, eid_t)[0].float().cpu().numpy()

            pr = denorm_dpos(out[0, 7:10], stats) * 1000.0
            gt = denorm_dpos(act[i, 7:10], stats) * 1000.0
            err = np.linalg.norm(pr - gt)
            errs.append(err); gts.append(gt); prs.append(pr)
            print(f"{i:>6}  {np.array2string(gt, precision=2, suppress_small=True):>22}"
                  f"  {np.array2string(pr, precision=2, suppress_small=True):>22}"
                  f"  {err:>6.2f}")

    gts, prs, errs = np.array(gts), np.array(prs), np.array(errs)
    print(f"\n  mean |error|:        {errs.mean():.2f} mm")
    print(f"  mean |ground truth|: {np.linalg.norm(gts, axis=1).mean():.2f} mm")
    print(f"  mean |prediction|:   {np.linalg.norm(prs, axis=1).mean():.2f} mm")
    for k, axis in enumerate("xyz"):
        if gts[:, k].std() > 1e-6 and prs[:, k].std() > 1e-6:
            r = np.corrcoef(gts[:, k], prs[:, k])[0, 1]
            print(f"  correlation {axis}:       {r:+.2f}")
        else:
            print(f"  correlation {axis}:       n/a (no variance)")

    print("\n=== reading ===")
    if np.linalg.norm(prs, axis=1).mean() < 0.3 * np.linalg.norm(gts, axis=1).mean():
        print("  Predictions are far smaller than ground truth: the head collapsed")
        print("  toward the mean action. Most training frames are near-stationary,")
        print("  so 'do almost nothing' is a low-loss solution. Fix by subsampling")
        print("  the demos to a lower control rate (larger deltas per step) and/or")
        print("  recording with steadier, more deliberate motion.")
    elif errs.mean() < 2.0:
        print("  The head reproduces the training actions well. The near-zero output")
        print("  in deploy.py is therefore a deployment mismatch - most likely the")
        print("  state encoding (deploy centres position on the startup pose, while")
        print("  training centred on each episode's mean).")
    else:
        print("  Partial fit: predictions have the right scale but poor accuracy.")
        print("  More demonstrations would be the main lever.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
