#!/usr/bin/env python3
"""
Find where the deployment path diverges from the training path.

check_fit.py calls the action head directly with cached VLM embeddings and
reproduces ground truth well. deploy.py goes through MiniCPMVLAInference.predict
with live images and produces almost nothing. Something between those two paths
is wrong.

This runs the SAME recorded frame through both and prints the difference:

    A  cached embedding      -> head.predict_action        (the check_fit path)
    B  recorded jpg on disk  -> m.predict(...)             (the deploy path)
    C  recorded jpg, state=0 -> m.predict(...)             (state ablation)

    A == B   -> both paths agree; the problem is the live camera or the
                arm being outside the demonstrated workspace
    A != B   -> m.predict differs from the direct call; most likely the state
                argument is not reaching the head the way training assumed

Usage:
    python compare_paths.py
    python compare_paths.py --episode ep0000 --frames 179 251 361
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/radtesting/MiniCPM-Robot/MiniCPM-RobotManip")
sys.path.insert(0, "/home/radtesting/franka_ws")
from vla_infer import MiniCPMVLAInference  # noqa: E402
from train import (DEMO_ROOT, OUT_DIR, CKPT, load_episode,  # noqa: E402
                   make_vectors, denorm_dpos, build_transitions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="ep0000")
    ap.add_argument("--frames", type=int, nargs="*", default=None)
    args = ap.parse_args()

    stats = json.load(open(os.path.join(OUT_DIR, "norm_stats.json")))
    blob = torch.load(os.path.join(OUT_DIR, "action_head.pt"), map_location="cpu")
    eid = blob["embodiment_id"]
    print(f"embodiment {eid}")

    d = os.path.join(DEMO_ROOT, args.episode)
    task, pos, aa, grip = load_episode(d)
    act, state = make_vectors(pos, aa, grip, stats)
    dpos_gt, _, _ = build_transitions(pos, aa, grip)

    if args.frames:
        idx = args.frames
    else:
        moving = np.where(np.linalg.norm(dpos_gt, axis=1) > 0.004)[0]
        idx = moving[np.linspace(0, len(moving) - 1, 5).astype(int)].tolist()

    cache = torch.load(os.path.join(OUT_DIR, "cache", f"{args.episode}.pt"))
    embs = cache["embs"]
    cached_task = cache["task"]
    print(f"task string in cache: {cached_task!r}")
    print(f"task string from traj: {task!r}")
    if cached_task != task:
        print("  WARNING: task strings differ; embeddings will not match")

    m = MiniCPMVLAInference(checkpoint_path=CKPT, device="cuda")
    head = m.model.action_head
    head.load_state_dict(blob["state_dict"], strict=False)
    head.eval()
    dev = next(head.parameters()).device
    dtype = next(head.parameters()).dtype
    eid_t = torch.tensor([eid], dtype=torch.long, device=dev)

    print(f"\n{'frame':>6} {'ground truth':>20} {'A direct':>20} "
          f"{'B m.predict':>20} {'C no state':>20}")

    for i in idx:
        gt = denorm_dpos(act[i, 7:10], stats) * 1000

        # A: cached embedding, direct head call
        with torch.no_grad():
            e = embs[i:i + 1].to(dev, dtype)
            s = torch.from_numpy(state[i]).to(dev, dtype).view(1, 1, 80)
            a = head.predict_action(e, s, eid_t)[0].float().cpu().numpy()
        A = denorm_dpos(a[0, 7:10], stats) * 1000

        # B: recorded images through the full predict path, same state
        p_img = os.path.join(d, "primary", f"{i:05d}.jpg")
        w_img = os.path.join(d, "wrist", f"{i:05d}.jpg")
        b = m.predict(images=[p_img, w_img], text=task,
                      state=state[i], embodiment_id=eid, seed=0).numpy()
        B = denorm_dpos(b[0, 7:10], stats) * 1000

        # C: same but with a zero state vector
        c = m.predict(images=[p_img, w_img], text=task,
                      state=np.zeros(80, np.float32),
                      embodiment_id=eid, seed=0).numpy()
        C = denorm_dpos(c[0, 7:10], stats) * 1000

        f = lambda v: np.array2string(v, precision=2, suppress_small=True)
        print(f"{i:>6} {f(gt):>20} {f(A):>20} {f(B):>20} {f(C):>20}")

    # Also compare the embeddings themselves for one frame.
    i = idx[0]
    p_img = os.path.join(d, "primary", f"{i:05d}.jpg")
    w_img = os.path.join(d, "wrist", f"{i:05d}.jpg")
    with torch.no_grad():
        inp = m.preprocess([p_img, w_img], task)
        live = m.model._vlm_forward(inp).hidden_states[-1][0].float().cpu()
    cached = embs[i].float()
    print(f"\nembedding check, frame {i}:")
    print(f"  cached shape {tuple(cached.shape)}, recomputed {tuple(live.shape)}")
    if cached.shape == live.shape:
        print(f"  max abs difference: {(cached - live).abs().max():.5f}")
        print(f"  mean abs value:     {cached.abs().mean():.5f}")

    print("\n=== reading ===")
    print("  A close to ground truth, B near zero -> m.predict is the problem")
    print("  A and B both close to ground truth   -> the recorded path is fine;")
    print("                                          deploy fails on live input")
    print("  B close to C                          -> state is being ignored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
