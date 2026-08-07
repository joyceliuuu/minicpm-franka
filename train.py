#!/usr/bin/env python3
"""
Fine-tune MiniCPM-RobotManip's action head on locally recorded demonstrations.

The VLM is frozen, so its output embeddings are computed once and cached to
disk; training then runs only the action head against those cached tensors.
For a few thousand transitions this turns a multi-hour job into a few minutes.

Stages:
    python train.py cache      # run the VLM over every frame, save embeddings
    python train.py fit        # train the action head on the cache
    python train.py both       # cache then fit

Outputs (under OUT_DIR):
    cache/ep####.pt      cached vl_embs per episode
    norm_stats.json      normalization statistics derived from your data
    action_head.pt       fine-tuned action head weights

Action layout written into the 80-dim vector (indices 7..16):
    7:10   delta end-effector position, min-max normalized to [-1, 1]
    10:16  delta rotation as 6D (first two columns of the rotation matrix)
    16     gripper command, -1 open / +1 closed

State layout uses the same indices with absolute pose instead of deltas.
"""

import argparse
import json
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, "/home/radtesting/MiniCPM-Robot/MiniCPM-RobotManip")
from vla_infer import MiniCPMVLAInference  # noqa: E402

DEMO_ROOT = "/opt/models/MiniCPM-RobotManip/demos"
OUT_DIR = "/opt/models/MiniCPM-RobotManip/finetune"
CKPT = "openbmb/MiniCPM-RobotManip"

EMBODIMENT_ID = 31          # "new_embodiment" slot; leaves LIBERO's eid 25 intact
HORIZON = 30                # action chunk length, fixed by the architecture
A0, A1 = 7, 17              # active slice of the 80-dim action/state vector


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_episode(d):
    """Read one episode directory into arrays."""
    j = json.load(open(os.path.join(d, "traj.json")))
    frames = j["frames"] if isinstance(j, dict) else j
    task = j.get("task", "pick up the box and place it on the plate") \
        if isinstance(j, dict) else "pick up the box and place it on the plate"
    pos = np.array([f["pos"] for f in frames], dtype=np.float64)
    aa = np.array([f["aa"] for f in frames], dtype=np.float64)
    grip = np.array([f["grip"] for f in frames], dtype=np.float64)
    return task, pos, aa, grip


def rot6d(mat):
    """First two columns of a rotation matrix, flattened. Naturally in [-1, 1]."""
    return mat[:, :2].T.reshape(-1)


def rot6d_to_matrix(v):
    """Gram-Schmidt the 6D representation back to a rotation matrix."""
    a1, a2 = v[:3], v[3:]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - (b1 @ a2) * b1
    b2 /= np.linalg.norm(b2)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


def build_transitions(pos, aa, grip):
    """
    Per-frame raw actions: delta position (3), delta rotation matrix, gripper.
    The final frame repeats the last action so every frame has a target.
    """
    n = len(pos)
    dpos = np.zeros((n, 3))
    drot = np.tile(np.eye(3), (n, 1, 1))
    dpos[:-1] = np.diff(pos, axis=0)
    for i in range(n - 1):
        Rc = R.from_rotvec(aa[i]).as_matrix()
        Rn = R.from_rotvec(aa[i + 1]).as_matrix()
        drot[i] = Rn @ Rc.T
    dpos[-1] = dpos[-2] if n > 1 else 0.0
    drot[-1] = drot[-2] if n > 1 else np.eye(3)
    return dpos, drot, grip


def compute_stats(episodes):
    """1st/99th percentile of delta position across the whole dataset."""
    allد = np.concatenate([build_transitions(p, a, g)[0] for _, p, a, g in episodes])
    lo = np.percentile(allد, 1, axis=0)
    hi = np.percentile(allد, 99, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    allpos = np.concatenate([p for _, p, _, _ in episodes])
    return {"dpos_min": lo.tolist(), "dpos_max": hi.tolist(),
            "dpos_span": span.tolist(),
            "pos_center": allpos.mean(0).tolist(),
            "pos_scale": 0.3}


def norm_dpos(dpos, st):
    lo = np.array(st["dpos_min"])
    span = np.array(st["dpos_span"])
    return np.clip(2.0 * (dpos - lo) / span - 1.0, -1.0, 1.0)


def denorm_dpos(x, st):
    lo = np.array(st["dpos_min"])
    span = np.array(st["dpos_span"])
    return (x + 1.0) / 2.0 * span + lo


def make_vectors(pos, aa, grip, st):
    """Build the (n, 80) action and state matrices for one episode."""
    n = len(pos)
    dpos, drot, g = build_transitions(pos, aa, grip)
    dpos_n = norm_dpos(dpos, st)

    act = np.zeros((n, 80), dtype=np.float32)
    act[:, 7:10] = dpos_n
    act[:, 10:16] = np.stack([rot6d(m) for m in drot])
    act[:, 16] = g * 2.0 - 1.0

    # State: absolute pose, roughly centred so the encoder sees sane magnitudes.
    state = np.zeros((n, 80), dtype=np.float32)
    center = np.array(st["pos_center"])
    scale = st.get("pos_scale", 0.3)
    state[:, 7:10] = np.clip((pos - center) / scale, -1, 1)
    state[:, 10:16] = np.stack([rot6d(R.from_rotvec(a).as_matrix()) for a in aa])
    state[:, 16] = g * 2.0 - 1.0
    return act, state


def chunk(act, i, horizon=HORIZON):
    """Action chunk starting at i, padded by repeating the final action."""
    n = len(act)
    idx = np.minimum(np.arange(i, i + horizon), n - 1)
    return act[idx]


# --------------------------------------------------------------------------
# stage 1: cache VLM embeddings
# --------------------------------------------------------------------------

def stage_cache():
    os.makedirs(os.path.join(OUT_DIR, "cache"), exist_ok=True)
    dirs = sorted(glob.glob(os.path.join(DEMO_ROOT, "ep*")))
    if not dirs:
        print(f"no episodes under {DEMO_ROOT}")
        return 1

    print(f"loading model on {'cuda' if torch.cuda.is_available() else 'cpu'} ...")
    m = MiniCPMVLAInference(checkpoint_path=CKPT, device="cuda")
    m.model.eval()

    for d in dirs:
        name = os.path.basename(d)
        outfile = os.path.join(OUT_DIR, "cache", f"{name}.pt")
        if os.path.exists(outfile):
            print(f"{name}: cached already, skipping")
            continue

        task, pos, aa, grip = load_episode(d)
        n = len(pos)
        embs = []
        with torch.no_grad():
            for i in range(n):
                p_img = os.path.join(d, "primary", f"{i:05d}.jpg")
                w_img = os.path.join(d, "wrist", f"{i:05d}.jpg")
                inp = m.preprocess([w_img], task)
                out = m.model._vlm_forward(inp)
                embs.append(out.hidden_states[-1][0].to(torch.bfloat16).cpu())
                if i % 50 == 0:
                    print(f"  {name}: {i}/{n}", end="\r")

        lens = {e.shape[0] for e in embs}
        if len(lens) != 1:
            print(f"\n{name}: WARNING inconsistent sequence lengths {lens}; "
                  f"padding to max")
            L = max(lens)
            embs = [F.pad(e, (0, 0, 0, L - e.shape[0])) for e in embs]

        torch.save({"embs": torch.stack(embs), "task": task}, outfile)
        print(f"  {name}: cached {n} frames, seq_len {embs[0].shape[0]}      ")

    print("cache complete")
    return 0


# --------------------------------------------------------------------------
# stage 2: train the action head
# --------------------------------------------------------------------------

def stage_fit(epochs, batch_size, lr, val_frac):
    dirs = sorted(glob.glob(os.path.join(DEMO_ROOT, "ep*")))
    episodes = [load_episode(d) for d in dirs]

    stats = compute_stats(episodes)
    json.dump(stats, open(os.path.join(OUT_DIR, "norm_stats.json"), "w"), indent=1)
    print("normalization stats written")
    print(f"  dpos_min {np.round(stats['dpos_min'], 4)}")
    print(f"  dpos_max {np.round(stats['dpos_max'], 4)}")

    # Assemble every (embedding, state, action-chunk) triple.
    E, S, A = [], [], []
    for d, (task, pos, aa, grip) in zip(dirs, episodes):
        name = os.path.basename(d)
        cf = os.path.join(OUT_DIR, "cache", f"{name}.pt")
        if not os.path.exists(cf):
            print(f"{name}: no cache, run 'cache' first")
            return 1
        embs = torch.load(cf)["embs"]
        act, state = make_vectors(pos, aa, grip, stats)
        n = min(len(embs), len(act))
        for i in range(n):
            E.append(embs[i])
            S.append(torch.from_numpy(state[i]))
            A.append(torch.from_numpy(chunk(act, i)))
    E = torch.stack(E)
    S = torch.stack(S)
    A = torch.stack(A)
    print(f"dataset: {len(E)} samples, embedding {tuple(E.shape[1:])}, "
          f"chunk {tuple(A.shape[1:])}")

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(E), generator=g)
    n_val = max(1, int(len(E) * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    print("loading model ...")
    m = MiniCPMVLAInference(checkpoint_path=CKPT, device="cuda")
    head = m.model.action_head
    head.train()

    for p in m.model.vlm.parameters():           # VLM stays frozen
        p.requires_grad_(False)
    params = [p for p in head.parameters() if p.requires_grad]
    print(f"training {sum(p.numel() for p in params)/1e6:.1f}M action-head params")

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    dev = next(head.parameters()).device
    dtype = next(head.parameters()).dtype
    eid = torch.full((batch_size,), EMBODIMENT_ID, dtype=torch.long, device=dev)

    # Loss is computed only on the active dims; padding dims stay at init,
    # matching how the released checkpoint was evidently trained.
    mask = torch.zeros(80, device=dev, dtype=dtype)
    mask[A0:A1] = 1.0

    def run_batch(idx, train=True):
        e = E[idx].to(dev, dtype)
        s = S[idx].to(dev, dtype).unsqueeze(1)
        a = A[idx].to(dev, dtype)
        b = len(idx)

        t = torch.rand(b, device=dev, dtype=dtype)
        noise = torch.randn_like(a)
        noisy = t[:, None, None] * noise + (1 - t[:, None, None]) * a
        buckets = (t.float() * head.num_timestep_buckets).long() \
            .clamp(0, head.num_timestep_buckets - 1)

        pred = head._predict(noisy, e, s, buckets, eid[:b])
        loss = (((pred - a) ** 2) * mask).sum() / (mask.sum() * b * HORIZON)

        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        return loss.item()

    print(f"\ntraining {epochs} epochs, batch {batch_size}, lr {lr}")
    for ep in range(epochs):
        head.train()
        order = train_idx[torch.randperm(len(train_idx))]
        losses = []
        for k in range(0, len(order) - batch_size + 1, batch_size):
            losses.append(run_batch(order[k:k + batch_size], train=True))

        head.eval()
        with torch.no_grad():
            vl = [run_batch(val_idx[k:k + batch_size], train=False)
                  for k in range(0, len(val_idx) - batch_size + 1, batch_size)]
        sched.step()
        v = np.mean(vl) if vl else float("nan")
        print(f"  epoch {ep+1:3d}/{epochs}  train {np.mean(losses):.5f}  val {v:.5f}")

    torch.save({"state_dict": head.state_dict(),
                "embodiment_id": EMBODIMENT_ID,
                "stats": stats},
               os.path.join(OUT_DIR, "action_head.pt"))
    print(f"\nsaved {os.path.join(OUT_DIR, 'action_head.pt')}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cache", "fit", "both"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if a.stage in ("cache", "both"):
        if stage_cache() != 0:
            return 1
    if a.stage in ("fit", "both"):
        return stage_fit(a.epochs, a.batch_size, a.lr, a.val_frac)
    return 0


if __name__ == "__main__":
    sys.exit(main())
