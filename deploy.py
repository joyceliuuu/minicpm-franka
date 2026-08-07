#!/usr/bin/env python3
"""
Run the fine-tuned MiniCPM-RobotManip action head on the Franka.

Unlike vla_loop.py (which unit-normalized the direction because the scale was
unknown), this denormalizes with the statistics generated from your own
demonstrations, so the deltas are in real metres.

Usage:
    python deploy.py                       # dry run, prints actions, no motion
    python deploy.py --live                # actually move the arm
    python deploy.py --live --exec-steps 5 --speed 0.05

Camera geometry must match the recordings exactly. If the tripod moved since
recording, retrain before using this.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import panda_py
import pyrealsense2 as rs
import torch
from PIL import Image
from panda_py import libfranka
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, "/home/radtesting/MiniCPM-Robot/MiniCPM-RobotManip")
from vla_infer import MiniCPMVLAInference  # noqa: E402

FT_DIR = "/opt/models/MiniCPM-RobotManip/finetune"
CKPT = "openbmb/MiniCPM-RobotManip"
BRIO_INDEX = 6
IMG_SIZE = (256, 256)

# Workspace bounds (metres, robot base frame). Motion outside this aborts.
SAFE_X = (0.20, 0.70)
SAFE_Y = (-0.40, 0.40)
SAFE_Z = (0.05, 0.65)

# Refuse to execute a single step larger than this, whatever the model says.
MAX_STEP_M = 0.03


def rot6d_to_matrix(v):
    a1, a2 = v[:3], v[3:]
    n1 = np.linalg.norm(a1)
    if n1 < 1e-8:
        return np.eye(3)
    b1 = a1 / n1
    b2 = a2 - (b1 @ a2) * b1
    n2 = np.linalg.norm(b2)
    if n2 < 1e-8:
        return np.eye(3)
    b2 /= n2
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.8.201")
    ap.add_argument("--text", default="pick up the box and place it on the plate")
    ap.add_argument("--iters", type=int, default=15, help="replanning iterations")
    ap.add_argument("--exec-steps", type=int, default=8,
                    help="steps of each 30-step chunk to execute before replanning")
    ap.add_argument("--speed", type=float, default=0.05)
    ap.add_argument("--embodiment-id", type=int, default=None,
                    help="override; defaults to the id stored in the checkpoint")
    ap.add_argument("--live", action="store_true", help="actually move the robot")
    args = ap.parse_args()

    # --- load fine-tuned head ----------------------------------------------
    ckpt_path = os.path.join(FT_DIR, "action_head.pt")
    if not os.path.exists(ckpt_path):
        print(f"no fine-tuned head at {ckpt_path}; run train.py fit first")
        return 1
    blob = torch.load(ckpt_path, map_location="cpu")
    stats = blob.get("stats") or json.load(open(os.path.join(FT_DIR, "norm_stats.json")))
    eid = args.embodiment_id if args.embodiment_id is not None else blob["embodiment_id"]

    lo = np.array(stats["dpos_min"])
    span = np.array(stats["dpos_span"])
    print(f"embodiment {eid}  |  dpos range {np.round(lo, 4)} .. "
          f"{np.round(lo + span, 4)} m")

    print("loading model ...")
    m = MiniCPMVLAInference(checkpoint_path=CKPT, device="cuda")
    missing, unexpected = m.model.action_head.load_state_dict(
        blob["state_dict"], strict=False)
    if missing or unexpected:
        print(f"  state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    m.model.action_head.eval()
    print("fine-tuned head loaded")

    # --- cameras ------------------------------------------------------------
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)
    brio = cv2.VideoCapture(BRIO_INDEX, cv2.CAP_V4L2)
    brio.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    brio.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    brio.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    brio.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(20):
        brio.read()

    # --- robot --------------------------------------------------------------
    p = panda_py.Panda(args.ip)
    g = libfranka.Gripper(args.ip)
    p.get_robot().automatic_error_recovery()
    p.get_robot().set_collision_behavior(
        [50.0] * 7, [50.0] * 7, [30.0] * 6, [30.0] * 6)

    # State normalization must match training: position centred on the
    # demonstration mean. Approximated here by the current pose at startup.
    center = np.array(stats["pos_center"])
    scale = stats.get("pos_scale", 0.3)
    print(f"state reference (from demos): {np.round(center, 3)}")
    grip_state = {"v": 0.0}

    def observe():
        for _ in range(2):
            brio.read()
        ok, bf = brio.read()
        wrist = np.asanyarray(pipe.wait_for_frames().get_color_frame().get_data())[:, :, ::-1]
        Image.fromarray(bf[:, :, ::-1]).resize(IMG_SIZE).save("/tmp/dp_primary.jpg")
        Image.fromarray(wrist).resize(IMG_SIZE).save("/tmp/dp_wrist.jpg")

        s = np.zeros(80, np.float32)
        pos = p.get_position()
        aa = R.from_quat(p.get_orientation()).as_rotvec()
        s[7:10] = np.clip((pos - center) / scale, -1, 1)
        s[10:16] = R.from_rotvec(aa).as_matrix()[:, :2].T.reshape(-1)
        s[16] = grip_state["v"] * 2.0 - 1.0
        return s, pos

    print(f"\n{'LIVE' if args.live else 'DRY RUN'} | {args.iters} iterations, "
          f"executing {args.exec_steps} steps each\n")

    try:
        for it in range(args.iters):
            state, cur = observe()
            act = m.predict(images=["/tmp/dp_wrist.jpg"],
                            text=args.text, state=state,
                            embodiment_id=eid, seed=0).numpy()

            blk = act[:, 7:17]
            dpos = (blk[:, :3] + 1.0) / 2.0 * span + lo          # metres
            grip_cmd = blk[:, 9]

            mag = np.linalg.norm(dpos[:args.exec_steps], axis=1)
            print(f"[{it:2d}] at={np.round(cur, 3)}  "
                  f"step0={np.round(dpos[0], 4)}  "
                  f"|step| {mag.min():.4f}-{mag.max():.4f} m  "
                  f"grip={grip_cmd[0]:+.2f}")

            if not args.live:
                continue

            for k in range(args.exec_steps):
                d = dpos[k]
                if np.linalg.norm(d) > MAX_STEP_M:
                    print(f"  step {k}: {np.linalg.norm(d):.3f} m exceeds limit, stop")
                    return 0
                tgt = p.get_position() + d
                if not (SAFE_X[0] < tgt[0] < SAFE_X[1]
                        and SAFE_Y[0] < tgt[1] < SAFE_Y[1]
                        and SAFE_Z[0] < tgt[2] < SAFE_Z[1]):
                    print(f"  target {np.round(tgt,3)} outside safe box, stop")
                    return 0

                drot = rot6d_to_matrix(blk[k, 3:9])
                rv = R.from_matrix(drot).as_rotvec()
                ang = np.linalg.norm(rv)
                if ang > 0.05:                      # cap at ~3 degrees per step
                    rv = rv / ang * 0.05
                drot = R.from_rotvec(rv).as_matrix()
                cur_R = R.from_quat(p.get_orientation()).as_matrix()
                new_q = R.from_matrix(drot @ cur_R).as_quat()

                p.move_to_pose(position=tgt, orientation=new_q,
                               speed_factor=args.speed)

                want = 1.0 if grip_cmd[k] > 0 else 0.0
                if want != grip_state["v"]:
                    grip_state["v"] = want
                    if want:
                        g.grasp(0.0, 0.1, 40, 0.08, 0.08)
                    else:
                        g.move(0.08, 0.1)
                    time.sleep(0.3)

    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        pipe.stop()
        brio.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())
