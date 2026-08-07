#!/usr/bin/env python3
"""
Smooth keyboard teleoperation for demonstration recording.

The earlier version called move_to_pose every tick, which is a blocking
point-to-point planner: it accelerated from rest and stopped again on every
keypress, so the motion was jerky and laggy.

This holds a CartesianImpedance controller open instead. The controller runs
its own 1 kHz loop and continuously tracks a target pose, so updating that
target at 15 Hz produces continuous motion. Keys nudge the target; the arm
flows toward it.

Controls (no Enter needed):
    w / s      +x / -x
    a / d      +y / -y
    q / e      +z / -z
    j / l      yaw
    i / k      pitch
    space      toggle gripper
    [ / ]      slower / faster
    ENTER      save and exit
    ESC        abort without saving

Usage:
    python teleop2.py 100
    python teleop2.py 100 --rate 15 --speed 60
"""

import argparse
import json
import os
import select
import sys
import termios
import time
import tty

import cv2
import numpy as np
import panda_py
import pyrealsense2 as rs
from PIL import Image
from panda_py import controllers, libfranka
from scipy.spatial.transform import Rotation as R

DEMO_ROOT = "/opt/models/MiniCPM-RobotManip/demos"
BRIO_INDEX = 6
IMG_SIZE = (256, 256)

SAFE_X = (0.25, 0.68)
SAFE_Y = (-0.38, 0.38)
SAFE_Z = (0.06, 0.62)

TRANSLATE = {
    "w": np.array([1.0, 0, 0]), "s": np.array([-1.0, 0, 0]),
    "a": np.array([0, 1.0, 0]), "d": np.array([0, -1.0, 0]),
    "q": np.array([0, 0, 1.0]), "e": np.array([0, 0, -1.0]),
}
ROTATE = {
    "j": np.array([0, 0, -1.0]), "l": np.array([0, 0, 1.0]),
    "i": np.array([0, -1.0, 0]), "k": np.array([0, 1.0, 0]),
}


class RawKeyboard:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def get(self):
        keys = []
        while select.select([sys.stdin], [], [], 0)[0]:
            keys.append(sys.stdin.read(1))
        return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", type=int)
    ap.add_argument("--ip", default="192.168.8.201")
    ap.add_argument("--rate", type=float, default=15.0, help="logging rate, Hz")
    ap.add_argument("--speed", type=float, default=60.0,
                    help="target velocity while a key is held, mm/s")
    ap.add_argument("--rot-speed", type=float, default=0.5,
                    help="target angular velocity, rad/s")
    ap.add_argument("--stiffness", type=float, default=400.0,
                    help="translational impedance stiffness (N/m)")
    ap.add_argument("--task", default="pick up the box and place it on the plate")
    ap.add_argument("--no-log", action="store_true",
                    help="drive the arm without recording (for practice)")
    args = ap.parse_args()

    out = os.path.join(DEMO_ROOT, f"ep{args.episode:04d}")
    if not args.no_log:
        if os.path.exists(out):
            print(f"{out} exists; delete it or choose another number")
            return 1
        os.makedirs(os.path.join(out, "primary"))
        os.makedirs(os.path.join(out, "wrist"))

    # --- cameras ------------------------------------------------------------
    pipe = brio = None
    if not args.no_log:
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
        [80.0] * 7, [80.0] * 7, [60.0] * 6, [60.0] * 6)
    g.move(0.08, 0.1)

    # 6x6 impedance: translational stiffness on the diagonal, rotational lower.
    imp = np.eye(6)
    imp[:3, :3] *= args.stiffness
    imp[3:, 3:] *= args.stiffness / 10.0

    ctrl = controllers.CartesianImpedance(impedance=imp, damping_ratio=1.0,
                                          nullspace_stiffness=0.5,
                                          filter_coeff=1.0)
    p.start_controller(ctrl)
    time.sleep(0.5)

    # The target the controller chases. Keys move this; the arm follows.
    tgt_pos = p.get_position().copy()
    tgt_quat = p.get_orientation().copy()

    period = 1.0 / args.rate
    lin_per_tick = args.speed / 1000.0 * period      # metres per tick
    ang_per_tick = args.rot_speed * period
    speed_scale = 1.0

    grip = 0.0
    rec, buf_p, buf_w = [], [], []
    aborted = False

    print(f"""
teleop episode {args.episode:04d}   ({args.rate:.0f} Hz, {args.speed:.0f} mm/s)

    w/s  x      a/d  y      q/e  z
    j/l  yaw    i/k  pitch
    space  gripper      [ / ]  speed
    ENTER  save         ESC  abort

Stay at the keyboard so you are not in the camera view.
""")

    t0 = time.time()
    try:
        with RawKeyboard() as kb:
            while True:
                tick = time.time()
                keys = kb.get()

                dpos = np.zeros(3)
                drot = np.zeros(3)
                for k in keys:
                    if k in ("\r", "\n"):
                        raise KeyboardInterrupt
                    if k == "\x1b":
                        aborted = True
                        raise KeyboardInterrupt
                    if k in TRANSLATE:
                        dpos += TRANSLATE[k]
                    if k in ROTATE:
                        drot += ROTATE[k]
                    if k == " ":
                        grip = 1.0 - grip
                        try:
                            if grip:
                                g.grasp(0.0, 0.1, 40, 0.08, 0.08)
                            else:
                                g.move(0.08, 0.1)
                        except Exception as exc:
                            print(f"\n  gripper: {exc}")
                    if k == "[":
                        speed_scale = max(0.2, speed_scale - 0.2)
                    if k == "]":
                        speed_scale = min(3.0, speed_scale + 0.2)

                # Advance the target. Non-blocking: the controller tracks it.
                if dpos.any():
                    tgt_pos = tgt_pos + dpos * lin_per_tick * speed_scale
                    tgt_pos[0] = np.clip(tgt_pos[0], *SAFE_X)
                    tgt_pos[1] = np.clip(tgt_pos[1], *SAFE_Y)
                    tgt_pos[2] = np.clip(tgt_pos[2], *SAFE_Z)
                if drot.any():
                    dR = R.from_rotvec(drot * ang_per_tick * speed_scale).as_matrix()
                    tgt_quat = R.from_matrix(
                        dR @ R.from_quat(tgt_quat).as_matrix()).as_quat()

                ctrl.set_control(tgt_pos, tgt_quat)

                # --- log actual state, not the target ----------------------
                if not args.no_log:
                    ok, bf = brio.read()
                    color = pipe.wait_for_frames().get_color_frame()
                    if ok and color:
                        wrist = np.asanyarray(color.get_data())[:, :, ::-1]
                        rec.append({
                            "t": tick - t0,
                            "pos": p.get_position().tolist(),
                            "aa": R.from_quat(p.get_orientation()).as_rotvec().tolist(),
                            "q": p.get_state().q,
                            "grip": grip,
                        })
                        buf_p.append(np.array(
                            Image.fromarray(bf[:, :, ::-1]).resize(IMG_SIZE)))
                        buf_w.append(np.array(
                            Image.fromarray(wrist).resize(IMG_SIZE)))

                act = p.get_position()
                lag = np.linalg.norm(tgt_pos - act) * 1000
                print(f"\r  {len(rec):4d}f {tick-t0:5.1f}s  "
                      f"xyz {act[0]:+.3f} {act[1]:+.3f} {act[2]:+.3f}  "
                      f"lag {lag:4.1f}mm  {'closed' if grip else 'open  '}  "
                      f"x{speed_scale:.1f}   ", end="")

                elapsed = time.time() - tick
                if elapsed < period:
                    time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print()
    except Exception as exc:
        print(f"\nerror: {exc}")
        aborted = True
    finally:
        try:
            p.stop_controller()
        except Exception:
            pass
        if pipe:
            pipe.stop()
        if brio:
            brio.release()

    if args.no_log:
        print("practice mode, nothing saved")
        return 0
    if aborted or not rec:
        os.system(f"rm -rf {out}")
        print("aborted, nothing saved")
        return 0

    n = min(len(rec), len(buf_p), len(buf_w))
    rec, buf_p, buf_w = rec[:n], buf_p[:n], buf_w[:n]
    print(f"writing {n} frames ...")
    for i, (a, b) in enumerate(zip(buf_p, buf_w)):
        Image.fromarray(a).save(os.path.join(out, "primary", f"{i:05d}.jpg"), quality=90)
        Image.fromarray(b).save(os.path.join(out, "wrist", f"{i:05d}.jpg"), quality=90)
    json.dump({"task": args.task, "frames": rec},
              open(os.path.join(out, "traj.json"), "w"))

    pos = np.array([f["pos"] for f in rec])
    t = np.array([f["t"] for f in rec])
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    grips = int(np.abs(np.diff([f["grip"] for f in rec])).sum())
    print(f"\nep{args.episode:04d}: {n} frames, {t[-1]:.1f}s, {n/t[-1]:.1f} Hz")
    print(f"  median step: {np.median(d)*1000:.2f} mm   (want 3-6)")
    print(f"  travel (m):  {np.round(pos.max(0)-pos.min(0), 3)}")
    print(f"  gripper toggles: {grips}   (want 2)")
    if np.median(d) * 1000 < 2.5:
        print("  -> too slow; hold keys longer or press ']' to speed up")
    return 0


if __name__ == "__main__":
    sys.exit(main())
