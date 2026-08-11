#!/usr/bin/env python3
"""
Move the Franka so the wrist camera looks STRAIGHT DOWN (image plane parallel
to the floor) at a set height, keep the current x,y, and save this as the
starting/scan pose.

- Height default: 0.42 m above the table/ground plane (see --table-z if the
  robot base is not mounted at that plane's level).
- Orientation: if T_EE_CAM.npy (from calibrate_handeye.py) is in the current
  directory, the CAMERA optical axis is levelled exactly using the
  calibration. Otherwise it falls back to pointing the GRIPPER straight down,
  which is the same thing if the camera is mounted parallel to the flange.
- Saves joint angles to starting_pos.npy and prints a SCAN_JOINTS line to
  paste into yolo_pick_place.py.

Usage:
    python go_start_pose.py [--robot-ip 192.168.8.201] [--height 0.42]
                            [--table-z 0.0] [--speed 0.15]
"""

import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as R
import panda_py
from panda_py import libfranka


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", default="192.168.8.201")
    ap.add_argument("--height", type=float, default=0.42,
                    help="camera/EE height above the table plane (m)")
    ap.add_argument("--table-z", type=float, default=0.0,
                    help="table plane height in base frame (m)")
    ap.add_argument("--speed", type=float, default=0.15)
    ap.add_argument("--x", type=float, default=0.45,
                    help="target x in base frame (m)")
    ap.add_argument("--y", type=float, default=0.0,
                    help="target y in base frame (m)")
    ap.add_argument("--yaw", type=float, default=90.0,
                    help="camera yaw about vertical, degrees")
    args = ap.parse_args()

    print("[init] connecting to robot", args.robot_ip)
    panda = panda_py.Panda(args.robot_ip,
                           realtime_config=libfranka.RealtimeConfig.kIgnore)

    T = np.asarray(panda.get_pose(), dtype=np.float64)   # T_base_ee, 4x4
    cur_pos = T[:3, 3]
    cur_rot = T[:3, :3]

    yaw = np.deg2rad(args.yaw)

    # Desired camera orientation: optical (z) axis pointing straight down,
    # i.e. R_base_cam = Rz(yaw) * Rx(pi).
    R_base_cam_target = (R.from_euler("z", yaw) *
                         R.from_euler("x", np.pi)).as_matrix()

    if os.path.exists("T_EE_CAM.npy"):
        T_ee_cam = np.load("T_EE_CAM.npy")
        R_ee_cam = T_ee_cam[:3, :3]
        R_target = R_base_cam_target @ R_ee_cam.T        # level the CAMERA
        print("[pose] using T_EE_CAM.npy -> camera axis levelled exactly")
    else:
        R_target = R_base_cam_target                     # level the GRIPPER
        print("[pose] no T_EE_CAM.npy found -> levelling the gripper axis "
              "(fine if the camera is mounted parallel to the flange)")

    tx, ty = args.x, args.y
    target_pos = np.array([tx, ty, args.table_z + args.height])
    T_target = np.eye(4)
    T_target[:3, :3] = R_target
    T_target[:3, 3] = target_pos

    print(f"[pose] current pos {np.round(cur_pos,3)} -> "
          f"target {np.round(target_pos,3)} (height {args.height} m, "
          f"yaw {args.yaw} deg)")
    input("Press ENTER to move (Ctrl+C to abort)...")

    # (No via-home detour: the analytical IK + fixed q7 sweep below pins a
    # deterministic joint solution regardless of the starting configuration.)

    # Large reorientations fail as a single Cartesian waypoint; solve IK and
    # move in joint space. The analytical Panda IK needs joint 7 pinned, and
    # the default value may have no solution for this orientation — sweep
    # candidates until one solves.
    # Evaluate ALL q7 candidates seeded from a NEUTRAL posture and keep the
    # solution closest to neutral overall — deterministic and natural-looking
    # (first-valid picks arbitrary/contorted IK branches).
    Q_NEUTRAL = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    best = None
    for q7 in np.linspace(-2.6, 2.6, 53):
        try:
            q = panda_py.ik(T_target, q_init=Q_NEUTRAL, q_7=q7)
        except Exception:
            continue
        q = np.asarray(q, dtype=np.float64)
        if np.any(np.isnan(q)):
            continue
        score = np.linalg.norm(q - Q_NEUTRAL)
        if best is None or score < best[0]:
            best = (score, q, q7)
    if best is not None:
        _, q_target, q7 = best
        print(f"[pose] IK: best-of-sweep q7={q7:+.2f} "
              f"(distance from neutral {best[0]:.2f} rad); joint move...")
        panda.move_to_joint_position(q_target, speed_factor=args.speed)
    else:
        print("[pose] IK found no solution at any q7; trying Cartesian move")
        panda.move_to_pose(T_target, speed_factor=args.speed)
    # panda-py's return value is unreliable on this setup; verify by
    # measuring convergence instead.
    import time
    t0 = time.time()
    err = 1e9
    while time.time() - t0 < 12.0:
        err = np.linalg.norm(np.asarray(panda.get_position()) - target_pos)
        if err < 0.007:
            break
        time.sleep(0.05)
    print(f"[pose] settled {err*1000:.1f} mm from target")
    if err > 0.02:
        raise SystemExit("did not converge (>20 mm off). Target may be out "
                         "of reach; jog nearer the workspace centre and rerun")

    q = np.asarray(panda.q, dtype=np.float64).copy()
    np.save("starting_pos.npy", q)
    print("\n[done] joints saved to starting_pos.npy")
    print("Paste into yolo_pick_place.py:\n")
    print("SCAN_JOINTS = [" + ", ".join(f"{v:.4f}" for v in q) + "]")


if __name__ == "__main__":
    main()
