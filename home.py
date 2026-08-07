#!/usr/bin/env python3
"""
Move the Panda to a (optionally randomized) home pose and open the gripper.

Usage:
    python home.py              # randomized home pose
    python home.py --fixed      # exact canonical ready pose, no jitter
    python home.py --jitter 0.0 # same as --fixed
    python home.py --speed 0.1  # a bit faster (default 0.05 = 5%)

Intended as the "reset" step between demonstration recordings:
each episode starts from a slightly different arm configuration so
the policy sees varied initial conditions rather than one memorized pose.
"""

import argparse
import sys
import time

import numpy as np
import panda_py
from panda_py import libfranka

# Canonical Franka "ready" joint configuration (radians).
# Gripper points straight down, arm in a well-conditioned mid-workspace pose.
Q_READY = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])

# Per-joint jitter (radians) applied around Q_READY when randomizing.
# Wrist joints get more freedom than the shoulder/elbow, which move the
# end effector a lot for small angle changes.
JITTER = np.array([0.25, 0.12, 0.25, 0.12, 0.30, 0.15, 0.40])

# Conservative joint limits (radians), inset from the hardware limits so
# a randomized target can never land on a limit violation.
Q_MIN = np.array([-2.75, -1.70, -2.75, -3.00, -2.75, -0.10, -2.80])
Q_MAX = np.array([ 2.75,  1.70,  2.75, -0.10,  2.75,  3.65,  2.80])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="192.168.8.201", help="robot IP")
    ap.add_argument("--speed", type=float, default=0.05,
                    help="relative dynamics factor, 0-1 (default 0.05 = 5%%)")
    ap.add_argument("--jitter", type=float, default=1.0,
                    help="scale on the randomization, 0 disables it (default 1.0)")
    ap.add_argument("--fixed", action="store_true", help="no randomization")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for repeatability")
    ap.add_argument("--gripper-width", type=float, default=0.08,
                    help="width to open the gripper to, metres (default 0.08)")
    ap.add_argument("--no-gripper", action="store_true",
                    help="skip gripper opening (e.g. if no gripper attached)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # --- connect ------------------------------------------------------------
    try:
        p = panda_py.Panda(args.ip)
    except Exception as exc:
        print(f"Could not connect to robot at {args.ip}: {exc}")
        return 1

    state = p.get_state()
    print(f"connected  |  mode: {state.robot_mode}")
    if state.last_motion_errors:
        print(f"clearing previous errors: {state.last_motion_errors}")

    # Clear any latched reflex/error so the move can start.
    try:
        p.get_robot().automatic_error_recovery()
    except Exception as exc:
        print(f"error recovery failed (continuing anyway): {exc}")

    # Collision thresholds high enough not to trip on the arm's own dynamics,
    # low enough to still stop on a real collision.
    p.get_robot().set_collision_behavior(
        [50.0] * 7, [50.0] * 7,   # lower/upper torque thresholds (Nm)
        [30.0] * 6, [30.0] * 6,   # lower/upper force thresholds (N)
    )

    # --- open the gripper first --------------------------------------------
    # Done before moving so nothing is still clamped while the arm travels.
    if not args.no_gripper:
        try:
            g = libfranka.Gripper(args.ip)
            g.move(args.gripper_width, 0.1)
            time.sleep(0.5)
            print(f"gripper opened to {g.read_once().width:.3f} m")
        except Exception as exc:
            print(f"gripper command failed (continuing): {exc}")

    # --- pick a target configuration ---------------------------------------
    if args.fixed or args.jitter == 0.0:
        q_target = Q_READY.copy()
        print("target: canonical ready pose (no jitter)")
    else:
        offset = rng.uniform(-1.0, 1.0, size=7) * JITTER * args.jitter
        q_target = np.clip(Q_READY + offset, Q_MIN, Q_MAX)
        print(f"target: randomized  |  offset (rad): {np.round(offset, 3)}")

    # --- move ---------------------------------------------------------------
    print(f"moving at {args.speed:.0%} speed ...")
    try:
        p.move_to_joint_position(q_target, speed_factor=args.speed)
    except Exception as exc:
        print(f"\nmove failed: {exc}")
        print("common causes: user-stop engaged, FCI not activated, or the arm")
        print("started too close to a joint limit. Check Desk, then retry.")
        return 1

    # --- report -------------------------------------------------------------
    pos = p.get_position()
    ori = p.get_orientation()
    print("\ndone.")
    print(f"  joints (rad): {np.round(p.get_state().q, 3)}")
    print(f"  eef position: {np.round(pos, 4)}")
    print(f"  eef quat:     {np.round(ori, 4)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
