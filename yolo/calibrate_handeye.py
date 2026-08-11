#!/usr/bin/env python3
"""
Eye-in-hand hand-eye calibration for a wrist-mounted RealSense on a Franka.
Produces T_EE_CAM (camera -> gripper/EE) for yolo_pick_place.py.

Usage:
  python calibrate_handeye.py --make-board          # generate board.png
  python calibrate_handeye.py --square-mm 39 --marker-mm 28
  python calibrate_handeye.py --solve-only          # re-solve from captures.npz

Keys during capture:  SPACE = capture   c = compute   q = quit
Every capture is autosaved to captures.npz, so a crash or quit never loses
the session; --solve-only recomputes from that file with no robot needed.

Requires opencv-contrib-python >= 4.7 (CharucoDetector).
"""

import argparse
import os
import sys

import numpy as np
import cv2

SQUARES_X, SQUARES_Y = 7, 5
SQUARE_MM_DEFAULT = 30.0
MARKER_MM_DEFAULT = 22.0
ARUCO_DICT = cv2.aruco.DICT_5X5_100
CAPTURE_FILE = "captures.npz"


# --------------------------------------------------------------------------
def get_handeye_solver():
    """Locate calibrateHandEye across OpenCV 4.x / 5.x module layouts."""
    fn = getattr(cv2, "calibrateHandEye", None)
    if fn is None and hasattr(cv2, "calib"):
        fn = getattr(cv2.calib, "calibrateHandEye", None)
    if fn is None and hasattr(cv2, "registration"):
        fn = getattr(cv2.registration, "calibrateHandEye", None)
    if fn is None:
        sys.exit(
            "Your OpenCV build has no calibrateHandEye (moved/removed in "
            "OpenCV 5). Fix by installing the stable 4.x contrib build:\n\n"
            "  pip uninstall -y opencv-python opencv-contrib-python "
            "opencv-python-headless\n"
            "  pip install 'opencv-contrib-python==4.11.0.86'\n\n"
            "Your captures are safe in captures.npz — after reinstalling, "
            "run:  python calibrate_handeye.py --solve-only")
    method = getattr(cv2, "CALIB_HAND_EYE_TSAI",
                     getattr(getattr(cv2, "calib", cv2),
                             "CALIB_HAND_EYE_TSAI", 0))
    return fn, method


def solve_and_report(R_g2b, t_g2b, R_t2c, t_t2c):
    fn, _ = get_handeye_solver()
    methods = []
    for name in ("TSAI", "PARK", "HORAUD", "DANIILIDIS"):
        m = getattr(cv2, f"CALIB_HAND_EYE_{name}",
                    getattr(getattr(cv2, "calib", cv2),
                            f"CALIB_HAND_EYE_{name}", None))
        if m is not None:
            methods.append((name, m))

    results = []
    print(f"\ncaptures used: {len(R_g2b)}")
    for name, m in methods:
        try:
            R_c2g, t_c2g = fn(R_g2b, t_g2b, R_t2c, t_t2c, method=m)
            off = float(np.linalg.norm(t_c2g)) * 1000
            print(f"  {name:<11} offset {off:12.1f} mm")
            results.append((name, R_c2g, np.asarray(t_c2g).reshape(3), off))
        except cv2.error as e:
            print(f"  {name:<11} failed: {str(e).splitlines()[-1][:60]}")

    # keep only physically plausible solutions (5..500 mm from flange)
    plausible = [r for r in results if 5.0 < r[3] < 500.0]
    if not plausible:
        print("\nNO plausible solution. The capture set lacks rotation "
              "diversity or contains corrupted pairs.\n"
              "Recommended: rm captures.npz and recapture ~15 poses where "
              "EVERY capture differs from the previous one by >15 deg of "
              "camera rotation (tilts AND wrist twists).")
        return

    # cross-method agreement check
    if len(plausible) >= 2:
        ts = np.array([r[2] for r in plausible])
        spread = np.linalg.norm(ts - ts.mean(axis=0), axis=1).max() * 1000
        print(f"  agreement spread across methods: {spread:.1f} mm "
              f"({'GOOD' if spread < 20 else 'POOR — treat with suspicion'})")

    name, R_c2g, t_c2g, off = plausible[0]
    T = np.eye(4)
    T[:3, :3] = R_c2g
    T[:3, 3] = t_c2g
    np.save("T_EE_CAM.npy", T)
    print("\n" + "=" * 70)
    print(f"selected method: {name}  |  camera offset from EE: {off:.1f} mm "
          f"(should match the physical mount, ~30-120 mm)")
    print("Paste this into yolo_pick_place.py:\n")
    print("T_EE_CAM = np.array([")
    for row in T:
        print("    [" + ", ".join(f"{v:+.6f}" for v in row) + "],")
    print("])")
    print("=" * 70)
    print("(also saved to T_EE_CAM.npy)\n")


def save_captures(R_g2b, t_g2b, R_t2c, t_t2c):
    np.savez(CAPTURE_FILE,
             R_g2b=np.asarray(R_g2b), t_g2b=np.asarray(t_g2b),
             R_t2c=np.asarray(R_t2c), t_t2c=np.asarray(t_t2c))


def load_captures():
    if not os.path.exists(CAPTURE_FILE):
        sys.exit(f"{CAPTURE_FILE} not found — run a capture session first.")
    d = np.load(CAPTURE_FILE)
    return (list(d["R_g2b"]), list(d["t_g2b"]),
            list(d["R_t2c"]), list(d["t_t2c"]))


def make_board(square_mm, marker_mm):
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    return cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), square_mm / 1000.0, marker_mm / 1000.0, d)


def save_board_png(board):
    px_per_m = 300 / 0.0254
    w = int(SQUARES_X * (SQUARE_MM_DEFAULT / 1000.0) * px_per_m)
    h = int(SQUARES_Y * (SQUARE_MM_DEFAULT / 1000.0) * px_per_m)
    cv2.imwrite("board.png", board.generateImage((w, h), marginSize=40))
    print("wrote board.png — print at 100% scale, then MEASURE a square.")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-board", action="store_true")
    ap.add_argument("--solve-only", action="store_true",
                    help="recompute from captures.npz (no robot/camera)")
    ap.add_argument("--square-mm", type=float, default=SQUARE_MM_DEFAULT)
    ap.add_argument("--marker-mm", type=float, default=MARKER_MM_DEFAULT)
    ap.add_argument("--robot-ip", default="192.168.8.201")
    ap.add_argument("--min-corners", type=int, default=12)
    args = ap.parse_args()

    board = make_board(args.square_mm, args.marker_mm)
    if args.make_board:
        save_board_png(board)
        return
    if args.solve_only:
        solve_and_report(*load_captures())
        return

    import pyrealsense2 as rs
    import panda_py
    from panda_py import libfranka as _lf

    print("[init] connecting to robot", args.robot_ip)
    panda = panda_py.Panda(args.robot_ip,
                           realtime_config=_lf.RealtimeConfig.kIgnore)

    print("[init] starting camera")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    intr = (profile.get_stream(rs.stream.color)
            .as_video_stream_profile().get_intrinsics())
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.array(intr.coeffs, dtype=np.float64)

    detector = cv2.aruco.CharucoDetector(board)

    # resume a previous session if present
    if os.path.exists(CAPTURE_FILE):
        R_g2b, t_g2b, R_t2c, t_t2c = load_captures()
        print(f"[resume] loaded {len(R_g2b)} previous captures from "
              f"{CAPTURE_FILE} (delete the file to start fresh)")
    else:
        R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []

    print("\nSPACE = capture, c = compute, q = quit. Need >= 8 captures; "
          "15-20 diverse poses recommended.\n")

    while True:
        frames = pipe.wait_for_frames()
        img = np.asanyarray(frames.get_color_frame().get_data())
        vis = img.copy()

        ch_corners, ch_ids, mk_corners, mk_ids = detector.detectBoard(img)
        pose_ok = False
        rvec = tvec = None
        n_corners = 0
        if ch_corners is not None and ch_ids is not None:
            ch_corners = np.asarray(ch_corners,
                                    dtype=np.float32).reshape(-1, 1, 2)
            ch_ids = np.asarray(ch_ids, dtype=np.int32).reshape(-1, 1)
            n_corners = min(len(ch_corners), len(ch_ids))
            ch_corners, ch_ids = ch_corners[:n_corners], ch_ids[:n_corners]
        if n_corners >= args.min_corners:
            try:
                cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)
            except cv2.error:
                for pt in ch_corners.reshape(-1, 2):
                    cv2.circle(vis, tuple(int(v) for v in pt), 4,
                               (0, 255, 255), -1)
            obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
            if obj_pts is not None and len(obj_pts) >= args.min_corners:
                ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist,
                                              flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    pose_ok = True
                    cv2.drawFrameAxes(vis, K, dist, rvec, tvec, 0.05)

        color = (0, 255, 0) if pose_ok else (0, 0, 255)
        cv2.putText(vis, f"captures: {len(R_g2b)}   board: "
                    f"{'OK' if pose_ok else 'NOT VISIBLE'}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("hand-eye calibration (wrist cam)", vis)
        k = cv2.waitKey(1) & 0xFF

        if k == ord(' '):
            if not pose_ok:
                print("  [skip] board not confidently detected")
                continue
            # panda-py's cached state can go stale when no controller is
            # running; reconnect to force a fresh read at capture time.
            del panda
            panda = panda_py.Panda(args.robot_ip,
                                   realtime_config=_lf.RealtimeConfig.kIgnore)
            dq = np.abs(np.asarray(panda.get_state().dq)).max()
            if dq > 0.01:
                print("  [skip] arm still moving — hold still and retry")
                continue
            T = np.asarray(panda.get_pose(), dtype=np.float64)
            if R_g2b:  # rotation diversity feedback vs previous capture
                R_rel = R_g2b[-1].T @ T[:3, :3]
                ang = np.degrees(np.arccos(
                    np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
                if ang < 10.0:
                    print(f"  [warn] only {ang:.1f} deg rotated vs previous "
                          f"capture — rotate more (tilt/twist) for a useful "
                          f"capture. Captured anyway.")
                else:
                    print(f"  [rot] {ang:.1f} deg vs previous — good")
            R_g2b.append(T[:3, :3].copy())
            t_g2b.append(T[:3, 3].copy())
            Rt, _ = cv2.Rodrigues(rvec)
            R_t2c.append(Rt)
            t_t2c.append(tvec.reshape(3).copy())
            save_captures(R_g2b, t_g2b, R_t2c, t_t2c)
            print(f"  [capture {len(R_g2b)}] saved. ee pos="
                  f"{np.round(T[:3,3],3)}  board dist={tvec.ravel()[2]:.3f} m")

        elif k == ord('c'):
            if len(R_g2b) < 8:
                print(f"  need >= 8 captures, have {len(R_g2b)}")
                continue
            solve_and_report(R_g2b, t_g2b, R_t2c, t_t2c)

        elif k == ord('q'):
            break

    pipe.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
