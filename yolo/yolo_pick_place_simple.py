#!/usr/bin/env python3
"""
Rule-based pick-and-place on a Franka Emika arm using YOLO detection.
EYE-IN-HAND version: RealSense mounted on the gripper.

Pipeline:
  Move to scan pose -> RealSense (color+depth, aligned) -> YOLO bbox ->
  depth ROI -> 3D point in camera frame -> T_base_ee(now) @ T_EE_CAM ->
  grasp pose in base frame -> scripted motion (pre-grasp, descend, grasp,
  lift, place).

Robot control: `panda-py` (Python bindings around libfranka).
    pip install panda-python pyrealsense2 ultralytics opencv-contrib-python \
                numpy scipy

SETUP REQUIRED before hardware runs (script enforces #1):
  1. T_EE_CAM: hand-eye calibration result, CAMERA frame -> GRIPPER (EE)
     frame. Eye-in-hand procedure: ChArUco board FIXED ON THE TABLE, move
     the arm to 15-20 varied poses where the wrist camera sees the board,
     record (T_base_ee, solvePnP board pose) pairs, cv2.calibrateHandEye
     -> R_cam2gripper, t_cam2gripper. Paste the 4x4 below.
  2. Fine-tuned YOLO weights containing your box class (COCO has none).
  3. FCI real-time requirements (RT kernel, clean NIC) as always.

Assumes a top-down grasp of a small box on a horizontal table, perceived
from a fixed overhead scan pose.
"""

import sys
import time

import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

import panda_py
from panda_py import libfranka

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
ROBOT_IP = "192.168.8.201"

WEIGHTS = "/home/intern/Desktop/pickplace/runs/detect/train-3/weights/best.pt"
TARGET_CLASS = "box"                           # class name in YOUR weights
CONF_THRESH = 0.30

# Hand-eye calibration result: 4x4 homogeneous transform, CAMERA frame ->
# GRIPPER/EE frame (eye-in-hand). REPLACE with your calibration output.
T_EE_CAM = np.array([
    [+0.016129, -0.999870, +0.000866, +0.050283],
    [+0.999860, +0.016133, +0.004393, -0.032898],
    [-0.004407, +0.000795, +0.999990, -0.039142],
    [+0.000000, +0.000000, +0.000000, +1.000000],
])

# Scan pose: joint angles (rad) giving the wrist camera a clear overhead
# view of the workspace. Jog the arm there by hand, then read
# `panda.get_state().q` (or panda.q) and paste. None -> uses move_to_start().
SCAN_JOINTS = [-0.8293, -0.4729, 0.6980, -2.1923, 0.2979, 1.8049, -2.5924]

# Workspace sanity bounds in base frame (metres).
WS_X = (0.22772959, 0.56885654)
WS_Y = (-0.3409705, 0.25)
WS_Z = (0.00, 0.30)

TABLE_Z = 0.010          # table surface height in base frame (m), measure it
BOX_HEIGHT = 0.06        # NOMINAL box height (m) — used only for the depth
                         # geometry-fallback; actual height is MEASURED from
                         # depth every detection (box is 10x4x12, orientation
                         # varies)
BOX_GRASP_WIDTH = 0.040  # graspable dimension (m): the 4 cm side

GRASP_BELOW_TOP = 0.020  # grip this far below the MEASURED box top (m)
MAX_GRASP_SPAN = 0.075   # widest footprint the gripper can straddle (m)
PRE_GRASP_CLEAR = 0.12   # hover/retreat height above grasp point (m)
LIFT_HEIGHT = 0.15       # lift after grasp (m)
PLACE_XY = (0.34952504, -0.02306019)  # drop location (base frame, m)
PLACE_AT_PICK_LOCATION = True  # True: release the box back where it was
                               # picked up (ignores PLACE_XY); False: carry
                               # to PLACE_XY

GRIPPER_SPEED = 0.05     # m/s
GRIPPER_FORCE = 55.0     # N (cardboard tolerates it; slip at speed does not)
MOVE_SPEED_FACTOR = 0.22 # one speed for everything
TRANSIT_SPEED = 0.22
CARRY_SPEED = 0.22
FINE_SPEED = 0.25        # precision verticals near the box

MAX_ATTEMPTS = 3
DEPTH_WIN = 7            # median window (px) for depth sampling
PERCEPTION_SAMPLES = 3   # frames per perception cycle; median is used
GRASP_XY_CORRECTION = (0.000, 0.000)  # constant bias correction (m), from
                                      # the measured-pick audit; leave zero
                                      # until measured


# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
class Camera:
    def __init__(self):
        self.pipe = rs.pipeline()
        cfg = rs.config()
        # Color at 1280x720 to MATCH the YOLO training photos (snap.py used
        # the camera's default 16:9 mode). Depth at 848x480; align() maps it
        # into the color frame's geometry at color resolution.
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        profile = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.depth_scale = (
            profile.get_device().first_depth_sensor().get_depth_scale()
        )
        for _ in range(15):
            self.pipe.wait_for_frames()

    def read(self):
        frames = self.align.process(self.pipe.wait_for_frames())
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data())
        return color, depth

    def deproject(self, u, v, z_m):
        return np.array(
            rs.rs2_deproject_pixel_to_point(self.intr, [float(u), float(v)], z_m)
        )

    def stop(self):
        self.pipe.stop()


# ----------------------------------------------------------------------------
# Perception
# ----------------------------------------------------------------------------
def detect_box(model, color):
    # Training photos were saved with R/B channels swapped (snap.py captured
    # RGB but imwrite assumed BGR). Swap live frames the same way so the
    # model sees the colors it was trained on.
    swapped = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    results = model(swapped, conf=CONF_THRESH, verbose=False)[0]
    best = None
    for det in results.boxes:
        name = model.names[int(det.cls)]
        if name != TARGET_CLASS:
            continue
        conf = float(det.conf)
        if best is None or conf > best[1]:
            best = (det.xyxy[0].cpu().numpy(), conf)
    return best


def median_depth(depth, u, v, win=DEPTH_WIN):
    h, w = depth.shape
    u, v = int(u), int(v)
    u0, u1 = max(0, u - win), min(w, u + win + 1)
    v0, v1 = max(0, v - win), min(h, v + win + 1)
    patch = depth[v0:v1, u0:u1].astype(np.float64)
    valid = patch[patch > 0]
    if valid.size < 5:
        return None
    return float(np.median(valid))


def top_face_rect(depth, bbox, depth_scale, z_med_m):
    """minAreaRect of the box's top surface within the bbox, or None."""
    x1, y1, x2, y2 = [int(round(c)) for c in bbox]
    roi = depth[y1:y2, x1:x2].astype(np.float64)
    if roi.size == 0:
        return None
    z_med_raw = z_med_m / depth_scale
    mask = ((roi > 0) &
            (np.abs(roi - z_med_raw) < 0.02 / depth_scale)).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    # shift rect centre back to full-image coordinates
    (cx, cy), (rw, rh), ang = rect
    return ((cx + x1, cy + y1), (rw, rh), ang)


def compute_grasp_in_base(cam, model, T_base_cam):
    """
    Perceive from the CURRENT camera pose. T_base_cam must be the transform
    valid at the moment the frame is captured (eye-in-hand: recompute every
    time from the robot pose).
    Returns (grasp_xyz_base, yaw_base) or None.
    """
    color, depth = cam.read()
    det = detect_box(model, color)

    vis = color.copy()
    if det is not None:
        b = det[0].astype(int)
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cv2.putText(vis, f"box {det[1]:.2f}", (b[0], b[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite("last_scan.jpg", vis)

    if det is None:
        return None
    bbox, conf = det
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0

    # Robust depth: median over the central region of the bbox, zeros ignored.
    x1, y1, x2, y2 = [int(round(c)) for c in bbox]
    mx, my = int(0.25 * (x2 - x1)), int(0.25 * (y2 - y1))
    roi = depth[y1 + my:y2 - my, x1 + mx:x2 - mx].astype(np.float64)
    valid = roi[roi > 0]
    if valid.size >= 20:
        z_m = float(np.median(valid)) * cam.depth_scale
        depth_src = "sensor"
    else:
        # Geometric fallback: box-top height is known; expected depth is the
        # calibrated camera height above that plane (camera ~level).
        cam_z = T_base_cam[2, 3]
        z_m = cam_z - (TABLE_Z + BOX_HEIGHT)
        depth_src = "geometry-fallback"
        if z_m <= 0.05:
            print("[perception] depth fallback implausible; skipping frame")
            return None

    def px_to_base(u, v, z):
        p_cam = cam.deproject(u, v, z)
        return (T_base_cam @ np.append(p_cam, 1.0))[:3]

    # Grasp point: prefer the centre of the depth-fitted TOP-FACE rectangle.
    # The bbox centre is perspective-biased on tall boxes (side face visible
    # off image-centre), which laterally offsets the grasp by up to ~2 cm.
    rect = top_face_rect(depth, bbox, cam.depth_scale, z_m)
    if rect is not None:
        gu, gv = rect[0]
    else:
        gu, gv = cx, cy
    p_base = px_to_base(gu, gv, z_m)

    # Measure the box top from depth (camera is level at the scan pose), so
    # the grasp height adapts to whichever face the box is resting on.
    cam_z = T_base_cam[2, 3]
    box_top_z = cam_z - z_m
    box_h = box_top_z - TABLE_Z
    if box_h < 0.02:
        print(f"[perception] measured box height {box_h*100:.1f} cm "
              f"implausible; skipping frame")
        return None
    grasp_z = max(TABLE_Z + 0.015, box_top_z - GRASP_BELOW_TOP)
    grasp = np.array([p_base[0], p_base[1], grasp_z])

    # Yaw: deproject two points along the top-face long axis and measure the
    # direction in the BASE frame. Valid for any camera orientation.
    yaw = 0.0
    if rect is not None:
        (rcx, rcy), (rw, rh), ang = rect
        ang_rad = np.deg2rad(ang if rw >= rh else ang + 90.0)  # long axis
        du, dv = np.cos(ang_rad), np.sin(ang_rad)
        L = 20  # px offset along the long axis
        pA = px_to_base(rcx - L * du, rcy - L * dv, z_m)
        pB = px_to_base(rcx + L * du, rcy + L * dv, z_m)
        long_dir = pB[:2] - pA[:2]
        if np.linalg.norm(long_dir) > 1e-6:
            long_yaw = np.arctan2(long_dir[1], long_dir[0])
            # Empirically on this hand: aligning the gripper to the LONG
            # axis puts the finger travel across the SHORT axis.
            yaw = long_yaw

    # Graspability: the footprint's SHORT side must fit in the gripper.
    if rect is not None:
        short_px = min(rect[1])
        short_m = short_px * z_m / cam.intr.fx
        if short_m > MAX_GRASP_SPAN:
            print(f"[perception] box footprint {short_m*100:.1f} cm across "
                  f"its narrow side — too wide for the gripper in this "
                  f"orientation. Stand the box on a smaller face.")
            return None
    else:
        short_m = float('nan')

    print(f"[perception] conf={conf:.2f} grasp={np.round(grasp,3)} "
          f"yaw={np.rad2deg(yaw):.1f} deg  depth={z_m:.3f} m ({depth_src})  "
          f"box_h={box_h*100:.1f} cm  width={short_m*100:.1f} cm")
    return grasp, yaw


def in_workspace(p):
    return (WS_X[0] <= p[0] <= WS_X[1] and
            WS_Y[0] <= p[1] <= WS_Y[1] and
            WS_Z[0] <= p[2] <= WS_Z[1])


# ----------------------------------------------------------------------------
# Motion
# ----------------------------------------------------------------------------
def top_down_quat(yaw):
    return R.from_euler("xyz", [np.pi, 0.0, yaw]).as_quat()


def _wait_converged(panda, pos, tol_m, timeout_s):
    """Poll until within tol OR progress stalls (no point waiting out an
    asymptotic impedance tail). Returns (err, converged)."""
    t0 = time.time()
    best = 1e9
    last_improve = t0
    err = 1e9
    while time.time() - t0 < timeout_s:
        err = np.linalg.norm(np.asarray(panda.get_position()) - pos)
        if err < tol_m:
            return err, True
        if err < best - 0.001:
            best = err
            last_improve = time.time()
        elif time.time() - last_improve > 1.2:
            return err, False        # stalled: stop waiting, act
        time.sleep(0.03)
    return err, False


def move(panda, pos, quat, tol_m=0.007, timeout_s=12.0, straight=False,
         speed=None):
    """Move to pose. Joint-space (IK) by default for big reorientations;
     (straight-line mode available but unused). Stall-aware convergence
    instead of fixed waits; joint-space finisher closes small residuals."""
    if speed is None:
        speed = MOVE_SPEED_FACTOR
    pos = np.asarray(pos, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R.from_quat(np.asarray(quat, dtype=np.float64)).as_matrix()
    T[:3, 3] = pos

    if straight:
        err = 1e9
        for _try in range(2):
            panda.move_to_pose(T, speed_factor=speed)
            err, ok = _wait_converged(panda, pos, tol_m, timeout_s / 2)
            if ok:
                return
        if err < 0.06:
            # typical impedance stall: finish quietly in joint space (arc
            # over <=6 cm is sub-mm; no recovery needed)
            return move(panda, pos, quat, tol_m=tol_m,
                        timeout_s=timeout_s, straight=False, speed=speed)
        # Cartesian path hit a joint limit or obstruction well short of the
        # target: recover and complete via joint space (arcs, but arrives).
        print(f"[move] straight path failed at {err*1000:.0f} mm; "
              f"falling back to joint-space route")
        try:
            panda.recover()
        except Exception:
            pass
        return move(panda, pos, quat, tol_m=tol_m,
                    timeout_s=timeout_s, straight=False, speed=speed)

    # LEAST-MOTION selection: among all valid IK solutions (sweeping the
    # redundant joint 7), choose the one CLOSEST to the current joint
    # configuration — minimises total joint travel, so the arm never twists
    # or flips to a far branch when a nearby one reaches the same pose.
    q_cur = np.asarray(panda.q, dtype=np.float64)
    best_q = None
    best_cost = np.inf
    # joint weights: penalise moving the big proximal joints (1-4) more than
    # the wrist, so "least motion" favours small, safe wrist-level changes.
    w = np.array([2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.5])
    cur_q7 = float(q_cur[6])
    for q7 in [cur_q7] + list(np.linspace(-2.7, 2.7, 41)):
        try:
            q = panda_py.ik(T, q_init=q_cur, q_7=q7)
        except Exception:
            continue
        q = np.asarray(q, dtype=np.float64)
        if np.any(np.isnan(q)):
            continue
        cost = float(np.sum(w * (q - q_cur) ** 2))
        if cost < best_cost:
            best_cost, best_q = cost, q
    if best_q is not None:
        panda.move_to_joint_position(best_q, speed_factor=speed)
    else:
        panda.move_to_pose(T, speed_factor=speed)

    err, ok = _wait_converged(panda, pos, tol_m, timeout_s)
    if ok:
        return
    raise RuntimeError(f"move did not converge: {err*1000:.1f} mm from "
                       f"target {np.round(pos, 3)}")





def go_to_scan_pose(panda):
    if SCAN_JOINTS is None:
        panda.move_to_start(speed_factor=TRANSIT_SPEED)
    else:
        panda.move_to_joint_position(np.asarray(SCAN_JOINTS, dtype=np.float64),
                                     speed_factor=TRANSIT_SPEED)
    time.sleep(0.15)  # brief settle before imaging


def recover(panda):
    try:
        panda.recover()
    except Exception as e:
        print(f"[recover] automatic error recovery failed: {e}")


# ----------------------------------------------------------------------------
# Main sequence
# ----------------------------------------------------------------------------
def pick_and_place(panda, gripper, cam, model):
    """Simple, smooth cycle: scan -> perceive -> hover -> straight down ->
    grip -> ONE straight lift -> lower -> release -> retreat -> scan."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
      try:
        print(f"\n=== attempt {attempt}/{MAX_ATTEMPTS} ===")

        # -- 0. scan pose, open gripper
        go_to_scan_pose(panda)
        gripper.move(0.08, GRIPPER_SPEED)

        # -- 1. perceive: median over a few frames
        T_base_cam = panda.get_pose() @ T_EE_CAM
        samples = []
        for _ in range(PERCEPTION_SAMPLES):
            s = compute_grasp_in_base(cam, model, T_base_cam)
            if s is not None:
                samples.append(s)
        if len(samples) < max(2, PERCEPTION_SAMPLES // 2):
            print("[main] no detection; retrying")
            time.sleep(0.5)
            continue
        gs = np.array([s[0] for s in samples])
        yaws = np.array([s[1] for s in samples])
        grasp = np.median(gs, axis=0)
        yaw = np.arctan2(np.median(np.sin(yaws)), np.median(np.cos(yaws)))
        yaw = ((yaw + np.pi / 2) % np.pi) - np.pi / 2  # two-finger symmetry
        # never let the descend target dip below a safe floor
        grasp[2] = max(grasp[2], TABLE_Z + 0.020)
        grasp = grasp + np.array([GRASP_XY_CORRECTION[0],
                                  GRASP_XY_CORRECTION[1], 0.0])
        print(f"[main] grasp={np.round(grasp, 3)} "
              f"yaw={np.rad2deg(yaw):.1f} deg")
        if not in_workspace(grasp):
            print(f"[main] REJECT: grasp outside workspace bounds")
            continue
        quat = top_down_quat(yaw)
        pre = grasp + np.array([0, 0, PRE_GRASP_CLEAR])

        # -- 2. hover above the box
        move(panda, pre, quat, tol_m=0.030)

        # -- 3. descend: joint-space, seeded from the CURRENT configuration
        # (q7-continuity keeps the same IK branch as the hover, so the only
        # joint motion is the small delta needed to go down — no elbow
        # re-adjustment, no Cartesian null-space drift)
        move(panda, grasp, quat, tol_m=0.003, timeout_s=18.0)

        # -- 4. ONE grip
        ok = gripper.grasp(width=BOX_GRASP_WIDTH * 0.8, speed=GRIPPER_SPEED,
                           force=GRIPPER_FORCE, epsilon_inner=0.02,
                           epsilon_outer=0.02)
        state = gripper.read_once()
        print(f"[grasp] ok={ok} width={state.width * 1000:.1f} mm")
        if not ok or state.width < 0.005:
            print("[grasp] closed on air; retreating and retrying")
            gripper.move(0.08, GRIPPER_SPEED)
            move(panda, pre, quat)
            continue

        # -- 5. ONE straight lift
        move(panda, grasp + np.array([0, 0, LIFT_HEIGHT]), quat,
             speed=CARRY_SPEED, tol_m=0.030)

        # -- 6. place: lower and release
        if PLACE_AT_PICK_LOCATION:
            place = grasp.copy()
        else:
            place = np.array([PLACE_XY[0], PLACE_XY[1], grasp[2]])
            move(panda, place + np.array([0, 0, LIFT_HEIGHT]), quat,
                 speed=CARRY_SPEED)
        move(panda, place + np.array([0, 0, 0.008]), quat, tol_m=0.004,
             timeout_s=15.0, speed=CARRY_SPEED)
        gripper.move(0.08, GRIPPER_SPEED)

        # -- 7. retreat, home
        move(panda, place + np.array([0, 0, PRE_GRASP_CLEAR]), quat, tol_m=0.030)
        go_to_scan_pose(panda)
        print("[main] SUCCESS")
        return True
      except Exception as e:
        print(f"[attempt {attempt}] aborted: {e}")
        print("[attempt] recovering and retrying...")
        recover(panda)
        time.sleep(1.0)
        try:
            gripper.move(0.08, GRIPPER_SPEED)
        except Exception:
            pass
        continue

    print("[main] all attempts exhausted")
    try:
        go_to_scan_pose(panda)
    except Exception:
        pass
    return False


def main():
    if np.allclose(T_EE_CAM, np.eye(4)):
        sys.exit("T_EE_CAM is still the identity placeholder. Run EYE-IN-HAND "
                 "hand-eye calibration (board fixed on table, "
                 "cv2.calibrateHandEye -> cam2gripper) and paste the 4x4 "
                 "before running on hardware.")

    print("[init] loading YOLO weights:", WEIGHTS)
    model = YOLO(WEIGHTS)
    if TARGET_CLASS not in model.names.values():
        sys.exit(f"Class '{TARGET_CLASS}' not in model classes "
                 f"{list(model.names.values())}. Fine-tune YOLO on your box "
                 f"or fix TARGET_CLASS.")

    print("[init] starting camera")
    cam = Camera()

    print("[init] connecting to robot", ROBOT_IP)
    panda = panda_py.Panda(ROBOT_IP,
                           realtime_config=libfranka.RealtimeConfig.kIgnore)
    gripper = libfranka.Gripper(ROBOT_IP)
    gripper.homing()

    try:
        pick_and_place(panda, gripper, cam, model)
    except Exception as e:
        print(f"[main] EXCEPTION: {e}")
        recover(panda)
        try:
            go_to_scan_pose(panda)
        except Exception:
            print("[main] could not return to scan pose; check Desk / unlock")
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
