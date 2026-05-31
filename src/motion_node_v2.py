#!/usr/bin/env python3

# RoboPong - Motion Node v2 (UNUSED!!!)


import json
import os
import sys
import threading

import rospy
import yaml
import numpy as np

from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory

import moveit_commander


# ---------------------------------------------------------------------------
# Safety constants (PRD §14). Hard-enforced regardless of solver output /
# parameter values.
# ---------------------------------------------------------------------------
SPEED_SCALE_MIN = 0.05
SPEED_SCALE_MAX = 0.80


# ---------------------------------------------------------------------------
# ThrowSolver — stub (Step 1)
# ---------------------------------------------------------------------------
# Moves to src/throw_solver.py in a later step (PRD §11). For the scaffold
# solve() always returns None, which is the signal for "no solution — caller
# should fall back or refuse the throw".
# ---------------------------------------------------------------------------

class ThrowSolver:
    VALID_MODES = ("lookup", "optimise", "learned")

    def __init__(self, mode, data_path, timeout,
                 release_angle_rad, z_release):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"solver_mode must be one of {self.VALID_MODES}, got {mode!r}")
        self.mode = mode
        self.data_path = data_path
        self.timeout = float(timeout)
        self.release_angle_rad = release_angle_rad
        self.z_release = z_release

    def solve(self, target_x, target_y):
        """Return (speed_scale, aim_angle) or None when no solution exists.

        Step 1 scaffold: always returns None. Callers must treat None as
        'no solution' and honour ~fallback_to_v1 / the spec §14.3 refusal
        behaviour.
        """
        return None


# ---------------------------------------------------------------------------
# LandingLogger — stub (Step 1)
# ---------------------------------------------------------------------------
# Moves to src/landing_logger.py in a later step (PRD §11). Persistence to
# solver_data.yaml is gated on the robopong/RecordLanding.srv definition —
# see Step 2. For now, record() only echoes to rosout when enabled.
# ---------------------------------------------------------------------------

class LandingLogger:
    def __init__(self, data_path, enabled):
        self.data_path = data_path
        self.enabled = bool(enabled)

    def record(self, cup_x, cup_y, speed_scale, x_landed, y_landed):
        if not self.enabled:
            return
        rospy.loginfo(
            f"[landing] cup=({cup_x:.3f},{cup_y:.3f}) "
            f"speed={speed_scale:.3f} landed=({x_landed:.3f},{y_landed:.3f}) "
            f"— persistence deferred to Step 2")


# ---------------------------------------------------------------------------
# MotionNodeV2
# ---------------------------------------------------------------------------

class MotionNodeV2:
    STATE_IDLE = "IDLE"
    STATE_BUSY = "BUSY"

    def __init__(self):
        rospy.init_node("motion_node_v2", anonymous=False)
        rospy.loginfo("RoboPong Motion Node v2 starting (SCAFFOLD).")

        # --- v1-compatible parameters -------------------------------------
        self.group_name        = rospy.get_param("~move_group", "arm")
        self.profile_path      = rospy.get_param("~throw_profile", self._default_profile_path())
        self.vel_scale         = rospy.get_param("~velocity_scaling", 1.0)
        self.acc_scale         = rospy.get_param("~acceleration_scaling", 1.0)
        self.cup_timeout       = rospy.get_param("~cup_position_timeout", 2.0)
        self.allow_placeholder = rospy.get_param("~allow_placeholder", False)

        # --- v2 parameters (inert in scaffold except fallback_to_v1) ------
        self.solver_mode          = rospy.get_param("~solver_mode", "lookup")
        self.solver_data          = rospy.get_param("~solver_data", self._default_solver_data_path())
        self.solver_timeout       = rospy.get_param("~solver_timeout", 0.5)
        self.release_angle_rad    = rospy.get_param("~release_angle_rad", None)
        self.z_release            = rospy.get_param("~z_release", None)
        self.record_landings      = rospy.get_param("~record_landings", False)
        self.fallback_to_v1       = rospy.get_param("~fallback_to_v1", True)
        self.bypass_moveit_timing = rospy.get_param("~bypass_moveit_timing", False)

        if self.bypass_moveit_timing:
            rospy.logwarn(
                "~bypass_moveit_timing=true — scaffold does NOT implement "
                "direct FollowJointTrajectory execution. Flag is a no-op "
                "until Step 7+.")

        # --- MoveIt -------------------------------------------------------
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.group = moveit_commander.MoveGroupCommander(self.group_name)
        self.group.set_max_velocity_scaling_factor(self.vel_scale)
        self.group.set_max_acceleration_scaling_factor(self.acc_scale)
        self.joint_names = list(self.group.get_active_joints())
        rospy.loginfo(f"MoveGroup '{self.group_name}' joints: {self.joint_names}")

        # --- Throw profile (loader is identical to v1) --------------------
        raw = self._load_profile(self.profile_path)
        self.ready_ordered, self.waypoints_ordered, self.aim, self.safety = \
            self._resolve_profile(raw)
        self._profile_is_placeholder = all(
            w == self.ready_ordered for _, w in self.waypoints_ordered)
        if self._profile_is_placeholder and not self.allow_placeholder:
            rospy.logwarn(
                "Throw profile is placeholder; v1-fallback throws will refuse. "
                "Set ~allow_placeholder:=true for dry runs.")

        # --- Solver + landing logger (stubs) ------------------------------
        self.solver = ThrowSolver(
            mode=self.solver_mode,
            data_path=self.solver_data,
            timeout=self.solver_timeout,
            release_angle_rad=self.release_angle_rad,
            z_release=self.z_release,
        )
        self.landing_logger = LandingLogger(
            data_path=self.solver_data,
            enabled=self.record_landings,
        )
        rospy.loginfo(
            f"Solver mode={self.solver_mode!r} (STUB: always returns None). "
            f"fallback_to_v1={self.fallback_to_v1}")

        # --- State --------------------------------------------------------
        self._state = self.STATE_IDLE
        self._state_lock = threading.Lock()
        self.latest_cup = None

        # --- ROS I/O ------------------------------------------------------
        self.pub_status = rospy.Publisher(
            "/robopong/motion_status", String, queue_size=1, latch=True)
        self.pub_debug = rospy.Publisher(
            "/robopong/throw_debug", String, queue_size=4)
        self.sub_cup = rospy.Subscriber(
            "/robopong/cup_position", PointStamped, self._cup_cb, queue_size=1)

        self.srv_ready = rospy.Service(
            "/robopong/go_ready", Trigger, self._handle_go_ready)
        self.srv_throw = rospy.Service(
            "/robopong/throw", Trigger, self._handle_throw)
        self.srv_calibrate = rospy.Service(
            "/robopong/calibrate_throw", Trigger, self._handle_calibrate_throw)
        # /robopong/record_landing is deferred until Step 2 — advertising it
        # requires the robopong/RecordLanding.srv type (not yet generated).

        self._publish_status("IDLE")
        rospy.loginfo("Motion Node v2 ready.")

    # ------------------------------------------------------------------
    # Status / debug
    # ------------------------------------------------------------------

    def _publish_status(self, status):
        self.pub_status.publish(String(data=status))

    def _publish_throw_debug(self, payload):
        # PRD §14.4 — every throw emits a debug blob, successful or not.
        self.pub_debug.publish(String(data=json.dumps(payload)))

    # ------------------------------------------------------------------
    # Profile loading + resolution (duplicated from v1; intended to be
    # factored into a shared module once v2 lands in the launch file).
    # ------------------------------------------------------------------

    def _default_profile_path(self):
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(pkg_dir, "../config/throw_profile.yaml")

    def _default_solver_data_path(self):
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(pkg_dir, "../config/solver_data.yaml")

    def _load_profile(self, path):
        rospy.loginfo(f"Loading throw profile: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        for key in ("ready_joints", "waypoints", "aim"):
            if key not in data:
                raise RuntimeError(f"throw profile missing required key: {key}")
        return data

    def _resolve_profile(self, raw):
        active = self.joint_names
        active_set = set(active)

        ready_dict = raw["ready_joints"]
        missing = active_set - set(ready_dict.keys())
        extra   = set(ready_dict.keys()) - active_set
        if missing or extra:
            raise RuntimeError(
                f"ready_joints mismatch. Missing: {sorted(missing)}  "
                f"Unknown: {sorted(extra)}  Active joints: {active}")
        ready_ordered = [float(ready_dict[name]) for name in active]

        waypoints_ordered = []
        for i, wp in enumerate(raw["waypoints"]):
            joints = wp.get("joints", {}) or {}
            bad = set(joints.keys()) - active_set
            if bad:
                raise RuntimeError(
                    f"waypoint[{i}] has unknown joints: {sorted(bad)}")
            ordered = list(ready_ordered)
            for name, val in joints.items():
                ordered[active.index(name)] = float(val)
            t = float(wp["time_from_start"])
            waypoints_ordered.append((t, ordered))

        times = [t for t, _ in waypoints_ordered]
        if not times or times[0] != 0.0:
            raise RuntimeError("first waypoint must have time_from_start: 0.0")
        if any(times[i] <= times[i-1] for i in range(1, len(times))):
            raise RuntimeError("waypoint time_from_start must be strictly increasing")

        aim = dict(raw["aim"])
        if aim.get("base_joint_name") not in active_set:
            raise RuntimeError(
                f"aim.base_joint_name '{aim.get('base_joint_name')}' not in "
                f"active joints {active}")
        aim["_base_idx"] = active.index(aim["base_joint_name"])
        aim.setdefault("base_offset_x", 0.0)
        aim.setdefault("base_offset_y", 0.0)
        aim.setdefault("aim_trim", 0.0)

        safety = raw.get("safety", {}) or {}
        safety.setdefault("base_joint_min", -np.pi)
        safety.setdefault("base_joint_max",  np.pi)

        return ready_ordered, waypoints_ordered, aim, safety

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _cup_cb(self, msg):
        with self._state_lock:
            if self._state != self.STATE_IDLE:
                return
        self.latest_cup = msg

    def _handle_go_ready(self, req):
        if not self._claim_busy():
            return TriggerResponse(success=False, message="busy")
        try:
            ok, err = self._go_ready()
            self._publish_status("READY" if ok else f"ERROR:go_ready:{err}")
            return TriggerResponse(success=ok, message=err or "ready")
        finally:
            self._release_busy()

    def _handle_calibrate_throw(self, req):
        # PRD §8.2 / Step 6. Scaffold advertises the service so callers can
        # discover it, but the real behaviour lands once the solver does.
        return TriggerResponse(
            success=False,
            message="calibrate_throw not implemented in scaffold (Step 6)")

    def _handle_throw(self, req):
        cup = self.latest_cup
        if cup is None:
            return TriggerResponse(success=False, message="no cup_position received")
        age = (rospy.Time.now() - cup.header.stamp).to_sec()
        if age > self.cup_timeout:
            return TriggerResponse(
                success=False, message=f"cup_position stale ({age:.1f}s)")

        if not self._claim_busy():
            return TriggerResponse(success=False, message="busy")

        # PRD §14.4 — always populate this and publish at the end.
        debug = {
            "cup": [cup.point.x, cup.point.y, cup.point.z],
            "solver_mode": self.solver_mode,
            "solver_result": None,
            "path": None,
            "ok": False,
            "message": None,
        }

        try:
            # Stage 1: solve. Stubbed in scaffold — always None.
            self._publish_status("SOLVING")
            solved = self.solver.solve(cup.point.x, cup.point.y)
            debug["solver_result"] = solved

            if solved is None:
                if not self.fallback_to_v1:
                    debug["message"] = "solver: no solution (fallback disabled)"
                    self._publish_status("ERROR:solver:no_solution")
                    return TriggerResponse(success=False, message=debug["message"])

                debug["path"] = "v1_fallback"
                if self._profile_is_placeholder and not self.allow_placeholder:
                    debug["message"] = "throw_profile is placeholder; fallback refused"
                    self._publish_status("ERROR:throw:placeholder")
                    return TriggerResponse(success=False, message=debug["message"])

                self._publish_status("THROWING")
                ok, err = self._throw_v1(cup.point.x, cup.point.y)
                debug["ok"] = ok
                debug["message"] = err or "throw ok (v1 fallback)"
                self._publish_status("IDLE" if ok else f"ERROR:throw:{err}")
                return TriggerResponse(success=ok, message=debug["message"])

            # Stage 2+: solver returned something. Not reachable in this
            # scaffold (solve() is a stub). When the real solver lands:
            #   - clamp speed_scale to [MIN, MAX]  (PRD §14.2)
            #   - build time-scaled trajectory      (PRD §7)
            #   - execute, verify MoveIt re-timing  (PRD §7)
            #   - log landing via self.landing_logger
            speed_scale, aim_angle = solved
            speed_scale = float(np.clip(speed_scale, SPEED_SCALE_MIN, SPEED_SCALE_MAX))
            debug["speed_scale"] = speed_scale
            debug["aim_angle"] = aim_angle
            debug["path"] = "v2_solver"
            debug["message"] = "v2 solver execution path not implemented in scaffold"
            self._publish_status("ERROR:throw:not_implemented")
            return TriggerResponse(success=False, message=debug["message"])

        finally:
            # PRD §14.4: debug always publishes.
            # PRD §14.5 ("arm returns to ready pose after every throw, even
            # on failure") is deferred — v1-fallback leaves the arm at the
            # last waypoint, matching v1 behaviour, and callers still drive
            # go_ready explicitly. Auto-return lands with the real v2 path.
            self._publish_throw_debug(debug)
            self._release_busy()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _claim_busy(self):
        with self._state_lock:
            if self._state != self.STATE_IDLE:
                return False
            self._state = self.STATE_BUSY
            return True

    def _release_busy(self):
        with self._state_lock:
            self._state = self.STATE_IDLE

    # ------------------------------------------------------------------
    # Motion primitives — v1 fallback path
    # (identical math to motion_node.py; duplicated for clean separation
    # while v1 is finalised)
    # ------------------------------------------------------------------

    def _go_ready(self):
        rospy.loginfo("Moving to ready pose")
        self.group.set_joint_value_target(self.ready_ordered)
        ok = self.group.go(wait=True)
        self.group.stop()
        return (ok, None if ok else "plan/execute failed")

    def _throw_v1(self, cup_x, cup_y):
        """Static-profile throw — same as motion_node.py::_throw_at."""
        dx = cup_x - float(self.aim["base_offset_x"])
        dy = cup_y - float(self.aim["base_offset_y"])
        base_angle = float(np.arctan2(dy, dx) + float(self.aim["aim_trim"]))

        bmin = float(self.safety["base_joint_min"])
        bmax = float(self.safety["base_joint_max"])
        if not (bmin <= base_angle <= bmax):
            return (False, f"base angle {base_angle:.2f} outside safety bounds")

        base_idx = self.aim["_base_idx"]
        aimed_ready = list(self.ready_ordered)
        aimed_ready[base_idx] = base_angle
        rospy.loginfo(
            f"Aiming: {self.aim['base_joint_name']} = {base_angle:.3f} rad")

        self.group.set_joint_value_target(aimed_ready)
        if not self.group.go(wait=True):
            self.group.stop()
            return (False, "failed to reach aimed ready pose")
        self.group.stop()

        traj = self._build_v1_trajectory(aimed_ready)
        ok = self.group.execute(traj, wait=True)
        self.group.stop()
        return (ok, None if ok else "throw trajectory execute failed")

    def _build_v1_trajectory(self, aimed_ready):
        base_idx = self.aim["_base_idx"]
        aim_delta = aimed_ready[base_idx] - self.ready_ordered[base_idx]

        jt = JointTrajectory()
        jt.joint_names = list(self.joint_names)
        for t, joints in self.waypoints_ordered:
            pt = JointTrajectoryPoint()
            shifted = list(joints)
            shifted[base_idx] += aim_delta
            pt.positions = shifted
            pt.time_from_start = rospy.Duration.from_sec(t)
            jt.points.append(pt)

        traj = RobotTrajectory()
        traj.joint_trajectory = jt
        return traj

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        MotionNodeV2().spin()
    except rospy.ROSInterruptException:
        pass
