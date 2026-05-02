#!/usr/bin/env python3
"""
RoboPong - Motion Node
-----------------------
Moves the arm to a 'ready to throw' pose and executes throw trajectories
aimed at a cup detected by vision_node.

Subscribes to:
    /robopong/cup_position      - geometry_msgs/PointStamped  (ignored while mid-throw)

Publishes:
    /robopong/motion_status     - std_msgs/String  (IDLE / READY / THROWING / ERROR:<detail>)

Services:
    /robopong/go_ready          - std_srvs/Trigger  (move to ready pose)
    /robopong/throw             - std_srvs/Trigger  (throw at latest cup_position)

Parameters:
    ~move_group            : MoveIt group name (default "arm" — lab setup)
    ~throw_profile         : path to throw_profile.yaml (default: config/throw_profile.yaml)
    ~velocity_scaling      : MoveIt max_velocity_scaling_factor (default 1.0)
    ~acceleration_scaling  : MoveIt max_acceleration_scaling_factor (default 1.0)
    ~cup_position_timeout  : seconds after which latest cup_position is stale (default 2.0)
    ~allow_placeholder     : permit throws even if profile is placeholder (default False)
    ~velocity_scale        : multiplier applied to every velocity in every
                             throw_waypoint at planning time (default 1.0).
                             Lab knob for ramping; e.g. velocity_scale:=0.25 for
                             a quarter-speed first throw without editing yaml.
"""

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

from ruckig import InputParameter, OutputParameter, Result, Ruckig


class MotionNode:
    STATE_IDLE = "IDLE"
    STATE_BUSY = "BUSY"

    def __init__(self):
        rospy.init_node("motion_node", anonymous=False)
        rospy.loginfo("RoboPong Motion Node starting...")

        # --- Parameters ---
        self.group_name        = rospy.get_param("~move_group", "arm")
        self.profile_path      = rospy.get_param("~throw_profile", self._default_profile_path())
        self.vel_scale         = rospy.get_param("~velocity_scaling", 1.0)
        self.acc_scale         = rospy.get_param("~acceleration_scaling", 1.0)
        self.cup_timeout       = rospy.get_param("~cup_position_timeout", 2.0)
        self.allow_placeholder = rospy.get_param("~allow_placeholder", False)
        self.velocity_scale = float(rospy.get_param("~velocity_scale", 1.0))

        # --- MoveIt ---
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.group = moveit_commander.MoveGroupCommander(self.group_name)
        self.group.set_max_velocity_scaling_factor(self.vel_scale)
        self.group.set_max_acceleration_scaling_factor(self.acc_scale)
        self.joint_names = list(self.group.get_active_joints())
        rospy.loginfo(f"MoveGroup '{self.group_name}' joints: {self.joint_names}")

        # --- Throw profile (load + resolve against active joint order) ---
        raw = self._load_profile(self.profile_path)
        (self.ready_ordered, self.aim, self.safety,
         self.ruckig_limits, self.throw_waypoints) = self._resolve_profile(raw)
        # Placeholder: every throw waypoint leaves the arm at ready (nothing moves).
        self._profile_is_placeholder = all(
            wp["positions"] == self.ready_ordered for wp in self.throw_waypoints)
        rospy.loginfo(
            f"Ruckig limits loaded: v={self.ruckig_limits['max_velocity']} "
            f"a={self.ruckig_limits['max_acceleration']} "
            f"j={self.ruckig_limits['max_jerk']}")
        rospy.loginfo(
            f"Throw waypoints ({len(self.throw_waypoints)}): "
            + ", ".join(wp["name"] for wp in self.throw_waypoints))
        rospy.loginfo(f"velocity_scale = {self.velocity_scale:.3f}")
        if self._profile_is_placeholder and not self.allow_placeholder:
            rospy.logwarn(
                "Throw profile placeholder (every throw_waypoint == ready_joints); "
                "/robopong/throw will refuse. Populate throw_waypoints in "
                "throw_profile.yaml or set ~allow_placeholder:=true.")

        # --- State ---
        self._state = self.STATE_IDLE
        self._state_lock = threading.Lock()
        self.latest_cup = None

        # --- ROS I/O ---
        self.pub_status = rospy.Publisher(
            "/robopong/motion_status", String, queue_size=1, latch=True)
        self.pub_throw_debug = rospy.Publisher(
            "/robopong/throw_debug", String, queue_size=1, latch=True)
        self.sub_cup = rospy.Subscriber(
            "/robopong/cup_position", PointStamped, self._cup_cb, queue_size=1)
        self.srv_ready = rospy.Service(
            "/robopong/go_ready", Trigger, self._handle_go_ready)
        self.srv_throw = rospy.Service(
            "/robopong/throw", Trigger, self._handle_throw)

        self._publish_status("IDLE")
        rospy.loginfo("Motion Node ready.")

    def _publish_status(self, status):
        self.pub_status.publish(String(data=status))

    # ------------------------------------------------------------------
    # Profile loading + resolution (dict → ordered list by joint name)
    # ------------------------------------------------------------------

    def _default_profile_path(self):
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(pkg_dir, "../config/throw_profile.yaml")

    def _load_profile(self, path):
        rospy.loginfo(f"Loading throw profile: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        for key in ("ready_joints", "throw_waypoints", "aim", "ruckig_limits"):
            if key not in data:
                raise RuntimeError(f"throw profile missing required key: {key}")
        return data

    def _resolve_profile(self, raw):
        active = self.joint_names
        active_set = set(active)

        # ready_joints must list every active joint exactly once
        ready_dict = raw["ready_joints"]
        missing = active_set - set(ready_dict.keys())
        extra   = set(ready_dict.keys()) - active_set
        if missing or extra:
            raise RuntimeError(
                f"ready_joints mismatch. Missing: {sorted(missing)}  "
                f"Unknown: {sorted(extra)}  Active joints: {active}")
        ready_ordered = [float(ready_dict[name]) for name in active]

        # aim
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

        # Ruckig kinematic limits — remap from yaml joint_order to active order
        rl_raw = raw["ruckig_limits"]
        yaml_order = rl_raw.get("joint_order")
        if yaml_order is None:
            raise RuntimeError("ruckig_limits.joint_order is required")
        if set(yaml_order) != active_set:
            raise RuntimeError(
                f"ruckig_limits.joint_order mismatch. Got {yaml_order}  "
                f"Active joints: {active}")
        ruckig_limits = {}
        for key in ("max_velocity", "max_acceleration", "max_jerk"):
            vals = rl_raw.get(key)
            if vals is None or len(vals) != len(yaml_order):
                raise RuntimeError(
                    f"ruckig_limits.{key} must be a list of length {len(yaml_order)}")
            by_name = dict(zip(yaml_order, vals))
            ruckig_limits[key] = [float(by_name[name]) for name in active]

        # throw_waypoints — list of {name, joints, velocities, accelerations,
        # duration}. joints/velocities/accelerations are sparse dicts
        # (joints default to ready, velocities and accelerations default to 0).
        # duration is an optional float (seconds) that pins the segment ending
        # at this waypoint to a fixed duration via Ruckig's minimum_duration;
        # omit/null lets Ruckig pick the time-optimal duration. Pan is
        # overridden at runtime.
        wp_raw = raw["throw_waypoints"]
        if not isinstance(wp_raw, list) or len(wp_raw) < 1:
            raise RuntimeError("throw_waypoints must be a non-empty list")
        n = len(active)
        def _resolve_sparse(d, label, i, name):
            bad = set(d.keys()) - active_set
            if bad:
                raise RuntimeError(
                    f"throw_waypoints[{i}] '{name}' has unknown {label}: {sorted(bad)}")
            out = [0.0] * n
            for jname, val in d.items():
                if val is None:
                    raise RuntimeError(
                        f"throw_waypoints[{i}] '{name}' {label} {jname} is null")
                out[active.index(jname)] = float(val)
            return out
        throw_waypoints = []
        for i, wp in enumerate(wp_raw):
            name = wp.get("name", f"wp{i}")
            joints = wp.get("joints", {}) or {}
            vels   = wp.get("velocities", {}) or {}
            accs   = wp.get("accelerations", {}) or {}
            bad_j = set(joints.keys()) - active_set
            if bad_j:
                raise RuntimeError(
                    f"throw_waypoints[{i}] '{name}' has unknown joints: {sorted(bad_j)}")
            positions = list(ready_ordered)
            for jname, val in joints.items():
                if val is None:
                    raise RuntimeError(
                        f"throw_waypoints[{i}] '{name}' joint {jname} is null")
                positions[active.index(jname)] = float(val)
            velocities    = _resolve_sparse(vels, "velocities",    i, name)
            accelerations = _resolve_sparse(accs, "accelerations", i, name)
            duration = wp.get("duration", None)
            if duration is not None:
                duration = float(duration)
                if duration <= 0.0:
                    raise RuntimeError(
                        f"throw_waypoints[{i}] '{name}' duration must be > 0")
            throw_waypoints.append({
                "name":          name,
                "positions":     positions,
                "velocities":    velocities,
                "accelerations": accelerations,
                "duration":      duration,
            })

        # Final waypoint should bring the arm to rest (else logwarn).
        if any(v != 0.0 for v in throw_waypoints[-1]["velocities"]):
            rospy.logwarn(
                f"Final throw_waypoint '{throw_waypoints[-1]['name']}' has "
                f"non-zero velocities; arm will not stop at end of throw.")
        # Intermediate waypoints should carry non-zero velocity through —
        # otherwise the throw degenerates into discrete carry-and-place moves.
        for wp in throw_waypoints[:-1]:
            if not any(v != 0.0 for v in wp["velocities"]):
                rospy.logwarn(
                    f"Intermediate throw_waypoint '{wp['name']}' has all-zero "
                    f"velocities; the arm will fully stop here, defeating the "
                    f"release-with-velocity design.")

        return ready_ordered, aim, safety, ruckig_limits, throw_waypoints

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

    def _handle_throw(self, req):
        if self._profile_is_placeholder and not self.allow_placeholder:
            return TriggerResponse(
                success=False,
                message="throw_profile is placeholder; tune it or set ~allow_placeholder:=true")
        cup = self.latest_cup
        if cup is None:
            return TriggerResponse(success=False, message="no cup_position received")
        age = (rospy.Time.now() - cup.header.stamp).to_sec()
        if age > self.cup_timeout:
            return TriggerResponse(
                success=False, message=f"cup_position stale ({age:.1f}s)")
        if not self._claim_busy():
            return TriggerResponse(success=False, message="busy")
        try:
            self._publish_status("THROWING")
            ok, err = self._throw_at(cup.point.x, cup.point.y)
            self._publish_status("IDLE" if ok else f"ERROR:throw:{err}")
            return TriggerResponse(success=ok, message=err or "throw ok")
        finally:
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
    # Motion primitives
    # ------------------------------------------------------------------

    def _go_ready(self):
        rospy.loginfo("Moving to ready pose")
        self.group.set_joint_value_target(self.ready_ordered)
        ok = self.group.go(wait=True)
        self.group.stop()
        return (ok, None if ok else "plan/execute failed")

    def _throw_at(self, cup_x, cup_y):
        # Aim: compute base-joint angle pointing arm at the cup
        dx = cup_x - float(self.aim["base_offset_x"])
        dy = cup_y - float(self.aim["base_offset_y"])
        base_angle = float(np.arctan2(dy, dx) + float(self.aim["aim_trim"]))

        bmin = float(self.safety["base_joint_min"])
        bmax = float(self.safety["base_joint_max"])
        if not (bmin <= base_angle <= bmax):
            return (False, f"base angle {base_angle:.2f} outside safety bounds")

        # Build aim-adjusted ready pose and move there
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

        # Ruckig-planned throw: aimed_ready → throw_waypoints[0] → ... →
        # throw_waypoints[-1], planned per-segment and concatenated into one
        # trajectory executed as a single group.execute() call.
        jt, err = self._plan_throw_ruckig(base_angle)
        if jt is None:
            return (False, err)
        traj = RobotTrajectory()
        traj.joint_trajectory = jt
        ok = self.group.execute(traj, wait=True)
        self.group.stop()
        return (ok, None if ok else "execution failed")

    def _plan_throw_ruckig(self, aim_angle):
        """
        Plans aimed_ready -> throw_waypoints[0] -> ... -> throw_waypoints[-1]
        as a sequence of Ruckig state-to-state segments, concatenated into a
        single JointTrajectory. Each waypoint contributes its position,
        velocity, and acceleration as Ruckig boundary conditions; an optional
        per-waypoint duration pins that segment's length via minimum_duration.
        Waypoint velocities are multiplied by ~velocity_scale at plan time.
        """
        n = len(self.joint_names)
        otg = Ruckig(n, 0.001)         # 1 ms control cycle
        inp = InputParameter(n)
        out = OutputParameter(n)
        inp.max_velocity     = list(self.ruckig_limits["max_velocity"])
        inp.max_acceleration = list(self.ruckig_limits["max_acceleration"])
        inp.max_jerk         = list(self.ruckig_limits["max_jerk"])

        pan_idx = self.aim["_base_idx"]
        zeros = [0.0] * n

        # Build state list: aimed_ready, then every throw_waypoint with pan
        # overridden by runtime aim and velocities scaled by ~velocity_scale.
        aimed_ready = list(self.ready_ordered)
        aimed_ready[pan_idx] = aim_angle
        states = [{"name": "aimed_ready",
                   "positions":     aimed_ready,
                   "velocities":    list(zeros),
                   "accelerations": list(zeros),
                   "duration":      None}]
        for wp in self.throw_waypoints:
            pos = list(wp["positions"])
            pos[pan_idx] = aim_angle
            vel = [v * self.velocity_scale for v in wp["velocities"]]
            states.append({
                "name":          wp["name"],
                "positions":     pos,
                "velocities":    vel,
                "accelerations": list(wp["accelerations"]),
                "duration":      wp["duration"],
            })

        points = []
        cumulative = 0.0
        seg_debug = []
        for i in range(len(states) - 1):
            a, b = states[i], states[i + 1]
            inp.current_position     = list(a["positions"])
            inp.current_velocity     = list(a["velocities"])
            inp.current_acceleration = list(a["accelerations"])
            inp.target_position      = list(b["positions"])
            inp.target_velocity      = list(b["velocities"])
            inp.target_acceleration  = list(b["accelerations"])
            if b["duration"] is not None:
                inp.minimum_duration = float(b["duration"])
            else:
                # Clear any previous segment's minimum_duration.
                try:
                    inp.minimum_duration = 0.0
                except Exception:
                    pass

            # Anchor the very first sample at t=0 (Ruckig's first sample is
            # at t=0.001s, and MoveIt expects the trajectory to start at the
            # robot's current state).
            if i == 0:
                anchor = JointTrajectoryPoint()
                anchor.positions     = list(a["positions"])
                anchor.velocities    = list(a["velocities"])
                anchor.accelerations = list(a["accelerations"])
                anchor.time_from_start = rospy.Duration(0)
                points.append(anchor)

            t = 0.0
            while True:
                res = otg.update(inp, out)
                if res == Result.Error:
                    return None, f"Ruckig failed on segment {a['name']} -> {b['name']}"
                t += 0.001
                pt = JointTrajectoryPoint()
                pt.positions     = list(out.new_position)
                pt.velocities    = list(out.new_velocity)
                pt.accelerations = list(out.new_acceleration)
                pt.time_from_start = rospy.Duration.from_sec(cumulative + t)
                points.append(pt)
                if res == Result.Finished:
                    break
                out.pass_to_input(inp)

            seg_debug.append({
                "segment":         f"{a['name']}->{b['name']}",
                "duration_s":      round(t, 4),
                "duration_pinned": b["duration"] is not None,
            })
            cumulative += t

        jt = JointTrajectory()
        jt.joint_names = list(self.joint_names)
        jt.points = points

        debug = {
            "aim_angle_rad":     round(float(aim_angle), 4),
            "velocity_scale":    round(self.velocity_scale, 4),
            "segments":          seg_debug,
            "waypoints_applied": [
                {"name":          s["name"],
                 "velocities":    [round(v, 4) for v in s["velocities"]],
                 "accelerations": [round(a, 4) for a in s["accelerations"]],
                 "duration":      s["duration"]}
                for s in states[1:]
            ],
            "total_points":      len(jt.points),
            "total_duration_s":  round(cumulative, 4),
        }
        self.pub_throw_debug.publish(String(data=json.dumps(debug)))
        rospy.loginfo(f"Ruckig throw plan: {debug}")

        return jt, None

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        MotionNode().spin()
    except rospy.ROSInterruptException:
        pass