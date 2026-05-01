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
    ~v_shoulder            : release velocity on arm_shoulder_lift_joint, rad/s
                             (default 2.5; clamped to [0.5, 5.0]). Primary distance knob.
    ~v_wrist               : release velocity on arm_wrist_1_joint, rad/s
                             (default 4.0; clamped to [0.5, 8.0]). Secondary snap knob.
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
        self.v_shoulder = float(np.clip(
            rospy.get_param("~v_shoulder", 2.5), 0.5, 5.0))
        self.v_wrist = float(np.clip(
            rospy.get_param("~v_wrist", 4.0), 0.5, 8.0))

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
        (self.ready_ordered, self.waypoints_ordered, self.aim, self.safety,
         self.ruckig_limits, self.release_ordered,
         self.follow_through_ordered) = self._resolve_profile(raw)
        self._profile_is_placeholder = (
            self.release_ordered == self.ready_ordered)
        rospy.loginfo(
            f"Ruckig limits loaded: v={self.ruckig_limits['max_velocity']} "
            f"a={self.ruckig_limits['max_acceleration']} "
            f"j={self.ruckig_limits['max_jerk']}")
        rospy.loginfo(
            f"Throw release velocities: v_shoulder={self.v_shoulder:.3f} rad/s, "
            f"v_wrist={self.v_wrist:.3f} rad/s")

        # Joint indices for the throw planner (kinematic-chain order from MoveIt)
        self._lift_idx = self.joint_names.index("arm_shoulder_lift_joint")
        self._wrist1_idx = self.joint_names.index("arm_wrist_1_joint")
        if self._profile_is_placeholder and not self.allow_placeholder:
            rospy.logwarn(
                "Throw profile placeholder (release_joints == ready_joints); "
                "/robopong/throw will refuse. Populate release_joints in "
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
        for key in ("ready_joints", "waypoints", "aim"):
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

        # waypoints are sparse dicts — fill missing joints from ready
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

        # Waypoints must be time-monotonic starting at 0
        times = [t for t, _ in waypoints_ordered]
        if not times or times[0] != 0.0:
            raise RuntimeError("first waypoint must have time_from_start: 0.0")
        if any(times[i] <= times[i-1] for i in range(1, len(times))):
            raise RuntimeError("waypoint time_from_start must be strictly increasing")

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
        rl_raw = raw.get("ruckig_limits")
        if rl_raw is None:
            raise RuntimeError("throw profile missing required key: ruckig_limits")
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

        # release_joints / follow_through_joints — sparse dicts, fill missing
        # joints from ready_joints. Pan is overridden by aim at throw time.
        def resolve_sparse(block_name):
            block = raw.get(block_name)
            if block is None:
                raise RuntimeError(f"throw profile missing required key: {block_name}")
            block = block or {}
            bad = set(block.keys()) - active_set
            if bad:
                raise RuntimeError(
                    f"{block_name} has unknown joints: {sorted(bad)}")
            ordered = list(ready_ordered)
            for name, val in block.items():
                if val is None:
                    raise RuntimeError(f"{block_name}.{name} is null — fill it in")
                ordered[active.index(name)] = float(val)
            return ordered

        release_ordered        = resolve_sparse("release_joints")
        follow_through_ordered = resolve_sparse("follow_through_joints")

        return (ready_ordered, waypoints_ordered, aim, safety, ruckig_limits,
                release_ordered, follow_through_ordered)

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

        # Ruckig-planned throw: A (aimed ready) → B (release) → C (follow-through),
        # concatenated and executed as one trajectory.
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
        Plans A→B (throw) and B→C (follow-through) as two sequential Ruckig
        state-to-state calls. Both are pre-computed, concatenated into one
        JointTrajectory, executed as a single group.execute() call. Ball
        leaves scoop during A→B execution.
        """
        n = len(self.joint_names)
        otg = Ruckig(n, 0.001)         # 1 ms control cycle
        inp = InputParameter(n)
        out = OutputParameter(n)

        inp.max_velocity     = list(self.ruckig_limits["max_velocity"])
        inp.max_acceleration = list(self.ruckig_limits["max_acceleration"])
        inp.max_jerk         = list(self.ruckig_limits["max_jerk"])

        pan_idx    = self.aim["_base_idx"]
        lift_idx   = self._lift_idx
        wrist1_idx = self._wrist1_idx

        # State A: aimed ready (other joints from ready_joints, pan = aim_angle)
        state_A_pos = list(self.ready_ordered)
        state_A_pos[pan_idx] = aim_angle

        # State B: release pose from yaml (release_joints), aim applied
        state_B_pos = list(self.release_ordered)
        state_B_pos[pan_idx] = aim_angle

        # State C: follow-through pose from yaml (follow_through_joints), aim applied
        state_C_pos = list(self.follow_through_ordered)
        state_C_pos[pan_idx] = aim_angle

        state_B_vel = [0.0] * n
        state_B_vel[lift_idx]   = self.v_shoulder
        state_B_vel[wrist1_idx] = self.v_wrist

        zeros = [0.0] * n

        def sample_segment(pos_start, vel_start, acc_start,
                           pos_end,   vel_end,   acc_end,
                           include_anchor):
            inp.current_position     = list(pos_start)
            inp.current_velocity     = list(vel_start)
            inp.current_acceleration = list(acc_start)
            inp.target_position      = list(pos_end)
            inp.target_velocity      = list(vel_end)
            inp.target_acceleration  = list(acc_end)

            points = []
            # Ruckig.update() advances one cycle before returning, so its first
            # sample is at t=0.001s, not t=0. For A->B we prepend an anchor at
            # the exact start state so MoveIt's first-point check passes; for
            # B->C the anchor is already the last point of A->B (skip it).
            if include_anchor:
                anchor = JointTrajectoryPoint()
                anchor.positions     = list(pos_start)
                anchor.velocities    = list(vel_start)
                anchor.accelerations = list(acc_start)
                anchor.time_from_start = rospy.Duration(0)
                points.append(anchor)

            t = 0.001
            while True:
                res = otg.update(inp, out)
                if res == Result.Error:
                    return None
                pt = JointTrajectoryPoint()
                pt.positions     = list(out.new_position)
                pt.velocities    = list(out.new_velocity)
                pt.accelerations = list(out.new_acceleration)
                pt.time_from_start = rospy.Duration.from_sec(t)
                points.append(pt)
                if res == Result.Finished:
                    break
                out.pass_to_input(inp)
                t += 0.001
            return points, t

        ab = sample_segment(state_A_pos, zeros,       zeros,
                            state_B_pos, state_B_vel, zeros,
                            include_anchor=True)
        if ab is None:
            return None, "Ruckig failed on A->B segment"
        points_AB, duration_AB = ab

        bc = sample_segment(state_B_pos, state_B_vel, zeros,
                            state_C_pos, zeros,       zeros,
                            include_anchor=False)
        if bc is None:
            return None, "Ruckig failed on B->C segment"
        points_BC, duration_BC = bc

        offset = rospy.Duration.from_sec(duration_AB)
        for pt in points_BC:
            pt.time_from_start += offset

        jt = JointTrajectory()
        jt.joint_names = list(self.joint_names)
        jt.points = points_AB + points_BC

        debug = {
            "aim_angle_rad": round(float(aim_angle), 4),
            "v_shoulder":    round(self.v_shoulder, 4),
            "v_wrist":       round(self.v_wrist, 4),
            "duration_AB_s": round(duration_AB, 4),
            "duration_BC_s": round(duration_BC, 4),
            "total_points":  len(jt.points),
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
