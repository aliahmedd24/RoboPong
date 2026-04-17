#!/usr/bin/env python3
"""
RoboPong - Motion Node
-----------------------
Moves the arm to a 'ready to throw' pose and executes throw trajectories
aimed at a cup detected by vision_node.

Subscribes to:
    /robopong/cup_position      - geometry_msgs/PointStamped  (ignored while mid-throw)

Services:
    /robopong/go_ready          - std_srvs/Trigger  (move to ready pose)
    /robopong/throw             - std_srvs/Trigger  (throw at latest cup_position)

Parameters:
    ~move_group            : MoveIt group name (default "manipulator")
    ~throw_profile         : path to throw_profile.yaml (default: config/throw_profile.yaml)
    ~velocity_scaling      : MoveIt max_velocity_scaling_factor (default 1.0)
    ~acceleration_scaling  : MoveIt max_acceleration_scaling_factor (default 1.0)
    ~cup_position_timeout  : seconds after which latest cup_position is stale (default 2.0)
    ~allow_placeholder     : permit throws even if profile is placeholder (default False)
"""

import os
import sys
import threading

import rospy
import yaml
import numpy as np

from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory

import moveit_commander


class MotionNode:
    STATE_IDLE = "IDLE"
    STATE_BUSY = "BUSY"

    def __init__(self):
        rospy.init_node("motion_node", anonymous=False)
        rospy.loginfo("RoboPong Motion Node starting...")

        # --- Parameters ---
        self.group_name        = rospy.get_param("~move_group", "manipulator")
        self.profile_path      = rospy.get_param("~throw_profile", self._default_profile_path())
        self.vel_scale         = rospy.get_param("~velocity_scaling", 1.0)
        self.acc_scale         = rospy.get_param("~acceleration_scaling", 1.0)
        self.cup_timeout       = rospy.get_param("~cup_position_timeout", 2.0)
        self.allow_placeholder = rospy.get_param("~allow_placeholder", False)

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
        self.ready_ordered, self.waypoints_ordered, self.aim, self.safety = \
            self._resolve_profile(raw)
        self._profile_is_placeholder = all(
            w == self.ready_ordered for _, w in self.waypoints_ordered)
        if self._profile_is_placeholder and not self.allow_placeholder:
            rospy.logwarn(
                "Throw profile is placeholder (all waypoints == ready_joints); "
                "/robopong/throw will refuse. Tune throw_profile.yaml or set "
                "~allow_placeholder:=true for dry runs.")

        # --- State ---
        self._state = self.STATE_IDLE
        self._state_lock = threading.Lock()
        self.latest_cup = None

        # --- ROS I/O ---
        self.sub_cup = rospy.Subscriber(
            "/robopong/cup_position", PointStamped, self._cup_cb, queue_size=1)
        self.srv_ready = rospy.Service(
            "/robopong/go_ready", Trigger, self._handle_go_ready)
        self.srv_throw = rospy.Service(
            "/robopong/throw", Trigger, self._handle_throw)

        rospy.loginfo("Motion Node ready.")

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
            ok, err = self._throw_at(cup.point.x, cup.point.y)
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

        # Execute throw trajectory (waypoints shifted by aim delta on base joint)
        traj = self._build_throw_trajectory(aimed_ready)
        ok = self.group.execute(traj, wait=True)
        self.group.stop()
        return (ok, None if ok else "throw trajectory execute failed")

    def _build_throw_trajectory(self, aimed_ready):
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
        # NOTE: MoveIt may re-time this to respect joint velocity/accel limits.
        # If throw speed is insufficient after tuning, switch to sending the
        # JointTrajectory directly via the controller's FollowJointTrajectory
        # action instead of group.execute().
        return traj

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        MotionNode().spin()
    except rospy.ROSInterruptException:
        pass
