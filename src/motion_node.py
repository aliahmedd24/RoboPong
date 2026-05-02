#!/usr/bin/env python3
"""
RoboPong - Motion Node
-----------------------
Moves the arm to a 'ready to throw' pose and executes throw trajectories
aimed at a cup detected by vision_node.

Throw mechanism:
    Ruckig-planned 3-DoF swing on the active throw joints
    (shoulder_lift + elbow + wrist_1) from READY_JOINTS to LAUNCH_JOINTS,
    sampled at the controller cycle and executed as a RobotTrajectory via
    moveit_commander's execute(). The plan ends at the launch pose with
    non-zero target velocities on all three throw joints; once the
    trajectory finishes the controller holds the launch pose, which forces
    the arm to brake abruptly. The ball's inertia carries it out of the
    scoop (the inertial throw). Pan + wrist_2/3 are held constant.

    Joint roles:
      shoulder_lift : heavy upward sweep (handles the abrupt brake well)
      elbow         : forward extension / horizontal velocity
      wrist_1       : keeps the flat scoop tilted into the trajectory so
                      the acceleration vector pushes the ball into the
                      scoop face rather than off the back

Subscribes to:
    /robopong/cup_position      - geometry_msgs/PointStamped  (ignored mid-throw)

Publishes:
    /robopong/motion_status     - std_msgs/String  (IDLE / READY / THROWING / ERROR:<detail>)
    /robopong/throw_debug       - std_msgs/String  (json plan summary)

Services:
    /robopong/go_ready          - std_srvs/Trigger
    /robopong/throw             - std_srvs/Trigger

Parameters:
    ~move_group            : MoveIt group name (default "arm")
    ~velocity_scaling      : MoveIt max_velocity_scaling_factor (default 1.0)
    ~acceleration_scaling  : MoveIt max_acceleration_scaling_factor (default 1.0)
    ~cup_position_timeout  : seconds after which latest cup_position is stale
    ~release_velocity_lift   : signed shoulder_lift velocity at launch pose (rad/s)
    ~release_velocity_elbow  : signed elbow velocity at launch pose (rad/s)
    ~release_velocity_wrist1 : signed wrist_1 velocity at launch pose (rad/s)
    ~max_velocity_lift / _elbow / _wrist1     : per-joint Ruckig vmax (rad/s)
    ~max_acceleration_lift / _elbow / _wrist1 : per-joint Ruckig amax (rad/s^2)
    ~max_jerk_lift / _elbow / _wrist1         : per-joint Ruckig jmax (rad/s^3)
    ~control_cycle           : Ruckig sample period (s, default 0.008)
"""

import json
import sys
import threading

import rospy
import numpy as np
import ruckig

from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory

import moveit_commander


class MotionNode:
    STATE_IDLE = "IDLE"
    STATE_BUSY = "BUSY"

    SHOULDER_LIFT = "arm_shoulder_lift_joint"
    ELBOW         = "arm_elbow_joint"
    WRIST_1       = "arm_wrist_1_joint"

    # The 3 active throw joints, in the order Ruckig is initialised with.
    THROW_JOINTS = (SHOULDER_LIFT, ELBOW, WRIST_1)

    # ==================================================================
    # Lab-tuned constants — edit here to retune.
    # Joint names use the lab's "arm_" tf_prefix.
    # ==================================================================

    # Pose A — "ready to throw" (top of underarm swing). Pan is overridden
    # by the aim model at runtime; the captured pan is just a parking value.
    READY_JOINTS = {
        "arm_shoulder_pan_joint":  -0.022503,
        "arm_shoulder_lift_joint": -0.282995,
        "arm_elbow_joint":          2.299617,
        "arm_wrist_1_joint":       -1.818120,
        "arm_wrist_2_joint":        1.581818,
        "arm_wrist_3_joint":        0.060110,
    }

    # Pose B — launch / abrupt-stop pose (end of forward extension). The
    # active throw joints (shoulder_lift, elbow, wrist_1) all displace from
    # pose A; pan + wrist_2/3 are held constant. shoulder_lift target is
    # taken from the original captured launch sample; elbow + wrist_1
    # offsets are tuned so the scoop face stays normal-ish to the
    # acceleration vector through the swing — retune in the lab.
    LAUNCH_JOINTS = {
        "arm_shoulder_pan_joint":  -0.023565,
        "arm_shoulder_lift_joint": -1.050000,   # less truncated; more sweep, still rising at brake
        "arm_elbow_joint":          1.400000,   # large forward extension (Δ ~0.9 rad), elbow is the launcher
        "arm_wrist_1_joint":       -1.000000,   # KEEP scoop face angled up-forward at brake
        "arm_wrist_2_joint":        1.581403,
        "arm_wrist_3_joint":        0.060168,
    }

    # Aim model:
    #   base_angle = atan2(cup_y - BASE_OFFSET_Y, cup_x - BASE_OFFSET_X) + AIM_TRIM
    BASE_JOINT_NAME = "arm_shoulder_pan_joint"
    BASE_OFFSET_X   = 0.0
    BASE_OFFSET_Y   = 0.0
    AIM_TRIM        = 0.0

    # Safety — throw is rejected if aim-adjusted base angle leaves this range.
    BASE_JOINT_MIN = -1.5708
    BASE_JOINT_MAX =  1.5708

    # Aim toggle. While False, the shoulder_pan is left at its READY value
    # for every throw and the cup_position / aim math is bypassed. Flip back
    # to True once the throw motion itself is dialled in.
    AIM_ENABLED = False

    # Per-joint Ruckig kinematic limits, indexed [shoulder_lift, elbow, wrist_1].
    # shoulder_lift carries the heavy sweep, wrist_1 is the snappy scoop tilt
    # joint so it can take a higher jerk/acc.
    DEFAULT_MAX_VELOCITY     = (3.0, 4.0,  6.0)   # rad/s — elbow vmax raised to allow -3.5 rad/s release
    DEFAULT_MAX_ACCELERATION = (20.0, 20.0, 25.0) # rad/s^2
    DEFAULT_MAX_JERK         = (60.0, 60.0, 120.0) # rad/s^3
    DEFAULT_CONTROL_CYCLE    = 0.008  # s (matches UR servo loop)

    # Velocities at the launch pose, signed (rad/s). Each is the velocity of
    # its joint at the moment the controller's hold of the launch pose forces
    # the abrupt brake — so non-zero values determine launch energy. Signs
    # follow the pose-A → pose-B direction of travel:
    #   shoulder_lift: pose A → B is decreasing (negative) → release neg
    #   elbow:         pose A → B is decreasing (negative) → release neg
    #   wrist_1:       pose A → B is increasing (positive) → release pos
    DEFAULT_RELEASE_VELOCITY = (-2.5, -3.5, 1.0)

    # Hold after the trajectory action returns, to ensure the controller has
    # fully nulled residual velocity before we accept new commands.
    COMPLETION_WAIT = 0.5

    def __init__(self):
        rospy.init_node("motion_node", anonymous=False)
        rospy.loginfo("RoboPong Motion Node starting...")

        # --- Parameters ---
        self.group_name       = rospy.get_param("~move_group", "arm")
        self.vel_scale        = rospy.get_param("~velocity_scaling", 1.0)
        self.acc_scale        = rospy.get_param("~acceleration_scaling", 1.0)
        self.cup_timeout      = rospy.get_param("~cup_position_timeout", 2.0)

        # 3-DoF throw vectors: [shoulder_lift, elbow, wrist_1]
        suffixes = ("lift", "elbow", "wrist1")
        self.release_velocity = [
            float(rospy.get_param(f"~release_velocity_{s}",
                                  self.DEFAULT_RELEASE_VELOCITY[i]))
            for i, s in enumerate(suffixes)]
        self.max_velocity = [
            float(rospy.get_param(f"~max_velocity_{s}",
                                  self.DEFAULT_MAX_VELOCITY[i]))
            for i, s in enumerate(suffixes)]
        self.max_acceleration = [
            float(rospy.get_param(f"~max_acceleration_{s}",
                                  self.DEFAULT_MAX_ACCELERATION[i]))
            for i, s in enumerate(suffixes)]
        self.max_jerk = [
            float(rospy.get_param(f"~max_jerk_{s}",
                                  self.DEFAULT_MAX_JERK[i]))
            for i, s in enumerate(suffixes)]
        self.control_cycle = float(rospy.get_param(
            "~control_cycle", self.DEFAULT_CONTROL_CYCLE))

        # --- MoveIt ---
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.group = moveit_commander.MoveGroupCommander(self.group_name)
        self.group.set_max_velocity_scaling_factor(self.vel_scale)
        self.group.set_max_acceleration_scaling_factor(self.acc_scale)
        self.joint_names = list(self.group.get_active_joints())
        rospy.loginfo(f"MoveGroup '{self.group_name}' joints: {self.joint_names}")

        (self.ready_ordered, self.launch_ordered,
         self.base_idx, self.throw_idx) = self._resolve_against_active()

        rospy.loginfo(
            f"Ruckig vmax={self.max_velocity} amax={self.max_acceleration} "
            f"jmax={self.max_jerk} cycle={self.control_cycle}")
        rospy.loginfo(f"throw joints       = {list(self.THROW_JOINTS)}")
        rospy.loginfo(f"release_velocity   = {self.release_velocity}")

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

    def _resolve_against_active(self):
        """Validate constants vs MoveIt active joints; return ordered ready/
        launch poses, the base-joint index, and the throw-joint indices
        (one per entry in THROW_JOINTS, same order)."""
        active = self.joint_names
        active_set = set(active)

        for label, table in (("READY_JOINTS", self.READY_JOINTS),
                             ("LAUNCH_JOINTS", self.LAUNCH_JOINTS)):
            missing = active_set - set(table.keys())
            extra   = set(table.keys()) - active_set
            if missing or extra:
                raise RuntimeError(
                    f"{label} mismatch. Missing: {sorted(missing)}  "
                    f"Unknown: {sorted(extra)}  Active joints: {active}")

        ready_ordered  = [float(self.READY_JOINTS[n])  for n in active]
        launch_ordered = [float(self.LAUNCH_JOINTS[n]) for n in active]

        if self.BASE_JOINT_NAME not in active_set:
            raise RuntimeError(
                f"BASE_JOINT_NAME '{self.BASE_JOINT_NAME}' not in {active}")
        for jn in self.THROW_JOINTS:
            if jn not in active_set:
                raise RuntimeError(f"throw joint '{jn}' not in {active}")

        base_idx  = active.index(self.BASE_JOINT_NAME)
        throw_idx = [active.index(jn) for jn in self.THROW_JOINTS]
        return ready_ordered, launch_ordered, base_idx, throw_idx

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
        # Aim disabled: skip the cup check and throw straight ahead from
        # the unmodified ready pose. Re-enable by setting AIM_ENABLED = True.
        if self.AIM_ENABLED:
            cup = self.latest_cup
            if cup is None:
                return TriggerResponse(success=False, message="no cup_position received")
            age = (rospy.Time.now() - cup.header.stamp).to_sec()
            if age > self.cup_timeout:
                return TriggerResponse(
                    success=False, message=f"cup_position stale ({age:.1f}s)")
            cup_x, cup_y = cup.point.x, cup.point.y
        else:
            cup_x, cup_y = None, None

        if not self._claim_busy():
            return TriggerResponse(success=False, message="busy")
        try:
            self._publish_status("THROWING")
            ok, err = self._throw_at(cup_x, cup_y)
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
        if self.AIM_ENABLED:
            # Aim: compute base-joint angle pointing arm at the cup.
            dx = cup_x - self.BASE_OFFSET_X
            dy = cup_y - self.BASE_OFFSET_Y
            base_angle = float(np.arctan2(dy, dx) + self.AIM_TRIM)

            if not (self.BASE_JOINT_MIN <= base_angle <= self.BASE_JOINT_MAX):
                return (False, f"base angle {base_angle:.2f} outside safety bounds")

            rospy.loginfo(f"Aiming: {self.BASE_JOINT_NAME} = {base_angle:.3f} rad")
        else:
            # Aim disabled — use the READY pan value unchanged.
            base_angle = float(self.READY_JOINTS[self.BASE_JOINT_NAME])
            rospy.loginfo(f"Aim disabled; pan held at READY = {base_angle:.3f} rad")

        # Wind-up: ready pose (aim-adjusted only if aim enabled) via MoveIt.
        aimed_ready = list(self.ready_ordered)
        aimed_ready[self.base_idx] = base_angle

        self.group.set_joint_value_target(aimed_ready)
        if not self.group.go(wait=True):
            self.group.stop()
            return (False, "failed to reach aimed ready pose")
        self.group.stop()

        # Plan the Ruckig swing and execute via moveit_commander. The base
        # joint is held at base_angle so the aim achieved by MoveIt is
        # preserved through the throw.
        try:
            traj = self._plan_ruckig_swing(base_angle)
        except RuntimeError as e:
            return (False, f"ruckig plan failed: {e}")

        rt = RobotTrajectory()
        rt.joint_trajectory = traj
        rospy.loginfo(
            f"Executing Ruckig swing: {len(traj.points)} pts, "
            f"end vel = {self.release_velocity}")
        if not self.group.execute(rt, wait=True):
            self.group.stop()
            return (False, "throw trajectory execution failed")
        self.group.stop()

        rospy.sleep(self.COMPLETION_WAIT)
        return (True, None)

    def _plan_ruckig_swing(self, base_angle):
        """3-DoF Ruckig plan over [shoulder_lift, elbow, wrist_1] from
        READY → LAUNCH, ending with self.release_velocity. Pan + wrist_2/3
        are held at their READY (= aimed_ready) values throughout. Time-
        synchronised across the three throw joints by Ruckig. Returns a
        trajectory_msgs/JointTrajectory ordered to match the MoveIt group."""
        n = len(self.THROW_JOINTS)
        start = [float(self.READY_JOINTS[j])  for j in self.THROW_JOINTS]
        end   = [float(self.LAUNCH_JOINTS[j]) for j in self.THROW_JOINTS]

        otg = ruckig.Ruckig(n, self.control_cycle)
        inp = ruckig.InputParameter(n)
        out = ruckig.OutputParameter(n)

        inp.current_position     = list(start)
        inp.current_velocity     = [0.0] * n
        inp.current_acceleration = [0.0] * n
        inp.target_position      = list(end)
        inp.target_velocity      = list(self.release_velocity)
        inp.target_acceleration  = [0.0] * n
        inp.max_velocity         = list(self.max_velocity)
        inp.max_acceleration     = list(self.max_acceleration)
        inp.max_jerk             = list(self.max_jerk)

        # Template carries the held-constant joints; throw_idx entries are
        # overwritten at every sample with Ruckig-planned values.
        template = list(self.ready_ordered)
        template[self.base_idx] = base_angle

        traj = JointTrajectory()
        traj.joint_names = list(self.joint_names)

        # Anchor at t=0 with the current state so the controller doesn't
        # snap to the first Ruckig sample.
        zero_pt = JointTrajectoryPoint()
        zero_pt.positions     = list(template)
        zero_pt.velocities    = [0.0] * len(self.joint_names)
        zero_pt.accelerations = [0.0] * len(self.joint_names)
        zero_pt.time_from_start = rospy.Duration.from_sec(0.0)
        traj.points.append(zero_pt)

        t = 0.0
        result = otg.update(inp, out)
        while result == ruckig.Result.Working:
            t += self.control_cycle
            pos = list(template)
            vel = [0.0] * len(self.joint_names)
            acc = [0.0] * len(self.joint_names)
            for k, joint_idx in enumerate(self.throw_idx):
                pos[joint_idx] = float(out.new_position[k])
                vel[joint_idx] = float(out.new_velocity[k])
                acc[joint_idx] = float(out.new_acceleration[k])
            pt = JointTrajectoryPoint()
            pt.positions  = pos
            pt.velocities = vel
            pt.accelerations = acc
            pt.time_from_start = rospy.Duration.from_sec(t)
            traj.points.append(pt)

            inp.current_position     = list(out.new_position)
            inp.current_velocity     = list(out.new_velocity)
            inp.current_acceleration = list(out.new_acceleration)
            result = otg.update(inp, out)

        if result != ruckig.Result.Finished:
            raise RuntimeError(f"ruckig result = {result}")

        # Final point exactly at the launch pose, with the release velocities.
        # Not decelerating to zero is intentional — the controller's hold of
        # the launch pose after this point is the abrupt brake that fires
        # the ball.
        final = JointTrajectoryPoint()
        pos = list(template)
        vel = [0.0] * len(self.joint_names)
        acc = [0.0] * len(self.joint_names)
        for k, joint_idx in enumerate(self.throw_idx):
            pos[joint_idx] = end[k]
            vel[joint_idx] = self.release_velocity[k]
        final.positions = pos
        final.velocities = vel
        final.accelerations = acc
        t += self.control_cycle
        final.time_from_start = rospy.Duration.from_sec(t)
        traj.points.append(final)

        debug = {
            "throw_joints":     list(self.THROW_JOINTS),
            "start":            [round(v, 4) for v in start],
            "end":              [round(v, 4) for v in end],
            "release_velocity": [round(v, 4) for v in self.release_velocity],
            "max_velocity":     list(self.max_velocity),
            "max_acceleration": list(self.max_acceleration),
            "max_jerk":         list(self.max_jerk),
            "control_cycle":    self.control_cycle,
            "n_points":         len(traj.points),
            "duration_s":       round(t, 4),
            "base_angle":       round(base_angle, 4),
        }
        self.pub_throw_debug.publish(String(data=json.dumps(debug)))
        return traj

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        MotionNode().spin()
    except rospy.ROSInterruptException:
        pass
