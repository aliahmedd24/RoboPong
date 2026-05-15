#!/usr/bin/env python3
import csv
import math
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rospy
import moveit_commander
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import PointStamped
from std_srvs.srv import Trigger, TriggerResponse
from control_msgs.msg import JointTrajectoryControllerState
from ruckig import Ruckig, InputParameter, OutputParameter, Result

# ════════════════════════════════════════════════════════════════════════
#  RoboPong — UR5e throw node
#
#  MoveIt plans the slow wind-up to ready (collision-aware).
#  Ruckig generates the A→B→C throw at 1ms with vel/accel explicit.
#  move_group.execute() forwards the trajectory unchanged.
#
#  Aim: shoulder_pan += atan2(cup_y, cup_x), applied to every waypoint.
#  All knobs are module-level constants below — no rosparams.
#
#  Every throw attempt dumps CSV + PNG (planned ± executed) under OUTPUT_DIR.
# ════════════════════════════════════════════════════════════════════════

DOF = 6
MOVE_GROUP = 'arm'

JOINT_NAMES = [
    'arm_shoulder_pan_joint',
    'arm_shoulder_lift_joint',
    'arm_elbow_joint',
    'arm_wrist_1_joint',
    'arm_wrist_2_joint',
    'arm_wrist_3_joint',
]

LIFT_IDX, ELBOW_IDX = 1, 2   # indices into pose vectors

MAX_VEL   = [3.14, 3.14, 3.14, 0.5,  0.5,  0.5 ]   # rad/s
MAX_ACCEL = [8.0,  8.0,  10.0, 2.0,  2.0,  2.0 ]   # rad/s²
MAX_JERK  = [25.0, 25.0, 40.0, 10.0, 10.0, 10.0]   # rad/s³

READY_POS   = [0.0,  0.000, -2.094,  0.0,  0.140,  0.0]   # A — cocked
RELEASE_POS = [0.0,  0.524, -0.349,  0.0,  0.559,  0.0]   # B — release
DECEL_POS   = [0.0,  0.698,  0.000,  0.0, -0.140,  0.0]   # C — follow-through

OMEGA = 1.5   # rad/s — release velocity, both active joints
RELEASE_VEL = [0.0, OMEGA, OMEGA, 0.0, 0.0, 0.0]

CUP_STALE_S    = 2.0
MOVE_VEL_SCALE = 0.3
MOVE_ACC_SCALE = 0.3

# Base aim offset (rad) — tune in lab so shoulder_pan=0 + AIM_OFFSET points
# at base +X. cup-angle is added on top per throw.
AIM_OFFSET = 0.0

OUTPUT_DIR             = os.path.expanduser('~/robopong_throws')
CONTROLLER_STATE_TOPIC = '/scaled_pos_joint_traj_controller/state'


def aim_shift(pose, aim):
    out = list(pose)
    out[0] += aim
    return out


def ruckig_segment(p0, v0, a0, p1, v1, a1):
    inp = InputParameter(DOF)
    inp.current_position, inp.current_velocity, inp.current_acceleration = p0, v0, a0
    inp.target_position,  inp.target_velocity,  inp.target_acceleration  = p1, v1, a1
    inp.max_velocity, inp.max_acceleration, inp.max_jerk = MAX_VEL, MAX_ACCEL, MAX_JERK

    out = OutputParameter(DOF)
    traj = []
    otg = Ruckig(DOF, 0.001)
    r = otg.update(inp, out)
    while r == Result.Working:
        traj.append((out.new_position[:], out.new_velocity[:], out.new_acceleration[:]))
        out.pass_to_input(inp)
        r = otg.update(inp, out)
    if r == Result.Finished:
        traj.append((out.new_position[:], out.new_velocity[:], out.new_acceleration[:]))
    return traj


def build_throw_trajectory(aim):
    ready   = aim_shift(READY_POS,   aim)
    release = aim_shift(RELEASE_POS, aim)
    decel   = aim_shift(DECEL_POS,   aim)
    return (ruckig_segment(ready,   [0.0]*DOF,   [0.0]*DOF,
                           release, RELEASE_VEL, [0.0]*DOF)
            + ruckig_segment(release, RELEASE_VEL, [0.0]*DOF,
                             decel,   [0.0]*DOF,   [0.0]*DOF))


def send_trajectory(move_group, full_traj):
    rt = RobotTrajectory()
    rt.joint_trajectory.joint_names = JOINT_NAMES
    rt.joint_trajectory.header.stamp = rospy.Time.now()
    for i, (p, v, a) in enumerate(full_traj):
        pt = JointTrajectoryPoint()
        pt.positions, pt.velocities, pt.accelerations = p, v, a
        pt.time_from_start = rospy.Duration((i + 1) * 0.001)   # avoid t=0
        rt.joint_trajectory.points.append(pt)
    move_group.execute(rt, wait=True)


def move_to_ready(move_group, aim=0.0):
    rospy.loginfo("[motion] Moving to ready (aim=%.3f) via MoveIt...", aim)
    move_group.set_joint_value_target(aim_shift(READY_POS, aim))
    move_group.set_max_velocity_scaling_factor(MOVE_VEL_SCALE)
    move_group.set_max_acceleration_scaling_factor(MOVE_ACC_SCALE)
    ok = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    if not ok:
        rospy.logerr("[motion] MoveIt failed to reach ready pose.")
    return ok


# ════════════════════════════════════════════════════════════════════════
#  RECORDER — captures executed trajectory from the FJT controller state
# ════════════════════════════════════════════════════════════════════════

class ControllerRecorder:
    """Buffers (t, lift_pos, lift_vel, elbow_pos, elbow_vel) between
    start() and stop(), reading from the FJT controller state topic."""

    def __init__(self):
        self.samples = []
        self._sub = None
        self._t0 = None
        self._lift_i = None
        self._elbow_i = None

    def start(self):
        self.samples = []
        self._t0 = rospy.Time.now()
        self._lift_i = None
        self._elbow_i = None
        self._sub = rospy.Subscriber(CONTROLLER_STATE_TOPIC,
                                     JointTrajectoryControllerState,
                                     self._cb, queue_size=500)

    def stop(self):
        if self._sub is not None:
            self._sub.unregister()
            self._sub = None

    def _cb(self, msg):
        if self._lift_i is None:
            try:
                self._lift_i  = list(msg.joint_names).index('arm_shoulder_lift_joint')
                self._elbow_i = list(msg.joint_names).index('arm_elbow_joint')
            except ValueError:
                return
        t = (rospy.Time.now() - self._t0).to_sec()
        self.samples.append((
            t,
            msg.actual.positions[self._lift_i],
            msg.actual.velocities[self._lift_i],
            msg.actual.positions[self._elbow_i],
            msg.actual.velocities[self._elbow_i],
        ))


# ════════════════════════════════════════════════════════════════════════
#  CSV + PLOT — dumped every throw attempt that got far enough to plan
# ════════════════════════════════════════════════════════════════════════

def _planned_rows(planned):
    """(t, lift_p, lift_v, lift_a, el_p, el_v, el_a) per planned tick."""
    rows = []
    for i, (p, v, a) in enumerate(planned):
        rows.append(((i + 1) * 0.001,
                     p[LIFT_IDX],  v[LIFT_IDX],  a[LIFT_IDX],
                     p[ELBOW_IDX], v[ELBOW_IDX], a[ELBOW_IDX]))
    return rows


def _executed_rows(samples):
    """Differentiate velocity to get acceleration. Empty if no samples."""
    rows = []
    for i, s in enumerate(samples):
        t, lp, lv, ep, ev = s
        if i == 0:
            la = ea = 0.0
        else:
            pt, _, plv, _, pev = samples[i - 1]
            dt = t - pt
            la = (lv - plv) / dt if dt > 0 else 0.0
            ea = (ev - pev) / dt if dt > 0 else 0.0
        rows.append((t, lp, lv, la, ep, ev, ea))
    return rows


def dump_and_plot(planned, executed_samples, label):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = os.path.join(OUTPUT_DIR, f"throw_{ts}_{label}")

    p_rows = _planned_rows(planned)
    e_rows = _executed_rows(executed_samples) if executed_samples else []

    with open(stem + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'source',
                    'shoulder_lift_pos', 'shoulder_lift_vel', 'shoulder_lift_accel',
                    'elbow_pos', 'elbow_vel', 'elbow_accel'])
        for r in p_rows:
            w.writerow([r[0], 'planned',  *r[1:]])
        for r in e_rows:
            w.writerow([r[0], 'executed', *r[1:]])

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    col_titles = ['shoulder_lift', 'elbow']
    row_labels = ['position (rad)', 'velocity (rad/s)', 'accel (rad/s²)']
    # in each row tuple: lift fields at cols 1,2,3 ; elbow at 4,5,6
    col_offsets = [1, 4]

    pt = [r[0] for r in p_rows]
    et = [r[0] for r in e_rows]

    for col, off in enumerate(col_offsets):
        for row in range(3):
            ax = axes[row][col]
            ax.plot(pt, [r[off + row] for r in p_rows],
                    label='planned', linewidth=1.5)
            if e_rows:
                ax.plot(et, [r[off + row] for r in e_rows],
                        label='executed', linewidth=1.0, alpha=0.8)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best')
            if row == 0:
                ax.set_title(col_titles[col])
            if col == 0:
                ax.set_ylabel(row_labels[row])
            if row == 2:
                ax.set_xlabel('t (s)')

    fig.suptitle(f"Throw {ts} — {label}")
    fig.tight_layout()
    fig.savefig(stem + '.png', dpi=100)
    plt.close(fig)

    rospy.loginfo("[motion] Dumped %s.{csv,png}", stem)


# ════════════════════════════════════════════════════════════════════════
#  NODE
# ════════════════════════════════════════════════════════════════════════

class MotionNode:
    def __init__(self):
        rospy.init_node('motion_node')
        moveit_commander.roscpp_initialize(sys.argv)
        self.move_group = moveit_commander.MoveGroupCommander(MOVE_GROUP)

        active = list(self.move_group.get_active_joints())
        if active != JOINT_NAMES:
            raise RuntimeError(
                f"Joint mismatch.\n  MoveIt: {active}\n  Node:   {JOINT_NAMES}")

        self.cup = None
        self.recorder = ControllerRecorder()

        rospy.Subscriber('/robopong/cup_position', PointStamped,
                         self._cup_cb, queue_size=1)
        rospy.Service('/robopong/go_ready', Trigger, self._go_ready)
        rospy.Service('/robopong/throw',    Trigger, self._throw)
        rospy.loginfo("[motion] Ready. Dumping plots to %s", OUTPUT_DIR)

    def _cup_cb(self, msg):
        self.cup = (msg.point.x, msg.point.y, msg.header.stamp)

    def _go_ready(self, _req):
        ok = move_to_ready(self.move_group, 0.0)
        return TriggerResponse(success=ok,
                               message="ready" if ok else "MoveIt failed")

    def _throw(self, _req):
        cup = self.cup
        if cup is None:
            return TriggerResponse(success=False, message="no cup")
        cx, cy, stamp = cup
        age = (rospy.Time.now() - stamp).to_sec()
        if age > CUP_STALE_S:
            return TriggerResponse(success=False,
                                   message=f"stale cup ({age:.2f}s)")

        aim = AIM_OFFSET
        aim += math.atan2(cy, cx)
        rospy.loginfo("[motion] Throw aim=%.3f at cup=(%.3f, %.3f)", aim, cx, cy)

        # Build the plan first so we still have something to plot if ready fails.
        planned = build_throw_trajectory(aim)

        if not move_to_ready(self.move_group, aim):
            dump_and_plot(planned, executed_samples=None, label="ready_failed")
            return TriggerResponse(success=False, message="ready failed")

        self.recorder.start()
        try:
            rospy.loginfo("[motion] Throw — %d points (%d ms), OMEGA=%.2f, aim=%.3f",
                          len(planned), len(planned), OMEGA, aim)
            send_trajectory(self.move_group, planned)
            rospy.loginfo("[motion] Throw complete.")
        finally:
            self.recorder.stop()

        dump_and_plot(planned, executed_samples=self.recorder.samples, label="ok")
        return TriggerResponse(success=True, message=f"threw at aim={aim:.3f}")


if __name__ == '__main__':
    try:
        MotionNode()
    except RuntimeError as e:
        rospy.logfatal("[motion] %s", e)
        sys.exit(1)
    rospy.spin()
