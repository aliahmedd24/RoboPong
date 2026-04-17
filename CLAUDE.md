# RoboPong — Claude Code Context

UR5e + overhead camera + custom scoop end-effector. Detect a cup on a table,
throw a ball so it lands inside. University lab project.

## Package
- Name: `group5robopong`
- ROS: Noetic (Ubuntu 24)
- Catkin workspace: `~/catkin_ws` (confirm on lab machine)
- Build: `cd ~/catkin_ws && catkin_make --pkg group5robopong`

## Golden Rule
The UR5e stack (bringup, MoveIt, camera, RViz) is managed by the lab admin.
- **Never modify `cu_ur5e_bringup`, `cu_ur5e_description`, or any lab package.**
- **Never replicate what `hardware.launch` already does** in `launch/robopong.launch`.
- Never write risky robot motions (fast trajectories, untested joint targets)
  without explicit user approval and lab access. Scaffolds with placeholder
  values only.

## Existing Lab Entry Point
```bash
roslaunch cu_ur5e_bringup hardware.launch
# brings up: robot driver, MoveIt, /move_group, RViz, camera
```
`scripts2/launch_hardware.sh` wraps this. `scripts2/` also has arm-pose /
controller-switching helpers — read before writing new ones.

## Launch File Rule
`launch/robopong.launch` must ONLY:
1. `<include>` the hardware launch
2. Add `<node>` tags for group5robopong scripts
Nothing else — no RViz, no state publishers, no robot_description reload.

## Coordinate Frames
- Nominal base frame: `base` (verify — may be `base_link`)
- Workspace: 80cm × 60cm
- UR5 base: 1.55m from workspace center
- Table Z: ROS param `~z_height` (default 0.0, measure in lab)

**Frame ambiguity to resolve in lab:** vision publishes `/robopong/cup_position`
with `frame_id = ~base_frame` (default `"base"`). If that frame's origin is the
UR5 base, cup coordinates are already robot-base-relative. If the origin is the
workspace center (which is what the homography calibration implies), the UR5 is
at `(+1.55, 0)` in that frame and `throw_profile.yaml`'s `aim.base_offset_x`
must be set to `1.55`. Confirm by comparing `/tf` output to vision output.

## Key Topics / Services
| Name | Type | Owner |
|------|------|-------|
| `<camera_topic>` (default `/cam/color/image_raw`) | `sensor_msgs/Image` | lab |
| `/move_group` | MoveIt action | lab |
| `/moveit_by_name` | `moveit_by_name/Command` | lab (named-pose shortcut) |
| `/robopong/cup_position` | `geometry_msgs/PointStamped` | vision_node |
| `/robopong/vision_debug` | `sensor_msgs/Image` | vision_node |
| `/robopong/vision_status` | `std_msgs/String` | vision_node |
| `/robopong/go_ready` | `std_srvs/Trigger` | motion_node |
| `/robopong/throw` | `std_srvs/Trigger` | motion_node |

## Named Poses (via `/moveit_by_name`)
`home`, `ready`, `center_above_table`, `present gripper`.
Phase-3 motion_node defines its **own** ready pose in `throw_profile.yaml`
(`ready_joints`) because the pre-throw pose depends on the chosen throw style
(single-joint vs multi-joint) — do not replace with `/moveit_by_name` call.

## Build Phases
- **Phase 1** ✅ `launch/robopong.launch` minimal (include + node tags only)
- **Phase 2** ✅ `vision_node.py` robustness pass + wired into launch
- **Phase 3** 🟡 `motion_node.py` + `throw_profile.yaml` scaffold — **launch entry is commented out until YAML is lab-tuned**
- **Phase 4** ⬜ `game_node.py` — IDLE → DETECTING → AIMING → THROWING → RESET

## Decisions already made (do not undo)
- **vision_node decodes `bgr8`** (not `rgb8`). The HSV cvtColor call is `BGR2HSV`;
  if the decode ever changes to RGB again, color detection silently breaks.
- **vision_node does NOT `cv2.undistort` before homography.** Reason: the
  homography in `config/homography.yaml` was calibrated against raw distorted
  pixels. Re-adding `undistort` without re-running calibration introduces
  cm-scale systematic error. Option (b) — recalibrate on undistorted frames —
  is a valid future path but requires lab time.
- **Throw waypoints are a sparse dict** keyed by joint name (not a positional
  list). motion_node resolves to ordered joint order at load time via
  `MoveGroupCommander.get_active_joints()`. This is deliberate: silent joint
  misalignment on a 30-kg arm is worse than a clear load-time error.
- **Throw profile placeholder guard**: while every waypoint equals `ready_joints`,
  `/robopong/throw` refuses. Override for dry runs with
  `~allow_placeholder:=true`.

## `throw_profile.yaml` schema (quick ref)
```yaml
ready_joints:          # full dict, every active joint present
  <joint_name>: <rad>
waypoints:             # list; first must have time_from_start: 0.0
  - time_from_start: <sec>
    joints: {<joint_name>: <rad>, ...}   # sparse — omitted joints = ready value
aim:
  base_joint_name: <name>       # usually shoulder_pan_joint
  base_offset_x: <m>            # UR5 base position in cup_position frame
  base_offset_y: <m>
  aim_trim: <rad>
safety:
  base_joint_min: <rad>
  base_joint_max: <rad>
```
Aim model: `base_angle = atan2(cup_y - offset_y, cup_x - offset_x) + aim_trim`.
All waypoints get shifted by the aim delta on the base joint at runtime.

## When at the lab: pre-flight checklist

### 1. Info to gather (paste outputs back into a session)
```bash
# camera
rostopic list | grep -iE 'cam|image'
rostopic info <camera_topic>
rostopic hz <camera_topic>
rostopic echo -n 1 <camera_topic_root>/camera_info

# TF
rosrun tf2_tools view_frames.py     # produces frames.pdf
rostopic echo -n 1 /tf_static | head -300

# MoveIt groups / joints / named targets
python3 - <<'EOF'
import rospy, moveit_commander, sys
rospy.init_node("probe", anonymous=True)
moveit_commander.roscpp_initialize(sys.argv)
r = moveit_commander.RobotCommander()
print("Groups:", r.get_group_names())
for g in r.get_group_names():
    mg = moveit_commander.MoveGroupCommander(g)
    print(f"--- {g} ---")
    print("  active_joints:", mg.get_active_joints())
    print("  planning_frame:", mg.get_planning_frame())
    print("  end_effector_link:", mg.get_end_effector_link())
    print("  named_targets:", mg.get_named_targets())
EOF

# controllers
rosservice call /controller_manager/list_controllers

# moveit_by_name
rosservice type /moveit_by_name
rossrv show moveit_by_name/Command

# hardware.launch internals
cat $(rospack find cu_ur5e_bringup)/launch/hardware.launch

# URDF (for scoop frame + geometry)
rosparam get /robot_description > /tmp/robot_description.urdf
```

### 2. Physical measurements
- [ ] Table surface Z in `base` frame (for vision_node `~z_height`)
- [ ] UR5 base position in vision's `frame_id` (for `aim.base_offset_*`)
- [ ] Scoop tip offset from last UR5 link (if not already in URDF)
- [ ] Ball: diameter, mass
- [ ] Cup: inner diameter, height

### 3. Vision sanity
```bash
# with hardware.launch + vision_node running
rostopic hz /robopong/cup_position
rosrun rqt_image_view rqt_image_view /robopong/vision_debug
rostopic echo /robopong/vision_status
```
If HSV ranges miss the cup under lab lighting, tune `GREEN_*` / `RED_*`
constants in `src/vision_node.py`. Try the red cup (real) via
`<arg name="cup_color" default="red"/>` when launching.

### 4. Motion dry-run (placeholder profile, safe)
```bash
# leave throw_profile.yaml placeholders, start motion_node directly:
rosrun group5robopong motion_node.py _allow_placeholder:=true
# then:
rosservice call /robopong/go_ready
rosservice call /robopong/throw   # moves to aimed ready, then no-op throw
```
Confirm the node reaches the ready pose safely before tuning waypoints.

### 5. Throw tuning (incremental, in this order)
1. Fix joint names in YAML if the active joint list has prefixes.
2. Pick a conservative `ready_joints`.
3. Set `aim.base_offset_*` based on resolved frame ambiguity.
4. Add one throw waypoint, slow (`time_from_start ≥ 1.0s`), small angle delta.
5. `~velocity_scaling` / `~acceleration_scaling` start at 0.1, ramp up.
6. Tighten `safety.base_joint_min/max` once cup region is characterised.
7. Flip `~allow_placeholder` back to `false`, uncomment the motion block in
   `launch/robopong.launch`.

## File map
- `launch/robopong.launch` — includes `hardware.launch`, declares group5robopong nodes
- `src/vision_node.py` — cup detection; publishes `/robopong/cup_position`
- `src/motion_node.py` — MoveIt-based go_ready / throw services
- `src/homography_calibration.py` — one-time pixel→world calibration utility
- `config/camera_intrinsics.yaml` — camera K/D (loaded but currently unused in vision_node; see "Decisions")
- `config/homography.yaml` — pixel→world homography + workspace dims
- `config/throw_profile.yaml` — throw parameters (placeholder; tune in lab)
- `robots/basic_setup.urdf.xacro` — scene URDF (tables, wall, tripod)
- `scripts2/` — lab admin's shell helpers for poses / controller switching
