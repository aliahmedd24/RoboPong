#!/bin/sh

if rosrun controller_manager controller_manager list | grep -q 'scaled_pos_joint_traj_controller.*( running )'; then
  rosrun controller_manager controller_manager stop scaled_pos_joint_traj_controller 2>/dev/null
fi
rosrun controller_manager controller_manager start twist_controller
