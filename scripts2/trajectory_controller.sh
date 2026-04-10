#!/bin/sh

if rosrun controller_manager controller_manager list | grep -q 'twist_controller.*( running )'; then
  rosrun controller_manager controller_manager stop twist_controller 2>/dev/null
fi
rosrun controller_manager controller_manager start scaled_pos_joint_traj_controller
