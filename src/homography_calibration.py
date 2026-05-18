#!/usr/bin/env python3
"""
RoboPong - Homography Calibration Script
-----------------------------------------
Computes the homography that maps overhead-camera pixels to a
workspace-local (x, y) plane whose origin sits at the workspace center.

The output yaml also records the workspace center in the robot 'base'
frame (WORKSPACE_CENTER_IN_BASE below), so vision_node can translate
into base at runtime without needing TF.

Workspace is 80 cm x 60 cm. Origin (0,0) of the calibration grid is the
workspace center. Measure WORKSPACE_CENTER_IN_BASE via TF (e.g.
`rosrun tf tf_echo base table_top`) before running this script.

Usage:
    python3 homography_calibration.py

Instructions:
    - Click the 4 workspace corners in this EXACT order:
        1. Top-Left
        2. Top-Right
        3. Bottom-Right
        4. Bottom-Left
    - Press 'r' to reset and re-click if you make a mistake
    - Press 'q' to quit without saving
    - After 4 points are clicked, press 's' to save
"""

import cv2
import numpy as np
import yaml
import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------

# Camera topic
CAMERA_TOPIC = "/cam/color/image_raw"

# Output path
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../config/homography.yaml"
)

# Real-world coordinates of the 4 workspace corners in METERS, in the
# workspace-local frame (origin at workspace center).
# Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
WORLD_POINTS = np.array([
    [-0.45,  0.355],   # Top-Left
    [ 0.45,  0.355],   # Top-Right
    [ 0.45, -0.355],   # Bottom-Right
    [-0.45, -0.355],   # Bottom-Left
], dtype=np.float32)

# Position of the workspace center in the robot 'base' frame.
# Measure via TF: `rosrun tf tf_echo base table_top` (or equivalent).
# vision_node adds this offset to homography output so cup_position is
# published in base frame.
WORKSPACE_CENTER_IN_BASE = (2.200, 0.000)

# -----------------------------------------------------------------------
# GLOBALS
# -----------------------------------------------------------------------

bridge = CvBridge()
latest_frame = None
clicked_points = []

# -----------------------------------------------------------------------
# CALLBACKS
# -----------------------------------------------------------------------

def image_callback(msg):
    global latest_frame
    try:
        latest_frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except Exception as e:
        rospy.logerr(f"cv_bridge error: {e}")

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([x, y])
            print(f"  Point {len(clicked_points)} clicked: pixel ({x}, {y})")

# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main():
    global clicked_points, latest_frame

    rospy.init_node("homography_calibration", anonymous=True)
    rospy.Subscriber(CAMERA_TOPIC, Image, image_callback)

    print("\n=== RoboPong Homography Calibration ===")
    print("Waiting for camera feed...")

    # Wait for first frame
    rate = rospy.Rate(30)
    while latest_frame is None and not rospy.is_shutdown():
        rate.sleep()

    print("Camera feed received!")
    print("\nINSTRUCTIONS:")
    print("  Click the 4 workspace corners in this order:")
    print("    1. Top-Left corner")
    print("    2. Top-Right corner")
    print("    3. Bottom-Right corner")
    print("    4. Bottom-Left corner")
    print("  Press 'r' to reset points")
    print("  Press 's' to save after 4 points are clicked")
    print("  Press 'q' to quit\n")

    cv2.namedWindow("Homography Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Homography Calibration", 1280, 720)
    cv2.setMouseCallback("Homography Calibration", mouse_callback)

    labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    colors = [(0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 0, 0)]

    while not rospy.is_shutdown():
        if latest_frame is None:
            continue

        display = latest_frame.copy()

        # Draw clicked points
        for i, pt in enumerate(clicked_points):
            cv2.circle(display, tuple(pt), 8, colors[i], -1)
            cv2.circle(display, tuple(pt), 10, (255, 255, 255), 2)
            cv2.putText(display, f"{i+1}. {labels[i]}",
                        (pt[0] + 12, pt[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

        # Draw lines between points if we have more than 1
        if len(clicked_points) > 1:
            for i in range(len(clicked_points) - 1):
                cv2.line(display,
                         tuple(clicked_points[i]),
                         tuple(clicked_points[i+1]),
                         (255, 255, 0), 2)
        if len(clicked_points) == 4:
            cv2.line(display,
                     tuple(clicked_points[3]),
                     tuple(clicked_points[0]),
                     (255, 255, 0), 2)

        # Status text
        if len(clicked_points) < 4:
            next_label = labels[len(clicked_points)]
            status = f"Click point {len(clicked_points)+1}: {next_label}"
            color = colors[len(clicked_points)]
        else:
            status = "All 4 points selected! Press 's' to save, 'r' to reset"
            color = (0, 255, 0)

        cv2.rectangle(display, (0, 0), (900, 35), (0, 0, 0), -1)
        cv2.putText(display, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Homography Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            clicked_points = []
            print("Points reset. Start clicking again.")

        elif key == ord('s') and len(clicked_points) == 4:
            # Compute homography
            pixel_points = np.array(clicked_points, dtype=np.float32)
            H, mask = cv2.findHomography(pixel_points, WORLD_POINTS)

            if H is None:
                print("ERROR: Could not compute homography. Try clicking again.")
                continue

            print("\nHomography matrix computed:")
            print(H)

            # Verify by projecting clicked points back
            print("\nVerification (pixel -> world):")
            for i, pt in enumerate(clicked_points):
                p = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
                result = cv2.perspectiveTransform(p, H)
                rx, ry = result[0][0]
                ex, ey = WORLD_POINTS[i]
                print(f"  {labels[i]}: got ({rx:.3f}, {ry:.3f}), expected ({ex:.3f}, {ey:.3f})")

            # Save to YAML
            output_dir = os.path.dirname(OUTPUT_PATH)
            os.makedirs(output_dir, exist_ok=True)

            data = {
                "homography_matrix": {
                    "rows": 3,
                    "cols": 3,
                    "data": H.flatten().tolist()
                },
                "workspace": {
                    "width_m": 0.90,
                    "height_m": 0.80,
                    "center_x_m": float(WORKSPACE_CENTER_IN_BASE[0]),
                    "center_y_m": float(WORKSPACE_CENTER_IN_BASE[1]),
                }
            }

            with open(OUTPUT_PATH, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)

            print(f"\nSaved to: {OUTPUT_PATH}")
            print("Homography calibration complete!")
            break

        elif key == ord('q'):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()