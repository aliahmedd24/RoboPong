#!/usr/bin/env python3
"""
RoboPong - Vision Node
-----------------------
Detects the cup in the overhead camera feed and publishes its
real-world position in the robot base frame.

Subscribes to:
    /cam/color/image_raw        - raw camera feed

Publishes to:
    /robopong/cup_position      - geometry_msgs/PointStamped (x, y, z in robot base frame)
    /robopong/vision_status     - std_msgs/String (status messages)
    /robopong/vision_debug      - sensor_msgs/Image (annotated debug feed)

Parameters:
    cup_color       : "green" (placeholder) or "red" (real cup)
    z_height        : float, table height in robot base frame (default 0.0)
    min_area        : minimum contour area to consider (default 500)
    smoothing_frames: number of frames to average position over (default 5)
"""

import rospy
import cv2
import numpy as np
import yaml
import os
from collections import deque

from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

# -----------------------------------------------------------------------
# HSV COLOR RANGES
# -----------------------------------------------------------------------

# Green cup (placeholder mug)
# Tune these if detection is off under lab lighting
GREEN_LOWER = np.array([35, 40, 40])
GREEN_UPPER = np.array([85, 255, 255])

# Red cup (real cup - red wraps around in HSV so we need two ranges)
RED_LOWER_1 = np.array([0,   100, 100])
RED_UPPER_1 = np.array([10,  255, 255])
RED_LOWER_2 = np.array([160, 100, 100])
RED_UPPER_2 = np.array([180, 255, 255])

# -----------------------------------------------------------------------
# VISION NODE CLASS
# -----------------------------------------------------------------------

class VisionNode:

    def __init__(self):
        rospy.init_node("vision_node", anonymous=False)
        rospy.loginfo("RoboPong Vision Node starting...")

        # --- Parameters ---
        self.cup_color      = rospy.get_param("~cup_color", "green")  # "green" or "red"
        self.z_height       = rospy.get_param("~z_height", 0.0)
        self.min_area       = rospy.get_param("~min_area", 500)
        self.smooth_frames  = rospy.get_param("~smoothing_frames", 5)

        rospy.loginfo(f"Cup color: {self.cup_color}")
        rospy.loginfo(f"Min contour area: {self.min_area}")

        # --- Load calibration files ---
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(pkg_dir, "../config")

        self.K, self.D = self.load_intrinsics(
            os.path.join(config_dir, "camera_intrinsics.yaml"))
        self.H = self.load_homography(
            os.path.join(config_dir, "homography.yaml"))

        # --- CV Bridge ---
        self.bridge = CvBridge()

        # --- Smoothing buffer ---
        self.position_buffer = deque(maxlen=self.smooth_frames)

        # --- Last detection time ---
        self.last_detection_time = rospy.Time.now()

        # --- Publishers ---
        self.pub_position = rospy.Publisher(
            "/robopong/cup_position", PointStamped, queue_size=1)
        self.pub_status = rospy.Publisher(
            "/robopong/vision_status", String, queue_size=1)
        self.pub_debug = rospy.Publisher(
            "/robopong/vision_debug", Image, queue_size=1)

        # --- Subscriber ---
        self.sub = rospy.Subscriber(
            "/cam/color/image_raw", Image, self.image_callback, queue_size=1,
            buff_size=2**24)

        rospy.loginfo("Vision Node ready. Waiting for frames...")

    # -----------------------------------------------------------------------
    # LOAD CALIBRATION
    # -----------------------------------------------------------------------

    def load_intrinsics(self, path):
        rospy.loginfo(f"Loading intrinsics from: {path}")
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        cm = data["camera_matrix"]["data"]
        dc = data["distortion_coefficients"]["data"]
        K = np.array(cm, dtype=np.float64).reshape(3, 3)
        D = np.array(dc, dtype=np.float64)
        rospy.loginfo("Intrinsics loaded OK")
        return K, D

    def load_homography(self, path):
        rospy.loginfo(f"Loading homography from: {path}")
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        H = np.array(data["homography_matrix"]["data"],
                     dtype=np.float64).reshape(3, 3)
        rospy.loginfo("Homography loaded OK")
        return H

    # -----------------------------------------------------------------------
    # IMAGE CALLBACK
    # -----------------------------------------------------------------------

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return

        # Step 1: Undistort
        undistorted = cv2.undistort(frame, self.K, self.D)

        # Step 2: Detect cup
        cup_pixel, debug_frame = self.detect_cup(undistorted)

        # Step 3: If detected, convert to world coords
        if cup_pixel is not None:
            world_x, world_y = self.pixel_to_world(cup_pixel)

            # Add to smoothing buffer
            self.position_buffer.append((world_x, world_y))
            self.last_detection_time = rospy.Time.now()

            # Compute smoothed position
            avg_x = np.mean([p[0] for p in self.position_buffer])
            avg_y = np.mean([p[1] for p in self.position_buffer])

            # Publish position
            pt = PointStamped()
            pt.header.stamp = rospy.Time.now()
            pt.header.frame_id = "base"
            pt.point.x = avg_x
            pt.point.y = avg_y
            pt.point.z = self.z_height
            self.pub_position.publish(pt)

            # Annotate debug frame
            u, v = cup_pixel
            cv2.putText(debug_frame,
                        f"Cup: ({avg_x:.3f}, {avg_y:.3f}) m",
                        (u + 15, v - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            self.pub_status.publish(String(data="DETECTED"))

        else:
            # Check if no detection for more than 2 seconds
            elapsed = (rospy.Time.now() - self.last_detection_time).to_sec()
            if elapsed > 2.0:
                warning = f"WARNING: No cup detected for {elapsed:.1f}s"
                self.pub_status.publish(String(data=warning))
                rospy.logwarn_throttle(5, warning)

        # Step 4: Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding="rgb8")
            self.pub_debug.publish(debug_msg)
        except Exception as e:
            rospy.logerr(f"Debug publish error: {e}")

    # -----------------------------------------------------------------------
    # CUP DETECTION
    # -----------------------------------------------------------------------

    def detect_cup(self, frame):
        """
        Detect the cup using HSV color segmentation.
        Returns (pixel_x, pixel_y) of cup center, or None if not found.
        Also returns annotated debug frame.
        """
        debug = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Build color mask based on cup_color parameter
        if self.cup_color == "green":
            mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

        elif self.cup_color == "red":
            # Red wraps around in HSV, need two ranges
            mask1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
            mask2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
            mask = cv2.bitwise_or(mask1, mask2)

        else:
            rospy.logerr(f"Unknown cup_color: {self.cup_color}. Use 'green' or 'red'.")
            return None, debug

        # Clean up mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            cv2.putText(debug, "No cup detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return None, debug

        # Filter by area and circularity, pick the best candidate
        best_contour = None
        best_score = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            # Circularity score: 1.0 = perfect circle
            circularity = (4 * np.pi * area) / (perimeter ** 2)

            # Score = area * circularity (prefer large, round contours)
            score = area * circularity
            if score > best_score:
                best_score = score
                best_contour = cnt

        if best_contour is None:
            cv2.putText(debug, "No valid cup contour", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return None, debug

        # Get center using moments
        M = cv2.moments(best_contour)
        if M["m00"] == 0:
            return None, debug

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Draw detection on debug frame
        cv2.drawContours(debug, [best_contour], -1, (0, 255, 0), 2)
        cv2.circle(debug, (cx, cy), 6, (0, 0, 255), -1)
        cv2.circle(debug, (cx, cy), 8, (255, 255, 255), 2)

        # Draw workspace boundary
        self.draw_workspace(debug)

        return (cx, cy), debug

    # -----------------------------------------------------------------------
    # PIXEL TO WORLD
    # -----------------------------------------------------------------------

    def pixel_to_world(self, pixel):
        """Convert pixel (u, v) to world (x, y) using homography matrix."""
        pt = np.array([[[float(pixel[0]), float(pixel[1])]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self.H)
        return float(result[0][0][0]), float(result[0][0][1])

    # -----------------------------------------------------------------------
    # DRAW WORKSPACE BOUNDARY
    # -----------------------------------------------------------------------

    def draw_workspace(self, frame):
        """Project workspace corners onto the image for visualization."""
        # World corners of workspace
        world_corners = np.array([
            [-0.40, -0.30],
            [ 0.40, -0.30],
            [ 0.40,  0.30],
            [-0.40,  0.30]
        ], dtype=np.float32)

        # Invert homography to go world -> pixel
        H_inv = np.linalg.inv(self.H)
        corners_3d = np.array(
            [[[c[0], c[1]]] for c in world_corners], dtype=np.float32)
        pixel_corners = cv2.perspectiveTransform(corners_3d, H_inv)

        pts = pixel_corners.reshape(-1, 2).astype(np.int32)
        cv2.polylines(frame, [pts], isClosed=True,
                      color=(255, 165, 0), thickness=2)

    # -----------------------------------------------------------------------
    # RUN
    # -----------------------------------------------------------------------

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            rate.sleep()


# -----------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------

if __name__ == "__main__":
    try:
        node = VisionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass