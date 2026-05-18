#!/usr/bin/env python3
"""
RoboPong - Game Orchestrator Node
---------------------------------
Sequences the existing motion services to play one round:
    /robopong/pick_up  -> (wait ball_load_delay) -> wait for fresh cup -> /robopong/throw

Exposes:
    /robopong/play_round  - std_srvs/Trigger, runs one full cycle.
    /robopong/start_game  - std_srvs/Trigger, loops play_round with
                            inter_round_delay between rounds.
    /robopong/stop_game   - std_srvs/Trigger, halts the loop after the current
                            round (or during the inter-round sleep).

Subscribes to:
    /robopong/cup_position - geometry_msgs/PointStamped (freshness check only).

Publishes (latched):
    /robopong/game_status  - std_msgs/String, state transitions for ops visibility.

Parameters (~):
    ball_load_delay     : seconds to dwell at pickup pose for a human to drop in
                          a ball (default 3.0).
    cup_wait_timeout    : max seconds to wait for a fresh cup before aborting the
                          round (default 10.0).
    cup_fresh_threshold : a cup_position is "fresh" if its stamp is within this
                          many seconds (default 2.0). Should match motion_node's
                          CUP_STALE_S.
    inter_round_delay   : seconds between rounds in continuous mode (default 10.0).
    service_timeout     : seconds to wait for motion services to appear on
                          startup (default 30.0).
"""

import threading

import rospy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse


PICK_UP_SRV  = '/robopong/pick_up'
GO_READY_SRV = '/robopong/go_ready'
THROW_SRV    = '/robopong/throw'


class GameNode:
    def __init__(self):
        rospy.init_node('game_node')

        self.ball_load_delay     = float(rospy.get_param('~ball_load_delay', 5.0))
        self.cup_wait_timeout    = float(rospy.get_param('~cup_wait_timeout', 10.0))
        self.cup_fresh_threshold = float(rospy.get_param('~cup_fresh_threshold', 2.0))
        self.inter_round_delay   = float(rospy.get_param('~inter_round_delay', 12.0))
        service_timeout          = float(rospy.get_param('~service_timeout', 30.0))

        self._cup_stamp = None
        self._cup_lock = threading.Lock()
        # Serialize play_round calls — concurrent rounds would race the arm.
        self._round_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._loop_thread = None

        self.status_pub = rospy.Publisher(
            '/robopong/game_status', String, queue_size=1, latch=True)
        self._publish_status('IDLE')

        rospy.Subscriber('/robopong/cup_position', PointStamped,
                         self._cup_cb, queue_size=1)

        rospy.loginfo("[game] Waiting for motion services...")
        try:
            rospy.wait_for_service(PICK_UP_SRV,  timeout=service_timeout)
            rospy.wait_for_service(GO_READY_SRV, timeout=service_timeout)
            rospy.wait_for_service(THROW_SRV,    timeout=service_timeout)
        except rospy.ROSException as e:
            rospy.logfatal("[game] Motion services not available: %s", e)
            raise

        self.pick_up_srv  = rospy.ServiceProxy(PICK_UP_SRV,  Trigger)
        self.go_ready_srv = rospy.ServiceProxy(GO_READY_SRV, Trigger)
        self.throw_srv    = rospy.ServiceProxy(THROW_SRV,    Trigger)

        rospy.Service('/robopong/play_round', Trigger, self._play_round)
        rospy.Service('/robopong/start_game', Trigger, self._start_game)
        rospy.Service('/robopong/stop_game',  Trigger, self._stop_game)
        rospy.loginfo(
            "[game] Ready. /robopong/play_round (one shot) or "
            "/robopong/start_game + /robopong/stop_game (continuous).")

    def _cup_cb(self, msg):
        with self._cup_lock:
            self._cup_stamp = msg.header.stamp

    def _publish_status(self, state, detail=None):
        text = state if detail is None else f"{state}:{detail}"
        self.status_pub.publish(String(data=text))
        rospy.loginfo("[game] status -> %s", text)

    def _cup_is_fresh(self):
        with self._cup_lock:
            stamp = self._cup_stamp
        if stamp is None:
            return False
        return (rospy.Time.now() - stamp).to_sec() <= self.cup_fresh_threshold

    def _wait_for_fresh_cup(self):
        deadline = rospy.Time.now() + rospy.Duration(self.cup_wait_timeout)
        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self._cup_is_fresh():
                return True
            rate.sleep()
        return False

    def _call(self, proxy, label):
        try:
            resp = proxy()
        except rospy.ServiceException as e:
            return False, f"{label} service exception: {e}"
        if not resp.success:
            return False, f"{label} failed: {resp.message}"
        return True, resp.message

    def _play_round(self, _req):
        if not self._round_lock.acquire(blocking=False):
            return TriggerResponse(success=False,
                                   message="round already in progress")
        try:
            # 1) Pickup pose
            self._publish_status('PICKUP')
            ok, msg = self._call(self.pick_up_srv, 'pick_up')
            if not ok:
                self._publish_status('ERROR', msg)
                return TriggerResponse(success=False, message=msg)

            # 2) Dwell for ball loading
            self._publish_status('LOADING', f"{self.ball_load_delay:.1f}s")
            rospy.sleep(self.ball_load_delay)

            # 3) Wait for a fresh cup detection before committing to throw
            self._publish_status('AIMING')
            if not self._wait_for_fresh_cup():
                detail = f"no fresh cup within {self.cup_wait_timeout:.1f}s"
                self._publish_status('ERROR', detail)
                return TriggerResponse(success=False, message=detail)

            # 4) Throw — motion_node will re-read cup_position and aim
            self._publish_status('THROWING')
            ok, msg = self._call(self.throw_srv, 'throw')
            if not ok:
                self._publish_status('ERROR', msg)
                return TriggerResponse(success=False, message=msg)

            self._publish_status('IDLE')
            return TriggerResponse(success=True, message="round complete")
        finally:
            self._round_lock.release()

    def _start_game(self, _req):
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return TriggerResponse(success=False, message="game already running")
        self._stop_event.clear()
        self._loop_thread = threading.Thread(target=self._game_loop, daemon=True)
        self._loop_thread.start()
        return TriggerResponse(success=True, message="game started")

    def _stop_game(self, _req):
        if self._loop_thread is None or not self._loop_thread.is_alive():
            return TriggerResponse(success=False, message="game not running")
        self._stop_event.set()
        return TriggerResponse(success=True, message="stop requested")

    def _game_loop(self):
        rospy.loginfo("[game] Continuous loop started (inter_round_delay=%.1fs).",
                      self.inter_round_delay)
        while not self._stop_event.is_set() and not rospy.is_shutdown():
            resp = self._play_round(None)
            if not resp.success:
                rospy.logwarn("[game] Round failed, stopping loop: %s",
                              resp.message)
                break
            if self._stop_event.is_set():
                break
            # Park at the pickup pose so the inter-round wait happens there
            # rather than wherever the throw trajectory ended.
            self._publish_status('PICKUP')
            ok, msg = self._call(self.pick_up_srv, 'pick_up')
            if not ok:
                rospy.logwarn("[game] Inter-round pickup failed: %s", msg)
                self._publish_status('ERROR', msg)
                break
            # Sleep between rounds, but stay responsive to stop requests.
            deadline = rospy.Time.now() + rospy.Duration(self.inter_round_delay)
            rate = rospy.Rate(10.0)
            while (not self._stop_event.is_set()
                   and not rospy.is_shutdown()
                   and rospy.Time.now() < deadline):
                rate.sleep()
        self._publish_status('IDLE')
        rospy.loginfo("[game] Continuous loop stopped.")


if __name__ == '__main__':
    try:
        GameNode()
    except Exception as e:
        rospy.logfatal("[game] %s", e)
        raise
    rospy.spin()
