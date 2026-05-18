#!/usr/bin/env python3
"""Commentator — turns /robopong/game_status into Piper TTS lines on /say.

State transitions are pure triggers: each new state enqueues a phrase, and
a worker thread drains the queue, holding the floor for the phrase's
duration before publishing the next one. Fast state transitions (e.g.
AIMING completing in milliseconds when the cup is already in range) no
longer preempt earlier phrases — they queue behind them.
"""

import queue
import random
import threading

import rospy
from std_msgs.msg import String


# State -> list of phrases. Pick one at random per transition.
# Leave a list empty (or omit the key) to stay silent on that state.
PHRASES = {
    'PICKUP':   ["Picking up.", "Grabbing position.", "Load me up."],
    'LOADING':  ["Drop the ball.", "Waiting on the ball.", "Feed me."],
    'AIMING':   ["Locking on.", "Finding the cup.", "Targeting."],
    'THROWING': ["Here it goes!", "Launching!", "Fire!"],
    'IDLE':     ["Standing by.", "Ready when you are.", "Take a breather."],
    'ERROR':    ["Something went wrong.", "I had a problem."],
}

# Seconds to wait after publishing a phrase before the next one is sent.
# Tune to roughly the spoken length of the longest phrase in each state's
# pool. Piper-spoken short lines typically run ~0.7–1.5 s.
DEFAULT_DURATION = 1.5
DURATION = {
    'PICKUP':   1.5,
    'LOADING':  1.8,
    'AIMING':   1.5,
    'THROWING': 1.5,
    'IDLE':     2.0,
    'ERROR':    2.0,
}

# Cap on queued (state, phrase) tuples. If the producer outruns the worker,
# drop the OLDEST entry — newer state is more relevant than backlog.
MAX_QUEUE = 5


class Commentator:
    def __init__(self):
        rospy.init_node('commentator_node')
        self.say_pub = rospy.Publisher('/say', String, queue_size=4)
        self._last_state = None
        self._queue = queue.Queue(maxsize=MAX_QUEUE)
        self._sub = rospy.Subscriber('/robopong/game_status', String,
                                     self._on_status, queue_size=4)
        self._worker = threading.Thread(target=self._drain, daemon=True)
        self._worker.start()
        rospy.loginfo("[commentator] Ready.")

    def _on_status(self, msg):
        # game_status is "STATE" or "STATE:detail" — only the state drives lines.
        state = msg.data.split(':', 1)[0]
        if state == self._last_state:
            return
        self._last_state = state

        pool = PHRASES.get(state)
        if not pool:
            return
        item = (state, random.choice(pool))
        # Drop oldest if backlog is at the cap.
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    dropped = self._queue.get_nowait()
                    rospy.logwarn("[commentator] queue full, dropped %s",
                                  dropped[0])
                except queue.Empty:
                    pass

    def _drain(self):
        while not rospy.is_shutdown():
            try:
                state, phrase = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if rospy.is_shutdown():
                return
            self.say_pub.publish(String(data=phrase))
            rospy.sleep(DURATION.get(state, DEFAULT_DURATION))


if __name__ == '__main__':
    Commentator()
    rospy.spin()
