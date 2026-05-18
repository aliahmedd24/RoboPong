#!/usr/bin/env python3
"""
RoboPong - Sound Effects Node
-----------------------------
Plays per-state sound effects keyed off /robopong/game_status.

Drop WAV files into <robopong>/sounds/ named after each state (lowercase):
    pickup.wav  loading.wav  aiming.wav  throwing.wav  error.wav  idle.wav

Files are optional — missing files mean that state is silent.

LOADING and AIMING loop continuously until the status transitions to
another state. All other states play once.

Parameters (~):
    sounds_dir : directory to look up <state>.wav files
                 (default: <robopong>/sounds)
"""

import os
import threading

import rospkg
import rospy
from std_msgs.msg import String

try:
    import pygame
except ImportError:
    pygame = None


LOOPING_STATES = {'LOADING', 'AIMING'}


class SoundNode:
    def __init__(self):
        rospy.init_node('sound_node')

        default_dir = os.path.join(
            rospkg.RosPack().get_path('robopong'), 'sounds')
        self.sounds_dir = rospy.get_param('~sounds_dir', default_dir)

        self._lock = threading.Lock()
        self._current_state = None
        self._current_channel = None
        self._sounds = {}  # STATE -> pygame.mixer.Sound

        self._audio_ok = self._init_audio()
        if self._audio_ok:
            self._preload_sounds()
        else:
            rospy.logwarn("[sound] Audio unavailable — node will be a no-op.")

        rospy.Subscriber('/robopong/game_status', String,
                         self._status_cb, queue_size=10)
        rospy.loginfo(
            "[sound] Listening on /robopong/game_status (sounds_dir=%s).",
            self.sounds_dir)

    def _init_audio(self):
        if pygame is None:
            rospy.logwarn(
                "[sound] pygame not installed (sudo apt install python3-pygame).")
            return False
        try:
            pygame.mixer.init()
            return True
        except pygame.error as e:
            rospy.logwarn("[sound] pygame.mixer.init failed: %s", e)
            return False

    def _preload_sounds(self):
        if not os.path.isdir(self.sounds_dir):
            rospy.logwarn("[sound] sounds_dir does not exist: %s",
                          self.sounds_dir)
            return
        for fname in sorted(os.listdir(self.sounds_dir)):
            if not fname.lower().endswith('.wav'):
                continue
            state = os.path.splitext(fname)[0].upper()
            path = os.path.join(self.sounds_dir, fname)
            try:
                self._sounds[state] = pygame.mixer.Sound(path)
                rospy.loginfo("[sound] Loaded %s <- %s", state, fname)
            except pygame.error as e:
                rospy.logwarn("[sound] Failed to load %s: %s", fname, e)

    def _status_cb(self, msg):
        # game_status payload is "STATE" or "STATE:detail" — split on ':'.
        state = msg.data.split(':', 1)[0].strip().upper()
        if not state:
            return
        with self._lock:
            if state == self._current_state:
                return
            self._current_state = state
            if self._audio_ok:
                self._play(state)

    def _play(self, state):
        if self._current_channel is not None:
            self._current_channel.stop()
            self._current_channel = None

        sound = self._sounds.get(state)
        if sound is None:
            return
        loops = -1 if state in LOOPING_STATES else 0
        try:
            self._current_channel = sound.play(loops=loops)
        except pygame.error as e:
            rospy.logwarn("[sound] Playback failed for %s: %s", state, e)


if __name__ == '__main__':
    try:
        SoundNode()
    except Exception as e:
        rospy.logfatal("[sound] %s", e)
        raise
    rospy.spin()
