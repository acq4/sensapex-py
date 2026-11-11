import threading
from unittest import TestCase

import numpy as np

from sensapex.sensapex import (
    LIBUM_INVALID_DEV,
    LIBUM_POS_DRIVE_BUSY,
    LIBUM_POS_DRIVE_COMPLETED,
    LIBUM_POS_DRIVE_FAILED,
    UMP,
    MoveRequest,
    UMError,
    timer,
)

DEV = 1
START = np.array([4649.8, 11208.3, 16032.5, 7163.1])
TARGET = [np.nan, np.nan, np.nan, 7214.4]


class FakeLib:
    """Stands in for libum: a goto sets the drive status to BUSY, as libum does on a sent goto."""

    def __init__(self):
        self.drive_status = LIBUM_POS_DRIVE_COMPLETED
        self.motor_busy = 0
        self.gotos = 0
        self.stops = 0

    def um_get_drive_status(self, h, dev):
        return self.drive_status

    def um_is_busy(self, h, dev):
        return self.motor_busy

    def um_goto_position_ext(self, h, *args):
        self.gotos += 1
        self.drive_status = LIBUM_POS_DRIVE_BUSY
        return 0

    def um_stop(self, h, dev):
        self.stops += 1
        return 0


def make_ump():
    """A UMP with a fake libum and a settable position; no SDK or hardware."""
    ump = UMP.__new__(UMP)
    ump.lock = threading.RLock()
    ump.loggers = {}
    ump._last_move = {}
    ump._completion_codes = {}
    ump.refused_goto_timeout = 3.0
    ump._debug = False
    ump.h = object()
    ump.lib = FakeLib()
    ump.default_max_accelerations = {DEV: 0}
    ump.pos = START.copy()
    ump.get_pos = lambda dev, timeout=0: list(ump.pos)
    return ump


def start_move(ump, target=TARGET, max_attempts=3):
    move = MoveRequest(ump, DEV, target, 100, True, True, 0, 0.1, 0.999, max_attempts=max_attempts)
    ump._last_move[DEV] = move
    move.start()
    return move


def refuse(ump, code=6):
    """Simulate the device refusing the last goto: FAILED status, code logged, no motion."""
    ump.lib.drive_status = LIBUM_POS_DRIVE_FAILED
    ump._completion_codes[DEV] = (code, timer())


class TestDriveStatus(TestCase):
    def test_failed_is_a_status_not_an_error(self):
        ump = make_ump()
        ump.lib.drive_status = LIBUM_POS_DRIVE_FAILED
        self.assertEqual(ump.get_drive_status(DEV), LIBUM_POS_DRIVE_FAILED)
        self.assertFalse(ump.is_busy(DEV))

    def test_invalid_device_raises(self):
        ump = make_ump()
        ump.lib.drive_status = LIBUM_INVALID_DEV
        with self.assertRaises(UMError) as cm:
            ump.get_drive_status(DEV)
        self.assertEqual(cm.exception.errno, LIBUM_INVALID_DEV)

    def test_completion_code_must_postdate_the_goto(self):
        ump = make_ump()
        ump._completion_codes[DEV] = (6, 100.0)
        self.assertEqual(ump.completion_code(DEV, since=99.0), 6)
        self.assertIsNone(ump.completion_code(DEV, since=101.0))


class TestRefusedGoto(TestCase):
    def test_refused_goto_is_resent_without_using_an_attempt(self):
        ump = make_ump()
        move = start_move(ump)
        refuse(ump)
        ump._update_moves()
        self.assertEqual(ump.lib.gotos, 2)
        self.assertEqual(move.attempts, 1)
        self.assertEqual(move.refusals, 1)
        self.assertEqual(move.last_completion_code, 6)
        self.assertFalse(move.finished)

    def test_refusals_beyond_max_attempts_keep_resending(self):
        ump = make_ump()
        move = start_move(ump, max_attempts=1)
        for _ in range(20):
            refuse(ump)
            ump._update_moves()
        self.assertEqual(ump.lib.gotos, 21)
        self.assertFalse(move.finished)

    def test_refused_past_timeout_fails_with_refusal_message(self):
        ump = make_ump()
        move = start_move(ump)
        refuse(ump)
        ump._update_moves()
        move._refusal_streak_start = timer() - 4.0
        refuse(ump)
        ump._update_moves()
        self.assertTrue(move.interrupted)
        reason = move.interrupt_reason
        self.assertTrue(reason.startswith("device refused the goto for "), reason)
        self.assertIn("completion code 6", reason)
        self.assertIn("2 goto(s) refused by the device without moving", reason)
        self.assertNotIn(DEV, ump._last_move)

    def test_acceptance_resets_the_refusal_streak(self):
        ump = make_ump()
        move = start_move(ump)
        refuse(ump)
        ump._update_moves()
        move._refusal_streak_start = timer() - 4.0
        ump.lib.motor_busy = 1  # the resent goto is carried out
        ump._update_moves()
        self.assertIsNone(move._refusal_streak_start)
        self.assertFalse(move.finished)

    def test_refused_refinement_within_fail_threshold_finishes(self):
        ump = make_ump()
        ump.pos = np.array([4649.8, 11208.3, 16032.5, 7214.2])  # 0.2 µm short: retry, not fail
        move = start_move(ump)
        refuse(ump)
        ump._update_moves()
        self.assertTrue(move.finished)
        self.assertFalse(move.interrupted)

    def test_unacknowledged_goto_is_reported_as_such(self):
        ump = make_ump()
        move = start_move(ump)
        ump.lib.drive_status = LIBUM_POS_DRIVE_FAILED
        move._refusal_streak_start = timer() - 4.0
        move._send_error = UMError("UM Error: -3: 'Timeout'", -3, None)
        ump._update_moves()
        self.assertTrue(move.interrupt_reason.startswith("device did not acknowledge the goto"),
                        move.interrupt_reason)


class TestFailedAfterMoving(TestCase):
    def test_failure_after_motion_is_a_normal_attempt(self):
        ump = make_ump()
        move = start_move(ump)
        ump.lib.motor_busy = 1
        ump._update_moves()  # BUSY, motor running
        ump.pos = np.array([4649.8, 11208.3, 16032.5, 7190.0])
        ump.lib.drive_status = LIBUM_POS_DRIVE_FAILED
        ump._completion_codes[DEV] = (3, timer())
        ump._update_moves()
        self.assertEqual(move.refusals, 0)
        self.assertEqual(move.attempts, 2)  # retried via the normal attempt budget
        self.assertEqual(move.last_completion_code, 3)

    def test_final_failure_reports_completion_code(self):
        ump = make_ump()
        move = start_move(ump, max_attempts=1)
        ump.pos = np.array([4649.8, 11208.3, 16032.5, 7190.0])  # moved, then stopped short
        ump.lib.drive_status = LIBUM_POS_DRIVE_FAILED
        ump._completion_codes[DEV] = (3, timer())
        ump._update_moves()
        self.assertTrue(move.interrupt_reason.startswith("move failed with completion code 3"),
                        move.interrupt_reason)


class TestCompletedShort(TestCase):
    def test_completed_short_fails_after_attempts(self):
        ump = make_ump()
        move = start_move(ump, max_attempts=1)
        ump.pos = np.array([4649.8, 11208.3, 16032.5, 7200.0])
        ump.lib.drive_status = LIBUM_POS_DRIVE_COMPLETED
        ump._update_moves()
        self.assertTrue(move.interrupt_reason.startswith("move finished but did not reach target position"))

    def test_failure_message_tolerates_none_in_target(self):
        ump = make_ump()
        move = start_move(ump, target=[None, None, None, 7214.4], max_attempts=1)
        ump.lib.drive_status = LIBUM_POS_DRIVE_COMPLETED
        ump._update_moves()
        self.assertIn("axis 3: 51.3", move.interrupt_reason)
