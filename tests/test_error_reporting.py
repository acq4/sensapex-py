import threading
import time
import warnings
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from sensapex.sensapex import (
    LIBUM_INVALID_ARG,
    LIBUM_INVALID_DEV,
    LIBUM_POS_DRIVE_BUSY,
    LIBUM_POS_DRIVE_COMPLETED,
    LIBUM_POS_DRIVE_FAILED,
    LIBUM_TIMEOUT,
    UMP,
    MoveRequest,
    SensapexDevice,
    UMError,
)

DEV = 1
START = np.array([4649.8, 11208.3, 16032.5, 7163.1])
ERRORSTRS = {LIBUM_TIMEOUT: b"Timeout", LIBUM_INVALID_ARG: b"Invalid argument", LIBUM_INVALID_DEV: b"Invalid device id"}


class FakeLib:
    """Stands in for libum: a goto sets the drive status to BUSY, as libum does on a sent goto."""

    def __init__(self):
        self.drive_status = LIBUM_POS_DRIVE_COMPLETED
        self.last_error = 0

    def um_get_drive_status(self, h, dev):
        return self.drive_status

    def um_goto_position_ext(self, h, *args):
        self.drive_status = LIBUM_POS_DRIVE_BUSY
        return 0

    def um_stop(self, h, dev):
        return 0

    def um_last_error(self, h):
        return self.last_error

    def um_errorstr(self, code):
        return ERRORSTRS.get(code, b"?")


def make_ump():
    """A UMP with a fake libum and a settable position; no SDK or hardware."""
    ump = UMP.__new__(UMP)
    ump.lock = threading.RLock()
    ump.loggers = {}
    ump._last_move = {}
    ump._debug = False
    ump.h = SimpleNamespace(contents=SimpleNamespace(timeout=20))
    ump.lib = FakeLib()
    ump.default_max_accelerations = {DEV: 0}
    ump.pos = START.copy()
    ump.get_pos = lambda dev, timeout=0: list(ump.pos)
    return ump


class TestDriveStatus(TestCase):
    def test_failed_is_a_status_not_an_error(self):
        ump = make_ump()
        ump.lib.drive_status = LIBUM_POS_DRIVE_FAILED
        ump.lib.last_error = LIBUM_INVALID_DEV  # stale, from an earlier call
        self.assertEqual(ump.get_drive_status(DEV), LIBUM_POS_DRIVE_FAILED)
        self.assertFalse(ump.is_busy(DEV))

    def test_invalid_device_raises(self):
        ump = make_ump()
        ump.lib.drive_status = LIBUM_INVALID_DEV
        with self.assertRaises(UMError) as cm:
            ump.get_drive_status(DEV)
        self.assertEqual(cm.exception.errno, LIBUM_INVALID_DEV)


class TestCallErrors(TestCase):
    def test_error_code_comes_from_return_value_not_last_error(self):
        ump = make_ump()
        ump.lib.last_error = LIBUM_TIMEOUT  # stale, from an earlier call
        ump.lib.um_fake = lambda h, *args: LIBUM_INVALID_ARG
        with self.assertRaises(UMError) as cm:
            ump.call("um_fake", 1)
        self.assertEqual(cm.exception.errno, LIBUM_INVALID_ARG)
        self.assertIn("Invalid argument", str(cm.exception))

    def test_immediate_invalid_device_is_reported_as_such(self):
        ump = make_ump()
        ump.lib.um_fake = lambda h, *args: LIBUM_INVALID_DEV
        with self.assertRaises(UMError) as cm:
            ump.call("um_fake", 1)
        self.assertEqual(cm.exception.errno, LIBUM_INVALID_DEV)

    def test_invalid_device_after_waiting_is_reported_as_no_reply(self):
        ump = make_ump()  # handle timeout 20 ms

        def no_reply(h, *args):
            time.sleep(0.03)
            return LIBUM_INVALID_DEV

        ump.lib.um_fake = no_reply
        with self.assertRaises(UMError) as cm:
            ump.call("um_fake", 1)
        self.assertEqual(cm.exception.errno, LIBUM_TIMEOUT)
        self.assertIn("no reply from device", str(cm.exception))

    def test_retry_resends_the_same_arguments(self):
        ump = make_ump()
        calls = []

        def fails(h, *args):
            calls.append(args)
            return LIBUM_TIMEOUT

        ump.lib.um_fake = fails
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(UMError):
                ump.call("um_fake", 1, 2, retries=2)
        self.assertEqual(calls, [(1, 2)] * 3)


class TestMoveFailureMessage(TestCase):
    def _fail_short(self, target):
        ump = make_ump()
        move = MoveRequest(ump, DEV, target, 100, True, True, 0, 0.1, 0.999, max_attempts=1)
        ump._last_move[DEV] = move
        move.start()
        ump.lib.drive_status = LIBUM_POS_DRIVE_COMPLETED
        ump._update_moves()
        return move

    def test_completed_short_fails_after_attempts(self):
        move = self._fail_short([np.nan, np.nan, np.nan, 7214.4])
        self.assertTrue(move.interrupted)
        self.assertTrue(move.interrupt_reason.startswith("move finished but did not reach target position"))

    def test_failure_message_tolerates_none_in_target(self):
        move = self._fail_short([None, None, None, 7214.4])
        self.assertTrue(move.interrupted)
        self.assertIn("axis 3: 51.3", move.interrupt_reason)


class TestFirmwareWarning(TestCase):
    def _device(self, version=None, error=None):
        def get_firmware_version(dev_id):
            if error is not None:
                raise error
            return version

        logger = Mock()
        ump = SimpleNamespace(get_firmware_version=get_firmware_version, get_logger=lambda dev_id: logger)
        return SimpleNamespace(dev_id=DEV, ump=ump), logger

    def test_old_firmware_warns(self):
        dev, logger = self._device(version=(1, 26, 45, 0, 0))
        SensapexDevice._warn_if_old_firmware(dev)
        logger.warning.assert_called_once()
        self.assertIn("uMp firmware 1.26.45.0 is older than 1.26.46.101", logger.warning.call_args[0][0])

    def test_current_firmware_is_quiet(self):
        for version in [(1, 26, 46, 101, 0), (1, 27, 0, 0, 0)]:
            dev, logger = self._device(version=version)
            SensapexDevice._warn_if_old_firmware(dev)
            logger.warning.assert_not_called()

    def test_unreadable_firmware_warns_without_raising(self):
        dev, logger = self._device(error=UMError("UM Error: -3: 'Timeout'", LIBUM_TIMEOUT, None))
        SensapexDevice._warn_if_old_firmware(dev)
        logger.warning.assert_called_once()
        self.assertIn("could not read the firmware version", logger.warning.call_args[0][0])
