import logging

from .devices.speaker import Speaker
from .devices.microphone import Microphone
from .devices.camera import Camera, CameraSnapshot
from walkie_sdk.robot import WalkieRobot

_log = logging.getLogger(__name__)

class WalkieInterface:
    def __init__(self, robot: WalkieRobot, microphone_device: int | str | None = None):
        self._robot = robot
        self._microphone = Microphone(device=microphone_device)
        # Give the speaker the mic so it can pause it while playing — otherwise the
        # robot transcribes its own TTS (a problem for any background listener).
        self._speaker = Speaker(mic=self._microphone)
        self._camera = Camera(robot)
        self._nav = robot.nav
        self._status = robot.status
        self._tools = robot.tools
        self._arm = robot.arm
    
    @property
    def robot(self) -> WalkieRobot:
        return self._robot

    @property
    def speaker(self) -> Speaker:
        return self._speaker

    @property
    def microphone(self) -> Microphone:
        return self._microphone

    @property
    def camera(self) -> Camera:
        return self._camera

    @property
    def nav(self):
        return self._nav

    @property
    def status(self):
        return self._status

    @property
    def tools(self):
        return self._tools

    @property
    def arm(self):
        return self._arm

    def capture_snapshot(self, *, log=None) -> CameraSnapshot | None:
        """Read this instant's RGB-D frame, camera pose, and intrinsics together.

        Reads depth, the RGB frame, the camera optical-frame pose, intrinsics, and
        the robot pose back-to-back into one
        :class:`~interfaces.devices.camera.CameraSnapshot`, so they all describe the
        same moment even before slow detection/LLM round-trips — the snapshot can
        then lift masks/bboxes from that frame to map-frame 3D points. The capture
        spans the whole interface (camera, transform, status), which is why it lives
        here rather than on the bare :class:`~interfaces.devices.camera.Camera`.

        Args:
            log: Optional callable taking a message string, for capture diagnostics.

        Returns:
            The snapshot, or ``None`` when depth or the RGB frame is unavailable.
        """
        if log is None:
            return CameraSnapshot.capture(self)
        return CameraSnapshot.capture(self, log=log)

    def close(self) -> None:
        """Best-effort teardown of every owned resource, for a clean shutdown.

        The SDK's rosbridge + zenoh connections spawn non-daemon threads; if
        they're left running the interpreter can't exit and hangs in
        ``threading._shutdown`` (the "Port 8500 still in use next launch"
        symptom — the process never actually dies). Disconnecting the robot
        stops those threads. Each step is isolated so one failing teardown
        can't strand the others, and the whole thing is idempotent.
        """
        for label, fn in (
            ("speaker", self._speaker.stop),
            ("camera", self._camera.close),
            ("robot", self._disconnect_robot),
        ):
            try:
                fn()
            except Exception:  # noqa: BLE001 — shutdown must not raise
                _log.exception("error closing %s during shutdown", label)

    def _disconnect_robot(self) -> None:
        robot = self._robot
        if robot is None:
            return
        # is_connected may itself raise on a half-dead transport; guard it.
        try:
            connected = robot.is_connected
        except Exception:  # noqa: BLE001
            connected = True
        if connected:
            robot.disconnect()
