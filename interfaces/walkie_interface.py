import logging
import os

from .devices.speaker import Speaker
from .devices.microphone import Microphone
from .devices.camera import Camera, CameraSnapshot
from walkie_sdk.robot import WalkieRobot

_log = logging.getLogger(__name__)


def _resolve_mic_device(explicit: int | str | None) -> int | str | None:
    """Pick the input device: explicit arg wins, else $WALKIE_MIC_DEVICE, else the
    system default (None). The env value may be an index ("4") or a name substring
    ("fifine") — a name is more robust than an index, which can shuffle when USB
    devices are re-enumerated across reboots.

    A name is resolved to the index of an INPUT-capable device whose name contains
    it: the same USB mic often also exposes a playback/monitor entry with zero input
    channels, and a bare name hands sounddevice that one (so capture stays silent).
    Returns the matched index, or the raw value as a last resort."""
    if explicit is not None:
        return explicit
    val = (os.getenv("WALKIE_MIC_DEVICE") or "").strip()
    if not val:
        return None
    if val.lstrip("-").isdigit():
        return int(val)
    try:
        import sounddevice as sd
        needle = val.lower()
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0 and needle in dev["name"].lower():
                return i
    except Exception as exc:  # noqa: BLE001 — fall back to the raw name
        _log.warning("mic device name lookup failed (%s); using %r as-is", exc, val)
    return val


class WalkieInterface:
    def __init__(self, robot: WalkieRobot, microphone_device: int | str | None = None):
        self._robot = robot
        self._microphone = Microphone(device=_resolve_mic_device(microphone_device))
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
