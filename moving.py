
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from track import HeadTracker

_WORLD_PKG = "head_tracking_world"

#loads 3D world from test.py so head tracker can be used to control the world
def _load_world_module():
    path = Path(__file__).resolve().parent / "test.py"
    spec = importlib.util.spec_from_file_location(_WORLD_PKG, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_WORLD_PKG] = mod
    loader = spec.loader
    if loader is None:
        raise RuntimeError(f"Cannot load {path}")
    loader.exec_module(mod)
    return mod

#connects the calibrated head tracker to the 3D world
def bridge_calibrated_tracker_to_world(head_tracker: HeadTracker) -> None:
    """Step 2 of the pipeline: hand the live ``HeadTracker`` to ``test.run_world_with_head_tracker``."""
    world = _load_world_module()
    world.run_world_with_head_tracker(head_tracker)

#controller that runs the head tracker first and then opens the 3D world
def main() -> None:
    print(
        "[moving.py] Step 1/2 — Webcam only (track.py). "
        "Show your face and press C in the calibration window. Q quits."
    )
    tracker = HeadTracker(verbose_direction_print=False, draw_landmarks_legacy=True)
    while True:
        state = tracker.tick_calibration_only()
        if state == "quit":
            tracker.close()
            sys.exit(0)
        if state == "done":
            break

    if not tracker.calibrated:
        tracker.close()
        sys.exit(1)

    print(
        "[moving.py] Step 2/2 — Bridge → test.py world (integrated). "
    )
    try:
        bridge_calibrated_tracker_to_world(tracker)
    finally:
        tracker.close()
        print("[moving.py] Done.")


if __name__ == "__main__":
    main()
