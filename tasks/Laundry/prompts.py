"""Spoken language for the Doing Laundry task (rulebook 5.4).

PLACEHOLDER. There is no operator dialogue in this test (no people involved
unless the robot asks for help), so this is mostly spoken status lines plus the
help requests the rulebook allows (opening the washing-machine door).
"""

from __future__ import annotations

# --- Spoken lines -----------------------------------------------------------
START_ANNOUNCE = "I am starting the laundry task."
ASK_OPEN_WASHER = (
    "I could not open the washing-machine door myself. Could you please open it for me?"
)
RETRIEVE_ANNOUNCE = "I will move the clothes to the folding table."
FOLD_ANNOUNCE = "I will now fold the clothes."
DONE_ANNOUNCE = "I have finished folding the laundry."

# Placeholder lines for the not-yet-implemented deformable-manipulation steps.
PICK_NOT_AVAILABLE = "I cannot pick up the clothing yet — manipulation is not implemented."
FOLD_NOT_AVAILABLE = "I cannot fold yet — folding is not implemented."
