# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The steps of `oot2 prepare` in order (architecture D24c) and the runner that keeps their record (D5).

Each step ticket adds its step to all_steps() when it lands: extraction and module decoding, lift, compile and link,
then the shader steps. A step names the earlier steps it reads from in Step.needs. The translate step needs only the
extraction and comes right after it, so that a translator that cannot work stops the run in minutes, not after the
compile. The ledger step reads what the game recorded on its last run and translates it as variants; it comes right
after the translate step and no later step reads its output.
"""

from __future__ import annotations

from .check import Check
from .extract import Extract
from .ledger import Ledger
from .modules import Modules
from .program import Compile, Lift, Link
from .runner import Step
from .translate import Translate


def all_steps() -> list[Step]:
    return [Check(), Extract(), Translate(), Ledger(), Modules(), Lift(), Compile(), Link()]
