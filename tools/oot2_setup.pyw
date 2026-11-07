# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The setup window over `oot2 prepare` (tools/installer/window.py). A .pyw file opens with pythonw.exe on a
double-click, so no console window comes with it.

    pythonw tools/oot2_setup.pyw [--selftest [--shot DIR]]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from installer.window import main  # noqa: E402

sys.exit(main())
