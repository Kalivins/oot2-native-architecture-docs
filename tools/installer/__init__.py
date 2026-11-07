# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The setup window over `oot2 prepare` (architecture D24c): tools/oot2_setup.pyw starts it.

    logic.py    everything the window decides, without Tk: the form, the parse of prepare's lines into step states,
                the plain-word errors, the child process and its cancel, the remembered paths, the game's start
    window.py   the window itself (tkinter), which only draws what logic.py decides
    shot.py     a PNG of a window, for the self-test's screenshots (Windows only, no extra package)

The window runs `tools/oot2.py prepare` as a child process and reads its plain output; it opens no connection of its
own, and what it remembers (two paths and a checkbox) stays in the player's profile.
"""
