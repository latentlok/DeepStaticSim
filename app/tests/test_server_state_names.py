"""Trame state names must not be JS reserved words.

Vue compiles template expressions with `new Function(...)`; a state variable named
`case` (or any reserved word) produces syntactically invalid render code and the
client dies with `Uncaught SyntaxError: Unexpected token 'case'` BEFORE mounting --
a fully blank page with every HTTP request returning 200. Measured via CDP on
2026-08-31; this guard makes the mistake impossible to repeat silently.
"""

from __future__ import annotations

import re
from pathlib import Path

# The ES reserved words plus the literals -- any of these as a trame state name
# breaks the generated render function.
RESERVED = {
    "break", "case", "catch", "class", "const", "continue", "debugger", "default",
    "delete", "do", "else", "enum", "export", "extends", "false", "finally", "for",
    "function", "if", "import", "in", "instanceof", "new", "null", "return", "super",
    "switch", "this", "throw", "true", "try", "typeof", "var", "void", "while",
    "with", "yield", "let", "static", "await",
}  # fmt: skip

APP = Path(__file__).resolve().parent.parent


def state_names(source: str) -> set[str]:
    """Names bound into trame state: v_model tuples and state.change registrations."""
    names = set(re.findall(r"""v_model=\(\s*["']([A-Za-z_]\w*)["']""", source))
    names |= set(re.findall(r"""state\.change\(\s*["']([A-Za-z_]\w*)["']\s*\)""", source))
    # state.change(key)(...) loops over a literal tuple of names
    for tup in re.findall(r"""for key in \(([^)]*)\):\s*\n\s*state\.change\(key\)""", source):
        names |= set(re.findall(r"""["']([A-Za-z_]\w*)["']""", tup))
    return names


def test_no_reserved_js_words_in_trame_state():
    for fname in ("server.py",):
        src = (APP / fname).read_text()
        names = state_names(src)
        assert names, f"{fname}: found no state names -- the regex went stale, fix the test"
        bad = sorted(names & RESERVED)
        assert not bad, (
            f"{fname}: state name(s) {bad} are JS reserved words -- Vue's compiled "
            f"render function becomes invalid JS and the page renders blank."
        )
