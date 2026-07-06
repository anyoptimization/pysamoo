"""Docs runner for ``pyclawd docs`` — maps pyclawd's docs verbs onto a Sphinx build of ``docs/source``.

``pyclawd docs`` is a thin orchestrator: it appends a sub-verb (``compile``/``run``/``all``/``build``/
``exec``/``clean``) to this script and runs it. pysamoo's docs are a single, pre-executed
``index.ipynb`` rendered by nbsphinx, so ``compile``/``run`` are no-ops (there is nothing to generate
and the notebook's stored outputs are reused rather than re-executing the expensive algorithm cells);
the ``all``/``build``/``render`` verbs all shell out to ``sphinx-build``.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
BUILD_DIR = ROOT / "build"
HTML = BUILD_DIR / "html"


def _sphinx_html() -> int:
    """Render the HTML site from ``docs/source`` (nbsphinx reuses the notebook's stored outputs)."""
    return subprocess.call([sys.executable, "-m", "sphinx", "-b", "html", str(SOURCE), str(HTML)])


def _exec_page(page: str) -> int:
    """Execute a single notebook in place and stream any error (``pyclawd docs exec <page>``)."""
    nb = SOURCE / page
    if not nb.exists():
        print(f"docs exec: no such page {nb}", file=sys.stderr)
        return 2
    return subprocess.call(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(nb)]
    )


def main(argv: list) -> int:
    verb = argv[0] if argv else "all"
    rest = argv[1:]
    if verb in ("compile", "run"):
        return 0  # single pre-executed notebook: nothing to compile or run
    if verb in ("all", "build", "render"):
        return _sphinx_html()  # --continue / --fast accepted and ignored (one page, stored outputs)
    if verb == "exec":
        return _exec_page(rest[0]) if rest else 2
    if verb == "clean":
        if BUILD_DIR.exists():
            shutil.rmtree(BUILD_DIR)
        return 0
    print(f"docs runner: unknown verb {verb!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
