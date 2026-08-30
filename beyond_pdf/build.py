"""Beyond PDF build entry point.

Assembles beyond_pdf/submission_folder into beyond_pdf/tmlr_do_not_modify
(the author kit) and starts the kit's local Docker/Jekyll render, by
invoking the kit's own compile_submission.py -- which already implements
that merge-and-serve sequence. compile_submission.py is left untouched so
the vendored kit stays a faithful, diffable copy.

Later tickets extend this module -- not compile_submission.py -- with the
steps that must run before assembly: Embed generation, static-figure SVG
regeneration, and table HTML emission. This is the one build entry point;
new steps land here rather than in parallel scripts.
"""
import subprocess
import sys
from pathlib import Path

BEYOND_PDF_DIR = Path(__file__).resolve().parent


def assemble_and_serve():
    subprocess.run(
        [sys.executable, "compile_submission.py"],
        cwd=BEYOND_PDF_DIR,
        check=True,
    )


if __name__ == "__main__":
    assemble_and_serve()
