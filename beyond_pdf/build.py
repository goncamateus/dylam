"""Beyond PDF build entry point.

Runs the Embed generators, then assembles beyond_pdf/submission_folder into
beyond_pdf/tmlr_do_not_modify (the author kit) and starts the kit's local
Docker/Jekyll render, by invoking the kit's own compile_submission.py --
which already implements that merge-and-serve sequence. compile_submission.py
is left untouched so the vendored kit stays a faithful, diffable copy.

Step order:
  1. build_scrubber() -- runs beyond_pdf/export.py (the lambda-simplex
     scrubber Embed generator from issue #38, wired into this build by
     issue #43) against real ChickenBanana/Dylam checkpoints and writes the
     Embed HTML under submission_folder/assets/html/submission/, where
     compile_submission.py's own copy step already looks (see its
     merge_files()). Those checkpoints -- scripts/train_q_learning.py
     --checkpoint-interval snapshots -- are NOT committed to this
     repository (see export.py's module docstring and issue #39's
     "Checkpoint availability" note), so this step is expected to be
     unavailable on a fresh clone. It fails gracefully: a diagnostic on
     stderr and a skip, not a fabricated artifact and not a crash that
     blocks the rest of the build.
  2. assemble_and_serve() -- the kit's existing merge-and-serve sequence.

Later tickets extend this module -- not compile_submission.py -- with the
remaining steps: the mechanism/ablation/per-environment/Pareto Embed
generators, static-figure SVG regeneration, and table HTML emission. This is
the one build entry point; new steps land here rather than in parallel
scripts.
"""
import argparse
import subprocess
import sys
from pathlib import Path

BEYOND_PDF_DIR = Path(__file__).resolve().parent
REPO_ROOT = BEYOND_PDF_DIR.parent
EXPORT_SCRIPT = BEYOND_PDF_DIR / "export.py"
CURRICULUM_DATA = REPO_ROOT / "result_analysis" / "curriculum" / "data"
ACTUAL_RETURNS = BEYOND_PDF_DIR / "data" / "chickenbanana_actual_returns.csv"
SUBMISSION_HTML_DIR = BEYOND_PDF_DIR / "submission_folder" / "assets" / "html" / "submission"
SCRUBBER_OUT = SUBMISSION_HTML_DIR / "lambda_simplex_scrubber.html"

# The published seed the scrubber Embed is built from: the lower of the two
# middle published seeds, ranked by mean summed per-Component return over the
# final 10% of episodes (export.py's DEFAULT_SEED_CRITERION). Matches the
# #38 artifact this build step replaces (see that commit's message).
SCRUBBER_SEED = 1764531329
SCRUBBER_LATTICE_STEP = 20

# Not committed -- see build_scrubber()'s docstring. Reproduce by training
# this seed with scripts/train_q_learning.py --setup Dylam --env
# CHICKENBANANA --seed 1764531329 --checkpoint-interval 10, then point
# --snapshots-dir at the run's models/.../snapshots directory.
DEFAULT_SNAPSHOTS_DIR = REPO_ROOT / "scripts" / "models" / "chickenbanana_dylam_snapshots"


def build_scrubber(snapshots_dir=DEFAULT_SNAPSHOTS_DIR):
    """Build the lambda-simplex scrubber Embed via beyond_pdf/export.py and
    write it into the submission folder's assets/html/submission/ directory,
    where compile_submission.py's merge step already picks up per-page HTML
    subdirectories. Returns True if the artifact was (re)built, False if the
    step was skipped because no checkpoints were found."""
    snapshots_dir = Path(snapshots_dir)
    if not snapshots_dir.is_dir():
        print(
            "beyond_pdf build: skipping the lambda-simplex scrubber Embed -- "
            f"no snapshots directory at {snapshots_dir}. This step needs real "
            "ChickenBanana/Dylam checkpoints from scripts/train_q_learning.py's "
            "--checkpoint-interval flag, which are not committed to this "
            "repository (see beyond_pdf/export.py's module docstring and "
            "issue #39's \"Checkpoint availability\" note). Train seed "
            f"{SCRUBBER_SEED} with --checkpoint-interval 10, then re-run this "
            "build with --snapshots-dir pointing at the resulting snapshots/ "
            "directory. The existing artifact in the submission folder (if "
            "any) is left as-is.",
            file=sys.stderr,
        )
        return False

    SUBMISSION_HTML_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(EXPORT_SCRIPT),
        "--snapshots", str(snapshots_dir),
        "--out", str(SCRUBBER_OUT),
        "--curriculum-data", str(CURRICULUM_DATA),
        "--seed", str(SCRUBBER_SEED),
        "--lattice-step", str(SCRUBBER_LATTICE_STEP),
        "--actual-returns", str(ACTUAL_RETURNS),
    ]
    subprocess.run(cmd, check=True)
    return True


def assemble_and_serve():
    subprocess.run(
        [sys.executable, "compile_submission.py"],
        cwd=BEYOND_PDF_DIR,
        check=True,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshots-dir", type=Path, default=DEFAULT_SNAPSHOTS_DIR,
                     help="ChickenBanana/Dylam checkpoint snapshots for the "
                     "scrubber Embed (not committed to this repository)")
    ap.add_argument("--skip-scrubber", action="store_true",
                     help="skip the scrubber Embed generation step")
    ap.add_argument("--skip-serve", action="store_true",
                     help="run the Embed generation steps but skip the kit's "
                     "assemble + Docker/Jekyll serve step")
    args = ap.parse_args()

    if not args.skip_scrubber:
        build_scrubber(args.snapshots_dir)

    if not args.skip_serve:
        assemble_and_serve()


if __name__ == "__main__":
    main()
