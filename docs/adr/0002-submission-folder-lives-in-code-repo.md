# The submission folder lives in the code repo, not the manuscript repo

`submission.md` is the manuscript, so it would naturally track `DyLam-TMLR`
alongside the `.tex` it is ported from. We put the whole `beyond_pdf/` tree —
submission folder, generators, unpacked author kit — in this repo instead.

Every Embed and every regenerated SVG is built by code here, from tidy CSVs and
checkpoints here. Splitting the folder from its generators would mean a
cross-repo copy step in `build.py` and a second place for the build to break.

## Consequences

The paper's prose now exists twice: as `.tex` in `DyLam-TMLR` and as markdown
here. The two will drift, and nothing detects it. `DyLam-TMLR` remains the source
of truth — when a claim changes, it changes there first and the port is re-run.
