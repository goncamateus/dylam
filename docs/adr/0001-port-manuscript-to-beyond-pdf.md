# Port the manuscript into the Beyond PDF format

Issue #38 built the λ-simplex scrubber as a *complement* to the PDF under review,
and its Out of Scope section explicitly ruled out both "porting the manuscript
itself into the Beyond PDF document format" and "conforming to TMLR's author-kit
directory layout". We are reversing both: the full manuscript is ported to
`submission.md` and the author kit becomes the delivery format.

The reason is that the complement framing turned out not to exist. TMLR's
submission instructions state that `submission.md` "serves as the complete
manuscript" and that the archival PDF must be produced by browser print-to-PDF of
the rendered page. There is no supported shape in which a Beyond PDF folder
attaches to a separately-authored PDF, so a slim companion page would have made a
three-page archival PDF stand in for a 17k-word paper.

## Consequences

The λ-simplex scrubber stops being the deliverable and becomes one Embed among
five. `beyond_pdf/dylam_beyond_pdf.html` will be deleted once that Embed ships
inside the submission folder, since a second committed copy would go stale on
the first rebuild after that point.
