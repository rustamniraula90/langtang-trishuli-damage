# Label audit: review record (27 Oct 2026)

**Resolved 27 Oct 2026:** the author set all 13 intact-roof "possibly damaged" rows (section B) to `plausible`
and accepted the section A calls as they stand. The paper's Label audit paragraph carries the final numbers
and the pending marker is removed.

All 200 rows of `label_audit.csv` now carry a `judgement` made from the HD chips in `chips_hd/`
(`hd_review.csv` adds a confidence flag). The rows below are the ones where the call is a matter of
policy or genuinely ambiguous. Edit `judgement`/`note` in `label_audit.csv`, then run
`.venv/bin/python 28_audit_summary.py` and update the "Label audit" paragraph of the paper.

## A. "Destroyed" buildings that may still stand (4 rows)
| sample | current | what the chip shows |
|---|---|---|
| AOI03_113 | unclear | village buried under sediment (green-cast 7 Sep image); a dark block with a shadow at the footprint may be a standing upper storey |
| AOI03_123 | unclear | same image and situation as 113 |
| AOI05_178 | disagree | building with light roof standing in a row of houses, mud around; no pre-event image |
| AOI05_179 | disagree | walls standing, roof stripped; "destroyed" vs "severely damaged" |

Already confirmed as standing (disagree, high confidence): AOI05_162, 164, 165, 167, 173, 176.
In 173 the point sits on the standing yellow-roofed house while its grey-roofed neighbour is buried,
so the point may simply be offset.

## B. Policy: "Possibly damaged" with an intact roof (13 rows)
A roof-level view cannot show ground-floor inundation. Rule applied: intact roof with the flood margin
within ~100 m -> `plausible`; intact roof and no flood trace anywhere in the 160 m window -> `disagree`.
If you prefer to call all intact-roof cases `plausible` (or all `disagree`), change the 13 rows together.

| sample | current | distance to flood trace in chip |
|---|---|---|
| AOI01_000 | plausible | sediment edge ~100 m |
| AOI01_002 | plausible | ~50 m |
| AOI01_008 | plausible | ~40 m downslope |
| AOI02_050 | plausible | debris field ~100 m |
| AOI02_051 | plausible | ~100 m |
| AOI03_101 | plausible | sediment on terraces below the road |
| AOI03_106 | plausible | ~60 m |
| AOI03_109 | plausible | debris field ~80 m below |
| AOI01_004 | disagree | none at walls; haze to one side |
| AOI02_052 | disagree | none in window |
| AOI02_056 | disagree | none in window; outline on roof corner |
| AOI03_104 | disagree | none nearby |
| AOI05_159 | disagree | none at building |

## C. Worth knowing, though not a label question
14 "not affected" footprints sit on trees or open ground with no building visible on either date
(AOI01_030/036/040, AOI02_080/085/087/090/091/094/096, AOI03_149, AOI05_185/193/199). They are
coded `unclear`. They point at Overture footprint placement; the Copernicus grades are not at issue.
