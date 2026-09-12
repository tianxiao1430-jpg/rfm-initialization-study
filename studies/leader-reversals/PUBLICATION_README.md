# Public package: reproduction and data guide

This supplement is published as repository artifact **v1.1.0**, on September
13, 2026. The paper PDF remains at its existing editorial version. This package
contains the analysis scripts, all derived tables, two PNG/SVG figures, the
Chinese report, source hashes, and verification records.

## Downloads

- [Analysis archive on Hugging Face](https://huggingface.co/datasets/tianxiao1430-jpg/rfm-initialization-study/resolve/main/archives/rfm-leader-reversals.zip)
- [Analysis archive on GitHub](https://github.com/tianxiao1430-jpg/rfm-initialization-study/releases/download/v1.1.0/rfm-leader-reversals.zip)
- [Required original scores on Hugging Face](https://huggingface.co/datasets/tianxiao1430-jpg/rfm-initialization-study/resolve/main/archives/rfm-mechanism-full.zip)
- [Required original scores on GitHub](https://github.com/tianxiao1430-jpg/rfm-initialization-study/releases/download/v1.0.0/rfm-mechanism-full.zip)

The raw archive already existed. All 4,222 original input files used by this
analysis were checked byte-for-byte against it before publication. The additional review check also verifies the 80 original metric-state files.
The source manifest's remaining two entries are this analysis's original protocol and
analyze.py. Raw score arrays are not duplicated inside the new analysis archive.

## Verify or recompute

Download both archives into an empty working directory. The following layout
preserves the paths recorded by the original analysis and its input hashes:

```text
replay/
  outputs/
    rfm-mechanism/
    rfm-leader-reversals/
```

```bash
python -m zipfile -e rfm-leader-reversals.zip replay
python -m zipfile -e rfm-mechanism-full.zip replay/outputs
cd replay/outputs/rfm-leader-reversals
python -m pip install -r requirements.txt
python publication_check.py
python -X utf8 verify_analysis.py
python -X utf8 verify_timeline.py
```

To regenerate the outputs, in the extracted copy:

```bash
python -X utf8 analyze.py
python -X utf8 verify_analysis.py
python -X utf8 review_addendum.py
python -X utf8 build_report.py
python -X utf8 verify_timeline.py
```

Python 3.12, NumPy 2.3.5 and Matplotlib 3.11.2 were used for this saved-output
analysis. It requires no GPU training. Figure fonts and generated timestamps can
vary across machines. The bundled publication checksum check applies to the
downloaded snapshot, before regenerating files. The Git checkout and Hugging
Face file tree expose the same package under studies/leader-reversals; use the
archive layout above for running scripts without changing any recorded paths.

## Table units

- `run_summary.csv`: 844 phase/name runs, including repeats and deterministic
  controls; these are not 844 independent initialization directions.
- `primary_all_steps_by_run.csv`: 9,600 rows = 80 directions x 2 views x 60
  anchors. `displaced` counts sequences whose anchor leader group is later beaten.
- `primary_all_steps_summary.csv`: aggregates those rows by modulus/view/anchor.
- `sequence_summary.csv`: one pointwise or aligned-profile score sequence per row.
- `anchor_sequences.csv`: sequence-level evidence at the stated anchor steps.
- `events.csv`: 56,452 event records across all phases, both views and event types.
  `leader_group_replaced` is a tied leader group being beaten; `wrong_to_correct`
  and `correct_to_wrong` are clear sign changes of the correctness margin.
  One crossing can generate both a leader and correctness record. Do not add
  event types and interpret that sum as unique crossings or independent samples.
  `previous_step` and `step` bracket the previous and new definite states.
  Pointwise class numbers are actual labels; profile class numbers are offsets
  after aligning each test point's correct answer to offset zero.
- `matched_amplitude_comparison.csv`: all 20 paired amplitude controls, including
  the 17 larger-amplitude directions without a correct-to-wrong event.
- `example_traces.csv`: all 60 steps for each of three explicitly selected examples.
- `sensitivity_*.csv`: four relative tie tolerances; the tolerance is a sensitivity
  convention, not a rigorous bound on the numerical error of the whole training.

Step 0 is the initial kernel fit, before any AGOP update. A step-t output follows
t feature updates. The 50 short probes end at 5 and are excluded from conclusions
requiring observation through 59. The protocol was written after the original
experiments; its choice of the old confirmation set is not a new preregistration.

The review follow-up adds `endpoint_stratification.csv` (nine modulus/endpoint
strata, including zero-count ambiguous endpoints), `endpoint_point_history.csv`
(1,540 points), and `saved_metric_coverage.csv` (80 original direction records).
`review_addendum.py` recomputes those tables from the original histories and
metric-state archives. Its additional source hashes are in
`review_input_manifest.json`; many history entries overlap the earlier manifest.

## Verification records

`verification.json` checks original hashes, per-run anchors, event witnesses and
semantic edge cases. `timeline-verification.json` independently checks every
primary anchor, matched amplitude controls and plotted scores. Their checks
refer to saved outputs and do not constitute independent model training.
`PUBLICATION_MANIFEST.json` and `publication_check.py` verify this delivered
package. The publication status is recorded on the two project home pages.
