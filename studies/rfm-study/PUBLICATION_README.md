# RFM initialization: compact publication data and code

Companion to "Sign Interactions at Fixed Spectral Amplitudes: Predicting and
Intervening on RFM Initialization under a Symmetry Fixed-Point Split", Xiao Tian.

This package contains all 844 run configurations and final tabular outcomes,
all 47,940 per-step metric records, 80 frozen direction forecasts, the 30 paired
intervention summaries (including individual controls), and parameter boundaries.
Run counts include probes and repeats and must not be treated as independent seeds.

Start with results/all_runs.csv, results/all_steps.csv, results/confirmation.csv,
results/intervention.csv, results/boundary.csv, and facts.json. Field definitions
are in data_dictionary.md. The original five figures are in figures/; the revision diagnostic is in revision/.

Run these read-only checks from the extracted rfm-study directory:

    python src/verify_artifact.py
    python publication_check.py

The first validates this release's file hashes. The second recalculates counts,
final/step table agreement, forecast errors and paired effects using only the
Python standard library. Neither constitutes independent scientific validation.

## Reproduction

With the recorded dependencies (requirements.txt) and a compatible CUDA PyTorch:

    python src/reproduce.py --phase smoke --output /path/to/new-smoke
    python src/reproduce.py --phase all --output /path/to/new-run

Outputs must be new directories. The smoke command compares an ordinary cross
trajectory and a custom sign intervention against the two included final.npz
reference files, including their complete saved score histories. Training was
performed on an RTX 4070, FP64, one PyTorch CPU thread, TF32 disabled. Details and
the prior clean-copy smoke report are preserved in results/.

All executed source files, frozen plans, models, intervention initializations,
and required reference_v2 inputs are included without changing their contents.
Large per-run states.npz files and most final.npz arrays are omitted; regenerate
them with --phase all before rerunning full array-based analysis/audit programs.
Every run's JSON/JSONL configuration, summaries and metrics remain included.
PUBLICATION_MANIFEST.json lists every omission and every included original hash.
FULL_LOCAL_MANIFEST.json preserves the original full-archive inventory; the new
artifact_manifest.json applies to this compact package. Historical local reports
may describe the full archive; use this README to interpret release completeness.

## Attribution and limitations

Code derives from marceltomas/breaking-data-symmetries, commit
311273bfc08adcde344c8b021fc5aa8d9970ad98. LICENSE is GPL-3.0 for code; NOTICE.md
preserves attribution and AI assistance disclosure. The manuscript's arXiv
distribution license is separate and does not replace the code license.
Codex assisted planning, implementation, execution, analysis, audit and writing.
The numerical results came from executed programs. These are setting-specific
computational findings; no independent human review or global novelty proof is
claimed. Packaging does not establish that arXiv accepted or announced the paper.

## Post hoc revision diagnostics

The 7 September 2026 revision adds revision/ with a CPU analysis script, extracted
inputs and provenance hashes, full direction-level diagnostics, tensor eigensystems,
and an analytic scaling table. These additions are outside the original freeze.
See revision/README.md for reproduction commands and limits. Original experimental
source, frozen forecasts and all original included results remain byte-identical.
