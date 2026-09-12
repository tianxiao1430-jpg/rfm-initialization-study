# Post hoc manuscript revision, 7 September 2026

These diagnostics were designed after viewing the frozen study outcomes. They do
not constitute new confirmation, new training, model refitting, or independent review.
All new values derive from executed analysis of saved arrays; no synthetic result
was substituted for an experiment. Codex assisted this analysis and manuscript revision.

From the extracted ancillary root, with NumPy, SciPy and Matplotlib installed:

    python revision/revision_analysis.py --inputs revision/inputs --output /path/to/new-diagnostics

The inputs folder contains unmodified response tensors and saved scores, together
with circulant coordinates extracted from stored m0 matrices. provenance.json
lists original file hashes and explains the extraction. To recreate the input
extraction from a full regenerated study, use --extract /path/to/full-study and
--inputs /path/to/new-inputs. Extraction needs the large original final.npz arrays,
but recomputing these diagnostics uses only the compact supplied inputs.

- confirmation_continuous.csv: all 80 directions, quantized accuracy errors,
  predicted/measured continuous profile margins, and measured pointwise margins.
- threshold_sensitivity.csv: frozen classifier agreement at four thresholds.
- control_gains.csv: favorable/unfavorable control families, three controls
  averaged within each of 30 directions; 60 rows are not 60 independent directions.
- adverse_profiles.csv: 30 unfavorable score profiles and paired diagnostics.
- tensor_spectra.csv, tensor_eigensystems.npz, tensor_covariance.csv: all output
  class spectra, eigenvectors, and exact-basis relabeling checks against finite Q.
- amplitude_checks.csv: stored mode-1 trajectories and general-cross amplitude grid.
- cost_extrapolation.csv: analytic operation counts and single-kernel storage,
  NOT measured wall-clock time, peak memory, or executed larger-modulus experiments.
- analysis.json: statistics, methods, original primary comparison results and input hashes.

New control-gain intervals use 10,000 paired integer-seed cluster bootstrap
resamples, RNG 20260907, with no multiplicity adjustment. The original primary
comparison retains its original RNG and intervals. No greedy short-probe sign
search was executed; it is discussed as an untested baseline. No proof of late
quadratic dominance or analytic optimal-sign formula is claimed.
