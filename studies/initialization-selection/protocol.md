# RFM initialization selection: a budget-matched pilot

## Material Passport
- Mode: run; synthetic numerical research, authored and implemented with Codex assistance.
- Question: does short-probe selection retain the benefit of a complete response bank at an equal kernel-fit budget?
- Status: protocol written before new-direction outcomes; a local hash seal is not external preregistration.
- Scope: the existing Gaussian RFM, modular addition, fixed-point split, FP64 RTX 4070. No general neural-network or unknown-task claim.

## Fixed design
Training uses every unequal pair. Six diagonal inputs at p=17 and eight at p=23 are validation inputs, chosen by a fixed permutation. The remaining 11/15 diagonal inputs are final test inputs. They are never evaluated by selection code. All methods know the modular-addition rule and receive the same validation labels. This is an algorithmic toy task: separating evaluation does not make the algebraically determined test labels unknowable.

Twenty new directions at p=17 (seeds 2000--2019) and twenty at p=23 (3000--3019) are the sampling units. Implementation checks use separate seeds. Seeds, split, scoring, budgets and analysis are sealed before these directions are run. No direction is excluded for poor accuracy.

Every candidate preserves the original noncirculant residual and each circulant Fourier-coordinate magnitude. Only coordinates 1 through d-1 may change sign; coordinate 0 stays at its original sign for every method. Thus the claim is restricted to this common relative-sign search space, not every possible initializer. The unmodified candidate is code 0.

## Methods and information
1. **bank**: rebuild the baseline, all single-mode and positive-pair probes, each for 60 fits. Build the finite-amplitude tensor from validation predictions only, using epsilon=0.002. Enumerate the common sign space and maximize the existing normalized correct-offset margin. No old fitted isotonic calibration or old direction outcomes are used. Train the chosen epsilon=0.01 initializer for 60 fits.
2. **random_probe**: evaluate a fixed random permutation of unique sign candidates, beginning with code 0, for 11 fits (indices 0--10). Select by validation accuracy, then mean pointwise margin divided by each point's centered-score RMS, then earlier evaluation order. Resume the chosen checkpoint for 49 further fits.
3. **greedy_probe**: start at code 0. Test coordinate flips in a seeded fixed order, accepting strict improvement under the same validation score. Cache already visited candidates. If an entire pass produces no unvisited neighbor, restart at the next unused code from the same random candidate ordering. Continue to the same unique-probe count as random_probe, then resume the best candidate for 49 fits. All exploratory restarts are charged.
4. **random_full**: fully train the longest affordable prefix of the same random ordering, beginning with code 0, and select on final validation scores. Retain code 0 as a descriptive unmodified reference; its computation is already charged here. It is not an extra free candidate for other methods.

All short/full selectors receive only training and validation objects, never test predictions. All final choices and model files must be sealed before a separate evaluator reads the held-out inputs. Bank construction also queries validation inputs only. The tensor assumes the known output-offset structure; this is prior task knowledge, not learned universality.

## Equal budget ceilings
The deployment workload is fixed at N=20 direction queries per modulus. A bank requires B=1+d(d+1)/2 trajectories, d=(p-1)/2. Its complete cost is 60B+60N fits. Each competitor gets the same per-query ceiling C=60+60B/N.

| p | bank trajectories B | C fits/query | short probes floor((C-49)/11) | short actual fits | full random candidates | full actual fits |
|---|---:|---:|---:|---:|---:|---:|
|17|37|171|11|170|2|120|
|23|67|261|19|258|4|240|

These are equal ceilings, not a claim of identical work consumed or identical seconds. Unused remainder cannot buy another complete candidate and is reported rather than padded with useless computation. Count fits, metric updates, candidates, synchronized wall time, and peak CUDA memory separately. Physical cross-method result reuse is forbidden in the measurement run, even when candidate codes coincide. Warmup and verification runs are excluded and separately recorded. Method order is randomized per direction, with one active experiment process.

Report both one-query cost (the entire bank charged once) and N=20 amortized cost. Also report the algebraic break-even query count from measured bank setup and mean measured online times. This is an extrapolation with workload assumptions; it is not an additional measured accuracy experiment at other budgets.

Timing includes candidate construction, validation transfers, selection, continuation, and model/checkpoint/log writes up to the final selected model (the tiny result-summary JSON is excluded). Bank setup includes its model files and tensor construction. Process startup, data/tensor loading, warmup, independent verification and final held-out evaluation are excluded. These are single-process wall times with CUDA synchronization, not isolated GPU kernel times; storage/desktop contention may affect them.

## Outcomes and analysis fixed in advance
Primary outcome: final held-out classification accuracy at fit index 59. Secondary: held-out continuous normalized margin, validation--test accuracy gap, correct counts, and costs. Publish every direction and every candidate's validation score; never select best seeds. All three bank-versus-competitor contrasts are reported separately for each modulus. Use paired direction bootstrap 95% intervals (10,000 resamples, fixed seed 20260909); two-sided paired Monte Carlo sign flips (100,000 draws) with Holm adjustment across the six contrasts. Sign flips assume a symmetric paired null and are not assignment-based randomization tests. Intervals are not multiplicity-adjusted and no power guarantee is claimed. Avoid pooling the two moduli as interchangeable tasks. For tie-heavy discrete accuracy, also report wins/ties/losses.

This is a finite pilot. A useful positive result requires comparable or better held-out accuracy at materially lower measured cost, not just a favorable p-value. If simple search matches or beats the bank, retain that result and narrow the original method claim. A null or negative result does not terminate data reporting.

## Required verification
- Original kernel/gradient/AGOP source bytes and GPL notice preserved; old study and submitted-paper files untouched.
- At least one full trajectory matches the established implementation; saved short checkpoint continuation matches uninterrupted execution exactly.
- Candidate norm, PSD, spectrum-magnitude and residual invariants; identical random candidate prefixes; budget and choice replay checks.
- No test evaluation file exists before the global selection seal; evaluator checks that seal and the frozen plan/code hashes.
- Recompute published accuracy/margins from saved final predictions. Re-run one complete selection per method and modulus and compare choices and numerical outputs, excluding timings.
- The eight reruns use the first confirmation seed of each modulus (2000/3000), fixed before outcomes. Inference is conditional on this one validation split per modulus. Final model condition numbers and near-tie counts are numerical diagnostics, not outcome-based exclusions.
- All required 40 directions x 4 methods completed, failure records retained, separate method/candidate/run counts.
- Deliver protocol, raw candidate logs, chosen models, held-out scores, direction tables, analysis, scientific figures and a concise Chinese report. Do not change the earlier manuscript or publish externally as part of this pilot.
