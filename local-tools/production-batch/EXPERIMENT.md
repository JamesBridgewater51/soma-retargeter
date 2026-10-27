# X2 temporal IK and per-sequence production batch

Authorized goal: compare inner-IK smoothness, velocity/delta bounds with near-bound avoidance, and their combination on the same 11 BONES-SEED clips. Keep the weight optimizer's outer smoothness score unchanged. Exclude collision/contact-dynamics redesign. Select a measured setting, then independently optimize weights and export every SOMA Uniform sequence as X2 CSV.

Clock starts 2026-10-27 14:18:38 UTC; target deadline 23:18:38 UTC. The nine-hour target was missed. A direct filesystem observation at 23:44:05 UTC found 141,666 of 142,220 canonical completion receipts; this historical count is not a full-dataset audit or a COMPLETE declaration. Conversion continues with the frozen method and search budget.

## Execution isolation

- Fork: JamesBridgewater51/soma-retargeter.
- Branch: feat/x2-temporal-per-sequence-batch.
- Existing native source and previous comparison outputs are preserved.
- Shared assets resolve through explicit STORAGE_ROOT; no data aliases.
- Temporal implementation, controlled ablation, and production scheduling have disjoint file ownership during development.

## Comparison gates

- Same source, calibrated targets, existing per-clip IK table, FPS, and full postprocessing across temporal variants.
- Include disabled/native control and report both base and full-post results.
- Measure task-space position/rotation fit; actuated-joint velocity, acceleration and jerk RMS, upper tails and maxima; velocity-limit violations and worst frame/joint.
- Derivatives are computed within each clip in radians, using native-loaded sample timing; production velocity validation uses each clip's native FPS. All full-corpus files load at 120.0 FPS after native rounding. Their raw BVH frame time, 0.008333 seconds, is preserved separately rather than substituted as an exact reciprocal FPS. Floating-root Euler derivatives are not used.
- Angle bounds are not velocity bounds. Warm starts are not temporal regularization.
- A claimed hard velocity limit must hold on the actual final exported CSV, including postprocessing effects.
- Preserve reports for rejected settings; do not force a winner if tracking degradation or violations persist.

## Production acceptance

- Every source sequence has its own optimizer result. No shared-table substitution or default output mislabeled optimized.
- Interrupted jobs resume at validated stage boundaries; native optimizer iteration resume is not claimed.
- CSV output is atomically published after finite-value, frame count, schema and independent velocity validation.
- Coverage is checked against the complete input manifest with duplicate/missing/error counts.
- Failures are explicit and isolated; unsuccessful runs are not completion.
- Progress and throughput-derived ETA are persisted and surfaced.

## Selected inner-IK configuration

`temporal-selected.json` freezes prediction residual multiplier 0.25, hard velocity bounds at the URDF limit, and normalized near-velocity-limit avoidance multiplier 0.1 with a 20% activation margin. The extra positional near-limit term is disabled; original native positional constraints remain. Outer weight-optimizer smoothness scoring is unchanged.

Corrected adaptive LM preserves damping across iterations; the earlier per-iteration-reset hard-limit experiments are retained as rejected implementation evidence, not used for selection. Across the same eleven clips and frozen optimized weights, pooled full-output position RMSE changes from 0.111528 to 0.113602 m, jerk RMS from 13244 to 7953 rad/s³, and maximum absolute jerk from 1594271 to 302619 rad/s³. Default-weight position RMSE changes from 0.124994 to 0.126818 m, jerk RMS from 10577 to 9076 rad/s³, and maximum absolute jerk from 1012506 to 294383 rad/s³. Both selected branches have zero exported velocity-limit exceedances at relative tolerance 1e-4. Rotation RMSE increases from 0.355380 to 0.362030 rad with frozen optimized weights and from 0.356299 to 0.362692 rad with default weights. Some default-weight P99 values increase; this is not an all-metrics improvement claim.

Evidence: the PT artifact `temporal-adaptive-damping-comparison/aggregate-comparison.json`, under `artifacts/soma-retargeter/agibot-x2ultra/`. `optimizer-selected.json` separately freezes four native outer iterations per sequence. This is an explicit bounded search budget, not a claim of fifty-iteration convergence. A genuine four/eight-iteration prefix pilot over all eleven inputs gave pooled final position RMSE 0.126034/0.125301 m, jerk RMS 7336/7124 rad/s³, and maximum absolute jerk 292626/309509 rad/s³; both passed all exported speed checks. Eight iterations doubled search time (314.94 to 629.18 seconds) for a 0.733 mm position gain. Four was selected for the execution deadline and retained lower observed maxima. Objective definitions, learning rate, guard policy, and temporary postprocessing overrides are otherwise unchanged.

## Verified launch constraints

- `clusterx --include` matches Kubernetes hostnames, not management node names. The authorized fragmented nodes `ecp-tcluster-a844` and `ecp-tcluster-ab45` correspond to `host-10-140-62-36` and `host-10-140-108-112`. The corrected probe scheduled on the first physical node in eight seconds; the management-name selector remained pending.
- The configured launcher already supplies the shared `/data` mount. Adding the same mount explicitly produces a rejected duplicate-mount request; inspect the composed payload.
- SSP job names have a 32-character maximum.
- The shared Python executable resolves into persistent storage. Launchers still explicitly prepend and verify their isolated source repository because the environment's editable installation otherwise selects the old repository.
- Eight real filesystem/configuration tests pass on the shared runtime. Batched optimization also passes native B1/B2 accepted/rejected-update parity and mixed 1/2/3-frame full-post/export checks on CPU; GPU end-to-end throughput remains a separate launch gate.
- The eleven disabled-temporal control CSVs reproduce all numerical values of the existing frozen optimized comparison outputs exactly. This checks backward compatibility, not temporal improvement.
