# X2 temporal IK and per-sequence production batch

Authorized goal: compare inner-IK smoothness, velocity/delta bounds with near-bound avoidance, and their combination on the same 11 BONES-SEED clips. Keep the weight optimizer's outer smoothness score unchanged. Exclude collision/contact-dynamics redesign. Select a measured setting, then independently optimize weights and export every SOMA Uniform sequence as X2 CSV.

Clock starts 2026-10-27 14:18 UTC; target deadline 23:18 UTC. This is a target, not an unmeasured throughput guarantee.

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
- All derivative metrics use original source dt, radians, within-clip differences. Floating-root Euler derivatives are not used.
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

## Verified launch constraints

- `clusterx --include` matches Kubernetes hostnames, not management node names. The authorized fragmented nodes `ecp-tcluster-a844` and `ecp-tcluster-ab45` correspond to `host-10-140-62-36` and `host-10-140-108-112`. The corrected probe scheduled on the first physical node in eight seconds; the management-name selector remained pending.
- The configured launcher already supplies the shared `/data` mount. Adding the same mount explicitly produces a rejected duplicate-mount request; inspect the composed payload.
- SSP job names have a 32-character maximum.
- The shared Python executable resolves into persistent storage. Launchers still explicitly prepend and verify their isolated source repository because the environment's editable installation otherwise selects the old repository.
- Eight real filesystem/configuration tests pass on the shared runtime. Batched optimization also passes native B1/B2 accepted/rejected-update parity and mixed 1/2/3-frame full-post/export checks on CPU; GPU end-to-end throughput remains a separate launch gate.
- The eleven disabled-temporal control CSVs reproduce all numerical values of the existing frozen optimized comparison outputs exactly. This checks backward compatibility, not temporal improvement.
