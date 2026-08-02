# Session: benchmark results site, CI coverage gaps, and a four-bug correctness sweep

**Date**: 2026-08-02
**Duration**: Full day session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Two threads that turned out to be the same story.

First, publish MTL5's benchmark numbers properly (#320): a landing page describing
the machines, linking to per-system result pages with figures and written
assessments — a structure that scales to more hardware without reorganizing.

Second, resolve the correctness bugs that `mtl5-python` had filed while binding
the dense and iterative surface. Four of them, each withholding a binding.

The connecting thread: every one of these defects lived in code that some test
or lane *appeared* to cover but structurally could not.

## Context

The day started with `cmake --preset release` building nothing. That is not a
bug in the preset — MTL5 is header-only, and the preset switched off tests and
examples while benchmarks default off, so the tree had zero buildable targets.
Turning benchmarks on immediately surfaced two source files that had not
compiled since `lu_numeric` was encapsulated, because **no CI runner compiles
`benchmarks/*.cpp`**.

That set the pattern for the rest of the day.

## Work Completed

### Benchmark results site (#320 → #325, #328)

A full measurement pass on the i7-12700K: five BLAS backends, three LAPACK
backends, four GEMM-scaling backends, five #297 kernel families, and both sparse
scoreboards — 20 CSVs. Every backend was verified to link what its label claims
before measuring (`ldd` plus each binary's self-reported config), so no curve is
silently the generic path wearing a vendor label.

The previously committed CSVs predated #227 and covered only `dot`/`nrm2`/
`gemv`/`gemm`; this run covers the full 15-op L1/L2/L3 core and adds BLIS, which
had never been measured.

Findings: native-fast GEMM at ~82% of OpenBLAS single-threaded and 6.51× on 8
P-cores (up from 5.82×); native LAPACK 8–45× behind the vendors; MKL 3.5× slower
than BLIS below N ≈ 300; native sparse LU 2–18× behind SuperLU with the gap
tracking fill ratio; sparse triangular solves flat-to-negative, reproducing the
existing #297 result.

**A published claim had to be retracted mid-session.** The first pass reported a
reproducible GEMM dip at N=465 that failed the #82 gate. Gathering supporting
evidence contradicted it: a fine sweep showed N=465 flat with its neighbours. The
"three reproducible repeats" backing the claim had been taken on a *loaded*
machine (38–40 GFLOP/s where idle gives 56.9), and their ratios were compared
against sweep-protocol ratios rather than like-for-like isolated runs at
neighbouring sizes. The control that would have caught it was never run. #328
corrected the page, kept the superseded claim visible rather than silently
rewriting it, and added a "treat differences under ~5 points as noise" rule.

### The #82 gate (#327 → #332)

The retraction exposed the real problem: the gate failed if *any* single size
fell below 80%, i.e. it gated on the minimum — the noisiest statistic available.

| statistic | run A | run B | swing |
|---|---:|---:|---:|
| median of ratios | 82.3% | 82.5% | **0.2 pt** |
| min (the old rule) | 76.3% | 78.7% | 2.4 pt |
| worst single size | — | — | **6.2 pt** |

Now asserts the median with a per-size floor at 0.70, so a genuine cliff still
fails while ±6-point noise does not. Verified against a synthetic 50% cliff,
which the median alone would have passed at 81.4%.

The issue had proposed "more iterations" as the variance fix; the data says
otherwise — within-run stddev is 0.3–2.2% at 10 iterations, so the variance is
*between* runs. That is recorded so it is not retried.

### Three CI coverage gaps (#329, #330 → #331, #333 → #334)

- **Benchmark sources compiled by no runner** — the gap that let the `lu_numeric`
  breakage survive. Now built on every push in native and vendor configurations.
  Verified the lane catches its own motivating bug by reintroducing it.
- **The Highway SIMD backend compiled nowhere.** `simd/batch.hpp`'s Highway
  specialization is the SIMD register type under the blocked GEMM, and no
  workflow set `MTL5_WITH_HIGHWAY` — every lane built the `size == 1` scalar
  fallback, while that flag is what every published benchmark number uses.
- **Draft → ready never re-ran CI.** `pull_request` declared no `types`, so
  `ready_for_review` fired nothing and Tier 2 stayed skipped from the draft-era
  runs. Silent, because a skipped job reports green. Found by hitting it on #332.

### Four correctness bugs (#335, #337, #324, #323)

| | defect | mechanism |
|---|---|---|
| #335 | `ldlt_bk` wrong on every pivot interchange, `info == 0` | `symmetric_swap` permuted L columns written by earlier steps, putting the factor in the "single global P" convention while the solve replays per-step permutations |
| #337 | `svd` all-NaN for ~30% of symmetric input | unscaled `sum(x^2)` underflow → `0/0` in `householder`, plus `0 * inf` in reflector application; and the alternating-QR iteration corrupted its own converged answer |
| #324 | `two_norm<Acc>` / `frobenius_norm<Acc>` uncompilable for 2 of 3 accumulator configurations | rounded out to the accumulator *type*, which has no `sqrt` unless it is a plain scalar |
| #323 | `ilu_0` wrong for every input | back substitution summed the diagonal *and* divided by it |

All four validated against external ground truth where one exists — LAPACK
`dsytrf` (5700 matrices, 4560→0 failures), `dgesvd` (1800 matrices, 0 NaN, 3.0e-15
worst deviation) — or against exactness where the algorithm is exact by
construction (ILU(0) on no-fill patterns: 0.000e+00).

### Release v5.8.0

Tagged and released. Minor, not patch: 22 `feat` commits since v5.7.0 add public
API. The `[Unreleased]` changelog section had no coverage of the four fixes, so
release prep added them — with `svd`'s algorithm/`tol` change and `ldlt_bk`'s
factor-layout change under `### Changed`, since neither is a silent improvement
for an existing caller.

## Decisions & Rationale

**Fix the cause, not the instance.** Three of the four bugs came with a correct
one-line fix in the issue, and in each case the underlying problem was redundant
or ambiguous state rather than a missing guard:

- `ilu_0` stored the diagonal in two places, and of two consumers one filtered it
  and one forgot. Storing it once makes `u_rows` strictly upper by construction —
  a second guard would leave the same trap for the next consumer.
- `two_norm<Acc>` could have rounded out to `mag_t` and compiled, but that
  narrows the sum *before* the square root, discarding what the accumulator
  bought. `accumulator_round_type` names the accumulator's own precision instead,
  and deliberately falls back to `mag_t` for custom accumulators so the quire
  specialization — which lives in a peer repo — needs no changes.
- `symmetric_swap` restricted to columns `>= k` rather than adding a second
  guard in the solve; the stored factor now matches `dsytrf` entry for entry.

**Tests must prove they cover the failing regime.** Every one of these bugs
survived under passing tests that structurally could not reach it: n=2 and n=3
for Bunch–Kaufman (no interchanges possible), small structured matrices for SVD,
plain arithmetic accumulators for the norms, and "does the solve converge" for
`ilu_0`. The new tests therefore assert the regime is live — that an interchange
occurred, that a 2×2 block appeared, that ILU(0) was exact — so they cannot
quietly stop covering the bug. Each was verified to fail without its fix.

**Publish the negative results.** `ilu_0` on a 2-D Laplacian takes 29 BiCGSTAB
iterations before *and* after the fix; only on a tridiagonal system, where the
factorization is exact, does it go 11 → 1. Recording that explains why the bug
survived and stops anyone concluding the fix did nothing. Likewise the retracted
N=465 claim is left visible on the results page.

**Measure before choosing.** The SVD rewrite was not the first instinct — a
cheaper stopping-criterion fix was tested first and rejected on evidence: the
diagonal's relative change decreases monotonically *through* the corruption, so
there is no threshold to tune.

## Outcome

- **14 PRs merged**, no open PRs, **no open bug-labelled issues**
- **v5.8.0 released** — https://github.com/stillwater-sc/mtl5/releases/tag/v5.8.0
- Benchmark results live at `/mtl5/benchmarks/`
- Three CI coverage gaps closed; the classes of code that hid these bugs are now
  compiled and exercised
- All four `mtl5-python` blockers cleared; that repo notified with the two
  behaviour changes and two corrections to its own checklist assumptions

Open follow-up: **#342**, filed after auditing #261's Part A — the accumulator
workspaces in `sparse_lu`, `supernodal_lu` and `supernodal_ldlt` are
`std::vector<Accumulator>` sized by the problem, free for a scalar but 8–67×
larger per element for a quire. Worth settling before Part C threads the
accumulator through more kernels.

## Issues / PRs

| Issue | PR | Subject |
|---|---|---|
| — | #319 | release preset builds benchmarks; harness fix + README + docs-site |
| #321 | #322, #336 | `bench_sparse` 3-D grid clamp; `SPARSE_SIZES` default |
| #320 | #325, #328 | i7-12700K result pages, figures, data; gate-finding correction |
| #327 | #332 | #82 gate on the median with a per-size floor |
| #330 | #329, #331 | benchmark build lane; Highway SIMD lane |
| #333 | #334 | CI runs on draft → ready |
| #335 | #338 | `ldlt_bk` pivot interchange |
| #337 | #339 | SVD one-sided Jacobi + Householder hardening |
| #324 | #340 | norm accumulator round-out |
| #323 | #341 | `ilu_0` back substitution |
| — | #343 | release v5.8.0 |
| #342 | — | open: quire workspace memory footprint |
