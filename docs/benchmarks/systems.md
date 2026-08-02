# Benchmark Systems

Every performance number published for MTL5 is tied to a specific machine, a
specific set of vendor libraries, and a specific pinning policy. This page is
the index of those machines. Each one links to its full result page.

Performance claims do not travel between machines. A GEMM efficiency measured
on a desktop hybrid CPU says little about a server part with different cache
sizes, memory channels, and turbo behaviour — so read the results page together
with the system description here.

## Systems

| System | CPU | Cores used | Results |
|--------|-----|-----------|---------|
| `i7-12700K` | 12th Gen Intel Core i7-12700K | 8 P-cores (E-cores excluded) | [Results](i7-12700k.md) |

## `i7-12700K` — desktop hybrid (Alder Lake)

The primary MTL5 development machine.

| | |
|---|---|
| CPU | 12th Gen Intel Core i7-12700K |
| Topology | 8 P-cores (2 threads each, logical 0–15) + 4 E-cores (logical 16–19) |
| Clocks | P-cores 4.9–5.0 GHz max, E-cores 3.8 GHz max, 800 MHz min |
| Cache | L1d 512 KiB (12 instances), L2 12 MiB (9 instances), L3 25 MiB (shared) |
| Memory | 31 GiB |
| OS | Ubuntu 24.04.4 LTS, kernel 6.8.0-136 |
| Compiler | GCC 13.3.0, `-O3 -DNDEBUG` (CMake `Release`) |
| CPU governor | `powersave` (`intel_pstate`; still boosts, but clocks are not pinned) |

### Vendor libraries

| Backend | Version | Selected by |
|---------|---------|-------------|
| OpenBLAS | 0.3.26 (pthread build) | `-DMTL5_WITH_BLAS=ON -DMTL5_WITH_LAPACK=ON` |
| BLIS | 0.9.0 | `-DMTL5_WITH_BLAS=ON -DBLA_VENDOR=FLAME` |
| Intel MKL | oneAPI 2026.1 | `-DBLA_VENDOR=Intel10_64lp` (after `setvars.sh`) |
| SuiteSparse KLU | 7.6.1 | `-DMTL5_WITH_SUITESPARSE_KLU=ON` |
| SuperLU | 6.0.1 | `-DMTL5_WITH_SUPERLU=ON` |
| Google Highway | 1.0.7 | `-DMTL5_WITH_HIGHWAY=ON` (native-fast SIMD) |

OpenBLAS is the active `libblas.so.3` alternative on this host, so a build
configured with `MTL5_WITH_BLAS=ON` and no `BLA_VENDOR` genuinely links
OpenBLAS rather than the reference netlib BLAS — the `openblas` label means what
it says.

### Pinning policy

This is a **hybrid P/E-core CPU**, which makes pinning mandatory rather than
optional. An unpinned single-threaded run lets short L1 kernels land on a 3.8 GHz
E-core and reports a number that has nothing to do with the 5.0 GHz P-core the
same code would reach in production. The failure mode is documented in
[the multi-core scaling investigation](../design/multicore-scaling-investigation.md).

| Run type | Pinning | Rationale |
|----------|---------|-----------|
| Single-thread sweeps | `taskset -c 4` | One P-core (5.0 GHz bin) |
| Scaling runs, `T` threads | first `T` of `0,2,4,6,8,10,12,14` | One logical id per **physical** P-core — SMT siblings excluded, so scaling reflects cores, not hyperthreads |

Single-threaded runs additionally pin every vendor's threading to one thread
(`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`BLIS_NUM_THREADS=1`, `MTL5_NUM_THREADS=1`) so no backend quietly uses more of
the machine than another.

E-cores are never used for measurement. Nothing here characterises them.

## Methodology

One executable per backend. MTL5 dispatches to BLAS/LAPACK **at compile time**,
so the build *is* the backend — there is no runtime policy switch, and each
binary measures only its own backend. This mirrors what a dependent application
actually gets. The reasoning is spelled out in the
[benchmark harness](../../benchmarks/README.md) page.

Each binary is verified to link what its label claims before any measurement —
`native` resolves no external BLAS, `openblas` resolves `libopenblas.so.0`,
`blis` resolves `libblis.so.4`, `mkl` resolves the `libmkl_*` trio — and each
binary self-reports its compiled configuration in its output header.

BLIS is a **BLAS-only** library. An MTL5 build against it has no LAPACK path, so
a "blis" factorization curve would silently be the generic native path wearing a
vendor label. BLIS therefore appears in BLAS results and is absent from LAPACK
results by construction, not by oversight.

## Adding a system

1. Run the suites documented in the [harness README](../../benchmarks/README.md) on the new machine.
2. Add a result page `docs/benchmarks/<system>.md` following the existing one.
3. Add a row to the table at the top of this page and a section describing the
   hardware, libraries, and pinning policy.
4. Register the page in `FILE_MAP` in `docs-site/sync-content.mjs`.

The per-system pages are self-contained, so a new machine never requires editing
an existing result page.
