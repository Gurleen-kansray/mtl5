# PPE — Platform Performance Engineering

A laboratory for measuring where GEMM performance comes from, and where it goes.

Six GEMM implementations, each adding exactly one optimization to the previous,
measured across seven operand types (`int8`/`int16`/`int32`/`int64` and
`fp16`/`fp32`/`fp64`) as a function of matrix size. Every kernel is verified
against the naive reference before its timing is recorded.

**Standalone by design.** Nothing here includes `<mtl/...>`. The kernels are
written from scratch so each step can be isolated and attributed; the findings
then feed back into MTL5's own kernels.

## Quick start

```bash
cd ppe
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
taskset -c 4 ./build/ppe_gemm --sizes 64,128,256,512,1024,2048 --ghz 5.0 --csv data/ppe_gemm.csv
./analyze/plot_ppe.py data/ppe_gemm.csv --outdir data
```

Pin to a performance core (`taskset`) — on a hybrid CPU an unpinned run can land
on an E-core and report a number that describes different silicon.

## Where to start reading

| | |
|---|---|
| **[docs/RESULTS.md](docs/results.md)** | the measurements, the graphs, and what they mean |
| [docs/00-overview.md](docs/00-overview.md) | method, the type matrix, and what "efficiency" is measured against |
| [docs/05-vectorization-cliff.md](docs/05-vectorization-cliff.md) | the most useful finding: where the progression goes backwards |

## Layout

```
ppe/
  include/ppe/kernels.hpp   the six-step progression
  include/ppe/harness.hpp   timing, verification, throughput
  include/ppe/peak.hpp      the peak model, and why it is a model
  src/ppe_gemm.cpp          driver -> CSV
  analyze/plot_ppe.py       the figures
  docs/                     one write-up per step, plus consolidated results
  data/                     CSVs and generated figures
```
