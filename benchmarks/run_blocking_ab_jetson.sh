#!/usr/bin/env bash

# Running the A/B there

# thermal state FIRST -- a throttled run reads as a blocking effect
sudo nvpmodel -q && sudo jetson_clocks
cat /sys/devices/virtual/thermal/thermal_zone*/temp

# compile the benchmarks
cmake --preset release -DMTL5_WITH_HIGHWAY=ON
cmake --build build-release --target bench_blocking_ab_detected bench_blocking_ab_default -j4

# run the benchmarks
BENCH_PCPUS=0,1,2,3,4,5,6,7 THREADS="1 8" ROUNDS=5 OUTDIR=benchmarks/data/jetson-orin ./benchmarks/run_blocking_ab.sh

./benchmarks/analyze_blocking_ab.py benchmarks/data/jetson-orin/blocking_ab_{detected,default}.csv

#  BENCH_PCPUS=0,1,...,7 because the A78AE has no SMT — every logical id is a physical core, unlike the x86 boxes. 
# No MTL5_NATIVE_ARCH needed either; NEON is 128-bit and there's nothing wider to reach.
# Once the CSVs exist, commit them plus the .sysinfo sidecars under benchmarks/data/jetson-orin/, 
# and I'll fill in the systems.md TODOs from the hardware block you ran — I still need that output (module name, L4T, clocks, memory, compiler, power mode).

