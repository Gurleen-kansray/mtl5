#!/usr/bin/env bash

echo "=== module / L4T ==="; cat /proc/device-tree/model 2>/dev/null; tr -d '\0' < /etc/nv_tegra_release 2>/dev/null
echo "=== topology / clocks ==="; lscpu | grep -E "Model name|Architecture|^CPU\(s\)|Thread|Core|Socket|MHz"; lscpu -e=CPU,CORE,MAXMHZ
echo "=== ISA (SVE matters for #427) ==="; grep -m1 Features /proc/cpuinfo
echo "=== cache + sharing, as MTL5 reads it ==="
for d in /sys/devices/system/cpu/cpu0/cache/index*; do \
  echo "$(cat $d/level) $(cat $d/type) $(cat $d/size) line=$(cat $d/coherency_line_size) shared=$(cat $d/shared_cpu_list)"; done
echo "=== memory (shared with GPU) / OS / compilers ==="; free -g | sed -n 2p; grep PRETTY_NAME /etc/os-release; uname -r; g++ --version | head -1
echo "=== power/thermal state -- numbers are meaningless without these ==="
sudo nvpmodel -q; sudo jetson_clocks --show 2>/dev/null | head -5; cat /sys/devices/virtual/thermal/thermal_zone*/temp

# Step 0 -- --preset ci now works here thanks to #435
cmake --preset ci -DMTL5_WITH_HIGHWAY=ON
cmake --build build-ci --target util_test_cache_info -j4
./build-ci/tests/unit/util_test_cache_info -s | grep -E "l1d=|runtime " | sort -u

