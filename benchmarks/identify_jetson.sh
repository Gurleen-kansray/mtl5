#!/usr/bin/env bash

# module name -- /proc/device-tree entries are NUL-terminated, hence tr
tr -d '\0' < /proc/device-tree/model; echo
tr -d '\0' < /etc/nv_tegra_release 2>/dev/null | head -1

# power mode: the likely reason only 6 cores are online
sudo nvpmodel -q                      # current mode, e.g. "NV Power Mode: MODE_15W"
sudo nvpmodel -q --verbose 2>/dev/null | head -30

# THE question -- are cores physically absent, or switched off by the mode?
cat /sys/devices/system/cpu/online
cat /sys/devices/system/cpu/offline     # non-empty => the mode disabled them
nproc

# clocks actually in force, and thermal headroom during the run
sudo jetson_clocks --show 2>/dev/null | head
paste <(cat /sys/devices/virtual/thermal/thermal_zone*/type) <(cat /sys/devices/virtual/thermal/thermal_zone*/temp)

# memory, shared with the GPU
free -g | sed -n 2p

