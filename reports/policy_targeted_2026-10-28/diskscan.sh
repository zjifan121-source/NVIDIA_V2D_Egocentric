#!/bin/bash
df -h /data/user_data/edmondwa | tail -1
cd /data/user_data/edmondwa
for d in *; do [ "$d" = v2d ] && continue; timeout 600 du -s "$d"; done | sort -n | tail -8
for d in v2d/* v2d/flash_runs/*; do timeout 600 du -s "$d"; done | sort -n | tail -10
echo DONE
