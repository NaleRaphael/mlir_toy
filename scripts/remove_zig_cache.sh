#!/usr/bin/env bash
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TARGET_DIR=${THIS_DIR}/..

# NOTE: for Zig < 0.13.0, cache folder was named as "zig-cache".
# https://github.com/ziglang/zig/issues/20077
ZIG_CACHE_DIR='.zig-cache'

pushd ${TARGET_DIR} > /dev/null
    if [[ -d ${ZIG_CACHE_DIR} ]]; then
        echo "[INFO] Found ${ZIG_CACHE_DIR} folder created here, cleaning..."
        rm -rf ${ZIG_CACHE_DIR}
    else
        echo "[WARN] Trying to clear ${ZIG_CACHE_DIR} but it's not found here"
    fi
popd > /dev/null

