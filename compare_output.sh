#!/usr/bin/env bash
set -eu

ChN=2

OUR_BIN=./zig-out/bin/toyc-ch${ChN}
TOY_BIN=./toy_bin/toyc-ch${ChN}

DATA_DIR=./toy_examples/Ch${ChN}
DATA_NAME=codegen.toy
INPUT_FILE=${DATA_DIR}/${DATA_NAME}

OTHER_ARGS=( '--emit=mlir' )

if [[ ! -f ${INPUT_FILE} ]]; then
    echo "[ERROR] failed to find input file"
    exit 1
fi

nvim -d <( ${OUR_BIN} ${INPUT_FILE} ${OTHER_ARGS[@]} 2>&1 ) \
    <( ${TOY_BIN} ${INPUT_FILE} ${OTHER_ARGS[@]} 2>&1 )

