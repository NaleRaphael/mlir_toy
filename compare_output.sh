#!/usr/bin/env bash
set -eu

ChN=7

OUR_BIN=./zig-out/bin/toyc-ch${ChN}
TOY_BIN=./toy_bin/toyc-ch${ChN}

DIR_TMP=/tmp
OUR_OUT=${DIR_TMP}/result_our.txt
TOY_OUT=${DIR_TMP}/result_toy.txt

DATA_DIR=./toy_examples/Ch${ChN}
DATA_NAME=struct-ast.toy
INPUT_FILE=${DATA_DIR}/${DATA_NAME}

OTHER_ARGS=(
    '--opt=true'
    # '--emit=llvm'
    '--mlir-print-stacktrace-on-diagnostic=true'
    '--mlir-print-op-on-diagnostic=true'
    '--mlir-print-op-generic=false'
    '--mlir-print-ir-before-all=true'
    '--mlir-print-ir-after-all=true'
    '--mlir-disable-threading=true'
)

OUR_ARGS=(
    '--emit=mlir_llvm'
)
TOY_ARGS=(
    '--emit=mlir-llvm'
)

if [[ ! -f ${INPUT_FILE} ]]; then
    echo "[ERROR] failed to find input file"
    exit 1
fi

if [[ ! -d ${DIR_TMP} ]]; then
    echo "[ERROR] DIR_TMP does not exist: ${DIR_TMP}"
    exit 1
fi

${OUR_BIN} ${INPUT_FILE} ${OTHER_ARGS[@]} ${OUR_ARGS[@]} > ${OUR_OUT} 2>&1
${TOY_BIN} ${INPUT_FILE} ${OTHER_ARGS[@]} ${TOY_ARGS[@]} > ${TOY_OUT} 2>&1

nvim -d ${OUR_OUT} ${TOY_OUT}
