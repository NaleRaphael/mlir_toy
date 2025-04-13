#!/usr/bin/env bash
set -eu -o pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

DIR_TMP=/tmp
DIR_TOY_BIN=${THIS_DIR}/../toy_bin
DIR_TOY_EXAMPLES=${THIS_DIR}/../toy_examples
DIR_ZIG_BIN=${THIS_DIR}/../zig-out/bin

# Files to save the output of toyc-chX
CPP_OUT="${DIR_TMP}/result_cpp.txt"
ZIG_OUT="${DIR_TMP}/result_zig.txt"

VERBOSE=${VERBOSE:-0}
SELECTED_CHAPTER_NUM=${1:-""}

CHAPTERS=( Ch1 Ch2 Ch3 Ch4 Ch5 Ch6 Ch7 )

# These options are used to print details while processing MLIR
ENABLE_DEBUG=${ENABLE_DEBUG:-0}
DEBUG_OPTIONS=(
    --mlir-print-stacktrace-on-diagnostic=true
    --mlir-print-op-on-diagnostic=true
    --mlir-print-op-generic=true
    --mlir-print-ir-before-all=true
    --mlir-print-ir-after-all=true
    --mlir-disable-threading=true
)

# NOTE: XFAIL means the file is expected to trigger error.
Ch1_FILES=( ast.toy )
Ch1_FILES_XFAIL=( empty.toy )
Ch1_EMIT_TYPES=( ast )
Ch1_OTHER_ARGS=()

Ch2_FILES=( ast.toy codegen.toy scalar.toy )
Ch2_FILES_XFAIL=( empty.toy invalid.mlir )
Ch2_EMIT_TYPES=( ast mlir )
Ch2_OTHER_ARGS=()

Ch3_FILES=(
    ast.toy
    codegen.toy
    scalar.toy
    transpose_transpose.toy
    trivial_reshape.toy
)
Ch3_FILES_XFAIL=( empty.toy invalid.mlir )
Ch3_EMIT_TYPES=( ast mlir )
Ch3_OTHER_ARGS=( --opt=true )

Ch4_FILES=(
    ast.toy
    codegen.toy
    scalar.toy
    transpose_transpose.toy
    trivial_reshape.toy
    shape_inference.mlir
)
Ch4_FILES_XFAIL=( empty.toy invalid.mlir )
Ch4_EMIT_TYPES=( ast mlir )
Ch4_OTHER_ARGS=( --opt=true )

Ch5_FILES=(
    ast.toy
    codegen.toy
    scalar.toy
    transpose_transpose.toy
    trivial_reshape.toy
    shape_inference.mlir
    affine-lowering.mlir
)
Ch5_FILES_XFAIL=( empty.toy invalid.mlir )
Ch5_EMIT_TYPES=( ast mlir mlir-affine )
Ch5_OTHER_ARGS=( --opt=true )

Ch6_FILES=(
    ast.toy
    codegen.toy
    scalar.toy
    transpose_transpose.toy
    trivial_reshape.toy
    shape_inference.mlir
    affine-lowering.mlir
    jit.toy
    llvm-lowering.mlir
)
Ch6_FILES_XFAIL=( empty.toy invalid.mlir )
Ch6_EMIT_TYPES=( ast mlir mlir-affine mlir-llvm llvm jit )
Ch6_OTHER_ARGS=( --opt=true )

Ch7_FILES=(
    ast.toy
    codegen.toy
    scalar.toy
    transpose_transpose.toy
    trivial_reshape.toy
    shape_inference.mlir
    affine-lowering.mlir
    jit.toy
    llvm-lowering.mlir
    struct-ast.toy
    struct-codegen.toy
    struct-opt.mlir
)
Ch7_FILES_XFAIL=( empty.toy invalid.mlir )
Ch7_EMIT_TYPES=( ast mlir mlir-affine mlir-llvm llvm jit )
Ch7_OTHER_ARGS=( --opt=true )

check_var() {
    local file_type=$1
    local var_name=$2
    local var_value=${!var_name}

    if [[ $file_type != "d" && $file_type != "f" ]]; then
        echo "[ERROR] Only these file types are available to check: d, f"
        exit 1
    fi

    if [[ -z $var_value ]]; then
        echo "[ERROR] Please specify a value for $var_name"
        exit 1
    fi

    if [[ $file_type == "d" ]]; then
        if [[ ! -d $var_value ]]; then
            echo "[ERROR] directory does not exist: $var_value"
            exit 1
        fi
    elif [[ $file_type == "f" ]]; then
        if [[ ! -f $var_value ]]; then
            echo "[ERROR] file does not exist: $var_value"
            exit 1
        fi
    fi
}

# Check whether a string is in array. Echo 0 if it's found.
is_in_array() {
    local target=$1
    shift   # exclude $1 for next argument declaration
    local the_array=( "$@" )
    local found=1

    for _val in "${the_array[@]}"; do
        if [[ "$_val" == "$target" ]]; then
            found=0
            break
        fi
    done

    echo ${found}
}

# Normalize argument (just for toyc binaries):
# - For C++ CLI arguments, underscores will be replaced with hyphens
# - For Zig CLI arguments, hyphens will be replaced with underscores
norm_arg() {
    local cpp_or_zig=$1
    local the_arg=$2

    if [[ "$cpp_or_zig" == "cpp" ]]; then
        echo ${the_arg//_/-}
    elif [[ "$cpp_or_zig" == "zig" ]]; then
        echo ${the_arg//-/_}
    else
        echo "[ERROR] the first argument for norm_arg() must be one of [cpp, zig]"
        exit 1
    fi
}

log_debug() {
    # XXX: In case there are arrays to expand in the message, we should get the
    # input argument in this way.
    local msg=( "$@" )
    if [[ ${VERBOSE} == 1 ]]; then
        echo "${msg[@]}"
    fi
}

run_and_cmp() {
    local cpp_bin=$1
    local zig_bin=$2
    local input_file=$3
    local emit_type=$4

    # Exclude $1 ~ $4 for the remaining arguments
    for i in {1..4}; do
        shift
    done
    local other_args=( "$@" )

    # Normalize the emit type for C++ and Zig impls
    local emit_type_cpp=$(norm_arg cpp $emit_type)
    local emit_type_zig=$(norm_arg zig $emit_type)

    log_debug "========== Start =========="
    log_debug "> running cpp impl: $cpp_bin $input_file --emit=$emit_type_cpp ${other_args[@]}"
    "$cpp_bin" "$input_file" --emit=$emit_type_cpp "${other_args[@]}" > ${CPP_OUT} 2>&1 || {
        echo "[ERROR] failed to run cpp impl, command:"
        echo "$cpp_bin $input_file --emit=$emit_type_cpp ${other_args[@]}"
        exit 1
    }

    log_debug "> running zig impl: $zig_bin $input_file --emit=$emit_type_zig ${other_args[@]}"
    "$zig_bin" "$input_file" --emit=$emit_type_zig "${other_args[@]}" > ${ZIG_OUT} 2>&1 || {
        echo "[ERROR] failed to run zig impl, command:"
        echo "$zig_bin $input_file --emit=$emit_type_zig ${other_args[@]}"
        exit 1
    }

    log_debug "> comparing output"
    diff -q ${ZIG_OUT} ${CPP_OUT}

    if [[ $? != 0 ]]; then
        # TODO: show what chapter, file, args we failed now.
        echo "[ERROR] test failed"
        exit 1
    fi
}

run_and_xfail() {
    local cpp_bin=$1
    local zig_bin=$2
    local input_file=$3
    local emit_type=$4

    # Exclude $1 ~ $4 for the remaining arguments
    for i in {1..4}; do
        shift
    done
    local other_args=( "$@" )

    # Normalize the emit type for C++ and Zig impls
    local emit_type_cpp=$(norm_arg cpp $emit_type)
    local emit_type_zig=$(norm_arg zig $emit_type)

    log_debug "========== Start =========="
    # XXX: well... C++ implementation won't return a non-zero exit code if it
    # failed to process file. So we only test with our implementation for now.

    # log_debug "> running cpp impl: $cpp_bin $input_file --emit=$emit_type_cpp ${other_args[@]}"
    # set +e
    # "$cpp_bin" "$input_file" --emit=$emit_type_cpp "${other_args[@]}" > /dev/null 2>&1 && {
    #     echo "[ERROR] command is expected to fail, but it didn't:"
    #     echo "$cpp_bin $input_file --emit=$emit_type_cpp ${other_args[@]}"
    #     exit 1
    # }
    # set -e

    log_debug "> running zig impl: $zig_bin $input_file --emit=$emit_type_zig ${other_args[@]}"
    set +e
    "$zig_bin" "$input_file" --emit=$emit_type_zig "${other_args[@]}" > /dev/null 2>&1 && {
        echo "[ERROR] command is expected to fail, but it didn't:"
        echo "$zig_bin $input_file --emit=$emit_type_zig ${other_args[@]}"
        exit 1
    }
    set -e
}

# ==============================================================================
# Prepare arguments and do checks
check_var d DIR_TMP
check_var d DIR_TOY_BIN
check_var d DIR_TOY_EXAMPLES
check_var d DIR_ZIG_BIN

if [[ -z ${SELECTED_CHAPTER_NUM} ]]; then
    echo "[ERROR] no chapter is selected"
    exit 1
fi

sel_ch="Ch${SELECTED_CHAPTER_NUM}"
sel_bin="toyc-ch${SELECTED_CHAPTER_NUM}"

if [[ $( is_in_array $sel_ch "${CHAPTERS[@]}" ) == 1 ]]; then
    echo "[ERROR] chapter $sel_ch is not available"
    exit 1
fi

name_files="${sel_ch}_FILES"
declare -n files="$name_files"

name_files_xfail="${sel_ch}_FILES_XFAIL"
declare -n files_xfail="$name_files_xfail"

name_other_args="${sel_ch}_OTHER_ARGS"
declare -n other_args="$name_other_args"

name_emit_types="${sel_ch}_EMIT_TYPES"
declare -n emit_types="$name_emit_types"

if [[ $ENABLE_DEBUG == 1 ]]; then
    other_args+=( "${DEBUG_OPTIONS[@]}" )
fi

echo "========== Running test for ${sel_ch} =========="
for _file in "${files[@]}"; do
    input_file="${DIR_TOY_EXAMPLES}/${sel_ch}/${_file}"
    cpp_bin="${DIR_TOY_BIN}/${sel_bin}"
    zig_bin="${DIR_ZIG_BIN}/${sel_bin}"

    check_var f input_file
    check_var f cpp_bin
    check_var f zig_bin

    input_type=${_file##*.}

    for _emit_type in "${emit_types[@]}"; do
        # AST can only be generated when input is written in toy language
        if [[ $_emit_type == "ast" ]] && [[ $input_type != "toy" ]]; then
            continue
        fi
        run_and_cmp "$cpp_bin" "$zig_bin" "$input_file" "$_emit_type" ${other_args[@]}
    done
done

for _file in "${files_xfail[@]}"; do
    input_file="${DIR_TOY_EXAMPLES}/${sel_ch}/${_file}"
    cpp_bin="${DIR_TOY_BIN}/${sel_bin}"
    zig_bin="${DIR_ZIG_BIN}/${sel_bin}"

    check_var f input_file
    check_var f cpp_bin
    check_var f zig_bin

    input_type=${_file##*.}

    for _emit_type in "${emit_types[@]}"; do
        # AST can only be generated when input is written in toy language
        if [[ $_emit_type == "ast" ]] && [[ $input_type != "toy" ]]; then
            continue
        fi
        run_and_xfail "$cpp_bin" "$zig_bin" "$input_file" "$_emit_type" ${other_args[@]}
    done
done

echo "========== Done =========="
