#!/usr/bin/env bash
SUDO_GID=$(id -g)
SUDO_UID=$(id -u)

DIR_RAMDISK=/ramdisk
CACHE_DIR_NAME=mlir_toy

if [[ ! -d ${DIR_RAMDISK} ]]; then
    echo "[ERROR] RAM disk ${DIR_RAMDISK} does not exist"
    exit 1
fi

cd ${DIR_RAMDISK}
sudo mkdir -p ${CACHE_DIR_NAME}
sudo chown ${SUDO_UID}:${SUDO_GID} ${CACHE_DIR_NAME}

