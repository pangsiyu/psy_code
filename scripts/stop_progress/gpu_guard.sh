#!/bin/bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

IFS=',' read -ra _STOP_PROGRESS_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
for _gpu in "${_STOP_PROGRESS_GPUS[@]}"; do
  _gpu="${_gpu//[[:space:]]/}"
  if [ "${_gpu}" = "3" ] || [ "${_gpu}" = "7" ]; then
    echo "Stop Progression experiments must not use physical GPUs 3 or 7. Current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
  fi
done
unset _gpu _STOP_PROGRESS_GPUS
