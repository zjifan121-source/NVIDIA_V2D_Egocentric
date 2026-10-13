# SPDX-License-Identifier: Apache-2.0
# The published afe7dad image predates the checkout's TensorRT backend.
# Preserve its CUDA/torch/native extensions; add only the baseline-pinned TRT runtime.
FROM ghcr.io/ryanzhou123232/v2d_foundation_pose@sha256:56980010c908ebac73ee168bf0087837520a192b2bfd4cbec2f08804870b209c
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer-bin=10.7.0.23-1+cuda12.6 \
    libnvinfer10=10.7.0.23-1+cuda12.6 \
    libnvinfer-lean10=10.7.0.23-1+cuda12.6 \
    libnvinfer-plugin10=10.7.0.23-1+cuda12.6 \
    libnvinfer-vc-plugin10=10.7.0.23-1+cuda12.6 \
    libnvinfer-dispatch10=10.7.0.23-1+cuda12.6 \
    libnvonnxparsers10=10.7.0.23-1+cuda12.6 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir 'tensorrt-cu12==10.7.0.post1' \
    && python -c 'import tensorrt; assert tensorrt.__version__.startswith("10.7.")' \
    && test -x /usr/src/tensorrt/bin/trtexec
