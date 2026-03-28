# Dockerfile for PD experiments
#
# Build:  bash pd_exp/docker-build.sh
# Run:    see run_bench.md
#
# Base image is built from rucnyz/vllm pd_competition branch,
# so PD scheduler code is already compiled in.
# To update: bash pd_exp/docker-build.sh --exp (fast, ~1min)

ARG BASE_IMAGE=vllm-openai
FROM ${BASE_IMAGE}

RUN pip install --no-cache-dir --ignore-installed blinker && \
    pip install --no-cache-dir matplotlib numpy transformers datasets quart

COPY pd_exp/ /workspace/pd_exp/
COPY benchmarks/ /workspace/benchmarks/

RUN chmod +x /workspace/pd_exp/**/*.sh 2>/dev/null || true

WORKDIR /workspace
ENTRYPOINT []
CMD ["/bin/bash"]
