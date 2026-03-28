# Dockerfile for PD experiments
#
# Build:  bash pd_exp/docker-build.sh
# Run:    see run_bench.md

ARG BASE_IMAGE=vllm-openai
FROM ${BASE_IMAGE}

# Install extra Python deps for experiments
RUN pip install --no-cache-dir --ignore-installed blinker && \
    pip install --no-cache-dir \
    matplotlib \
    numpy \
    transformers \
    datasets \
    quart

# Install git (needed for auto-update on startup)
RUN apt-get update -y && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Clone the repo (pd_exp + benchmarks)
RUN git clone --branch pd_competition --single-branch \
    https://github.com/rucnyz/vllm.git /workspace/repo

# Symlink pd_exp and benchmarks into workspace
RUN ln -s /workspace/repo/pd_exp /workspace/pd_exp && \
    ln -s /workspace/repo/benchmarks /workspace/benchmarks

WORKDIR /workspace

# Entrypoint: git pull then exec user command
RUN printf '#!/bin/bash\ncd /workspace/repo && git pull --ff-only origin pd_competition 2>/dev/null || true\nexec "$@"\n' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
