# Dockerfile for PD experiments
#
# Build (from repo root):
#   # Step 1: build vLLM base image (slow, cached after first build)
#   docker build -f docker/Dockerfile --target vllm-openai -t vllm-openai .
#
#   # Step 2: add experiment layer (fast)
#   docker build -f pd_exp/Dockerfile -t vllm-pd-exp .
#
# Run:
#   docker run --gpus '"device=4"' --rm -it \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     vllm-pd-exp bash
#
# Or use: bash pd_exp/docker-build.sh

ARG BASE_IMAGE=vllm-openai
FROM ${BASE_IMAGE}

# Install extra Python deps for experiments
RUN pip install --no-cache-dir \
    matplotlib \
    numpy \
    transformers \
    datasets

# Copy experiment scripts
COPY pd_exp/ /workspace/pd_exp/

WORKDIR /workspace

# Make scripts executable
RUN chmod +x /workspace/pd_exp/serve/*.sh /workspace/pd_exp/common.sh \
    /workspace/pd_exp/*.sh 2>/dev/null || true

# Override vllm-openai's ENTRYPOINT ["vllm", "serve"] so we can run bash
ENTRYPOINT []
CMD ["/bin/bash"]
