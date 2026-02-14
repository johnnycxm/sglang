#!/bin/bash
set -euo pipefail

HOSTNAME_VALUE=$(hostname)
GPU_ARCH="mp_31"   # default placeholder; will be overridden by hostname if matches
OPTIONAL_DEPS="${1:-}"

# Build python extras for MUSA (not HIP)
EXTRAS="dev_musa"
if [ -n "$OPTIONAL_DEPS" ]; then
    EXTRAS="dev_musa,${OPTIONAL_DEPS}"
fi
echo "Installing python extras: [${EXTRAS}]"

# Parse GPU architecture from hostname (e.g., linux-mi35x-gpu-1-... -> mi35x)
if [[ "${HOSTNAME_VALUE}" =~ ^linux-(mi[0-9]+[a-z]*)-gpu-[0-9]+ ]]; then
  GPU_ARCH="${BASH_REMATCH[1]}"
  echo "Detected GPU architecture from hostname: ${GPU_ARCH}"
else
  echo "Warning: could not parse GPU architecture from '${HOSTNAME_VALUE}', defaulting to ${GPU_ARCH}"
fi

# Install required dependencies in CI container
CONTAINER="fxy-sglang"
SG_LANG_DIR="/sgl-workspace/sglang"

# Fix pip cache permissions (safe to ignore errors)
docker exec "${CONTAINER}" chown -R root:root /sgl-data/pip-cache 2>/dev/null || true
docker exec "${CONTAINER}" pip install --cache-dir=/sgl-data/pip-cache --upgrade pip

# Uninstall existing packages to avoid conflicts
docker exec "${CONTAINER}" pip uninstall sgl-kernel sglang -y || true

# Clear Python bytecode caches
for dir in /opt/venv "${SG_LANG_DIR}"; do
    docker exec "${CONTAINER}" find "${dir}" -name "*.pyc" -delete 2>/dev/null || true
    docker exec "${CONTAINER}" find "${dir}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
done

# Clone or ensure sglang is present (assuming it's already mounted at /sglang-checkout)
# If not, you may need to git clone here — but your context suggests it's pre-mounted.

# Step 1: Build and install sgl-kernel with MUSA
echo "Building sgl-kernel for MUSA with MTGPU_TARGET=${GPU_ARCH}..."
docker exec -w "${SG_LANG_DIR}/sgl-kernel" "${CONTAINER}" bash -c "
    MTGPU_TARGET='${GPU_ARCH}' python3 setup_musa.py install
"

# Step 2: Patch musa-constraints.txt to use torch==2.7.1
echo "Patching musa-constraints.txt to use torch==2.7.1..."
docker exec "${CONTAINER}" sed -i 's/torch==2\.5\.0/torch==2.7.1/g' "${SG_LANG_DIR}/musa-constraints.txt"

# Step 3: Install sglang main package with MUSA extras
BUILD_TYPE="${BUILD_TYPE:-all}"  # default to 'all' if not set
if [ "${BUILD_TYPE}" = "srt" ]; then
    EXTRA_FLAG="srt_musa"
else
    EXTRA_FLAG="all_musa"
fi
echo "Installing sglang with extra: [${EXTRA_FLAG}]..."

docker exec -w "${SG_LANG_DIR}" "${CONTAINER}" pip install \
    --no-cache-dir \
    -e "python[${EXTRA_FLAG}]" \
    --constraint musa-constraints.txt

# Step 4: Build and install sgl-router
echo "Building and installing sgl-router..."
docker exec -w "${SG_LANG_DIR}/sgl-router" "${CONTAINER}" bash -c "
    python -m build && python -m pip install --force-reinstall dist/*.whl
"

# Step 5: Clean up pynvml (conflicts with MUSA environment)
echo "Uninstalling pynvml (not needed for MUSA)..."
docker exec "${CONTAINER}" pip uninstall pynvml -y || true

echo "SGLang and sgl-kernel installed successfully for MUSA (${GPU_ARCH})"
