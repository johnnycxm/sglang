# Adaptive AllReduce Configuration System

Automatically selects the optimal allreduce backend (FlashInfer Fusion, Torch Symmetric Memory, Custom AllReduce, or NCCL) based on batch size.

## Supported Backends

| Backend | Best For | Source Code |
|---------|----------|-------------|
| FlashInfer Fusion | Small batch (≤128) | [flashinfer_comm_fusion.py](../flashinfer_comm_fusion.py) |
| Torch Symmetric Memory | Large batch (>128), SM90+ | [torch_symm_mem.py](../../../../distributed/device_communicators/torch_symm_mem.py) |
| Custom AllReduce | NVLink GPUs | [custom_all_reduce.py](../../../../distributed/device_communicators/custom_all_reduce.py) |
| NCCL | General fallback | [parallel_state.py](../../../../distributed/parallel_state.py) |

## Usage

### Step 1: Run Tuning

```bash
# Using model name
torchrun --nproc_per_node=2 \
    python/sglang/srt/layers/allreduce/tuning_allreduce_config.py \
    --model nvidia/DeepSeek-V3-0324-FP4 \
    --tp-size 2

# Or specify hidden_size directly
torchrun --nproc_per_node=2 \
    python/sglang/srt/layers/allreduce/tuning_allreduce_config.py \
    --hidden-size 7168 \
    --tp-size 2
```

### Step 2: Enable

```bash
python -m sglang.launch_server \
    --model-path nvidia/DeepSeek-V3-0324-FP4 \
    --tp 2 \
    --enable-adaptive-allreduce
```

Config files are saved in `configs/` directory with format: `allreduce_config_hidden={hidden_size},device={device_name}.json`

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model name (auto-detect hidden_size) | None |
| `--hidden-size` | Hidden dimension | None |
| `--tp-size` | Tensor parallel size | 2 |
| `--batch-sizes` | Batch sizes to tune | range(1,3)+range(32,512,8) |
| `--output-dir` | Config output directory | ./configs/ |

## Config File Example

```json
{
    "1": {
        "backend_type": "flashinfer_fusion",
        "use_residual_rmsnorm_fusion": true,
        "backend_name": "flashinfer_fusion"
    },
    "128": {
        "backend_type": "flashinfer_fusion",
        "use_residual_rmsnorm_fusion": true,
        "backend_name": "flashinfer_fusion"
    },
    "256": {
        "backend_type": "torch_symm_mem",
        "use_residual_rmsnorm_fusion": false,
        "backend_name": "torch_symm_mem"
    }
}
```
