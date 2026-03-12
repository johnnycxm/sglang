from __future__ import annotations

from sglang.multimodal_gen.test.server.testcase_configs import (
    T2V_PROMPT,
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
    T2I_sampling_params,
)

ONE_GPU_MUSA_CASES_A: list[DiffusionTestCase] = [
    DiffusionTestCase(
        "qwen_image_t2i_musa",
        DiffusionServerArgs(
            model_path="Qwen/Qwen-Image",
            modality="image",
        ),
        T2I_sampling_params,
    ),
]


ONE_GPU_MUSA_CASES_B: list[DiffusionTestCase] = [
    DiffusionTestCase(
        "wan2_1_t2v_1.3b_musa",
        DiffusionServerArgs(
            model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            modality="video",
            custom_validator="video",
        ),
        DiffusionSamplingParams(
            prompt=T2V_PROMPT,
        ),
    ),
]


TWO_GPU_MUSA_CASES_A: list[DiffusionTestCase] = [
    DiffusionTestCase(
        "wan2_2_t2v_a14b_2gpu_musa",
        DiffusionServerArgs(
            model_path="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            modality="video",
            custom_validator="video",
            num_gpus=2,
        ),
        DiffusionSamplingParams(
            prompt=T2V_PROMPT,
        ),
    ),
]
