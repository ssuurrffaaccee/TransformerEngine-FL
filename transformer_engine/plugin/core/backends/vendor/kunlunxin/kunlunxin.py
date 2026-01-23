# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ...ops import TEFLBackendBase, FP8TensorMeta, NVTE_Fused_Attn_Backend


class KunLunXinBackend(TEFLBackendBase):
    @staticmethod
    def check_available() -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def get_flash_attention_class(self):
        from .flash_attention import FlashAttentionTorch
        return FlashAttentionTorch

    def get_attention_backend(self, attention_params=None):
        from packaging.version import Version as PkgVersion
        from ...logger_manager import get_logger
        logger = get_logger()

        # Read environment variables to determine which backends to enable
        use_flash_attention = int(os.getenv("NVTE_FLASH_ATTN", "1"))
        use_fused_attention = int(os.getenv("NVTE_FUSED_ATTN", "1"))
        use_unfused_attention = int(os.getenv("NVTE_UNFUSED_ATTN", "1"))

        # Log disabled backends
        if not use_flash_attention:
            logger.info_once("Disabling FlashAttention due to NVTE_FLASH_ATTN=0")
        if not use_fused_attention:
            logger.info_once("Disabling FusedAttention due to NVTE_FUSED_ATTN=0")
        if not use_unfused_attention:
            logger.info_once("Disabling UnfusedDotProductAttention due to NVTE_UNFUSED_ATTN=0")

        flash_attention_backend = PkgVersion("2.6.0") if use_flash_attention else None
        fused_attention_backend = NVTE_Fused_Attn_Backend.NVTE_No_Backend

        available_backends = [use_flash_attention, use_fused_attention, use_unfused_attention]

        return (
            use_flash_attention,
            flash_attention_backend,
            use_fused_attention,
            fused_attention_backend,
            use_unfused_attention,
            available_backends,
        )

    def generic_gemm(
        self,
        A: torch.Tensor,
        transA: bool,
        B: torch.Tensor,
        transB: bool,
        D: torch.Tensor,
        quantizer: Any,
        output_dtype: torch.dtype,
        bias: Optional[torch.Tensor],
        bias_type: Any,
        gelu: bool,
        gelu_in: Optional[torch.Tensor],
        grad: bool,
        workspace: torch.Tensor,
        workspace_size: int,
        accumulate: bool,
        use_split_accumulator: bool,
        comm_overlap: Optional[Any] = None,
        comm_type: Optional[Any] = None,
        extra_output: Optional[torch.Tensor] = None,
        bulk_overlap: bool = False,
        alpha: float = 1.0,
        beta: Optional[float] = None,
    ) -> Any:
        raise NotImplementedError("generic_gemm - not implemented in reference backend")

    def te_general_grouped_gemm(self, *args, **kwargs) -> Any:
        raise NotImplementedError("te_general_grouped_gemm - not implemented in reference backend")

    def quantize(self, tensor: torch.Tensor, quantizer: Any, output: Optional[torch.Tensor] = None, noop: Optional[torch.Tensor] = None) -> Any:
        raise NotImplementedError("quantize - not implemented in reference backend")

    def dequantize(self, input: torch.Tensor, otype: torch.dtype) -> torch.Tensor:
        raise NotImplementedError("dequantize - not implemented in reference backend")

    def bgrad_quantize(self, input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError("bgrad_quantize - not implemented in reference backend")

    def gelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("gelu - not implemented in reference backend")

    def geglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("geglu - not implemented in reference backend")

    def qgelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("qgelu - not implemented in reference backend")

    def qgeglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("qgeglu - not implemented in reference backend")

    def relu(self, input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("relu - not implemented in reference backend") 

    def reglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("reglu - not implemented in reference backend")

    def srelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("srelu - not implemented in reference backend")

    def sreglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("sreglu - not implemented in reference backend")

    def silu(self, input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("silu - not implemented in reference backend")

    def swiglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("swiglu - not implemented in reference backend")

    def clamped_swiglu(self, input: torch.Tensor, quantizer: Any, limit: float = 7.0, alpha: float = 1.702) -> Any:
        raise NotImplementedError("clamped_swiglu - not implemented in reference backend")

    def dgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("dgelu - not implemented in reference backend")

    def dgeglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("dgeglu - not implemented in reference backend")

    def dqgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("dqgelu - not implemented in reference backend")

    def dqgeglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("dqgeglu - not implemented in reference backend")

    def drelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("drelu - not implemented in reference backend")

    def dreglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("dreglu - not implemented in reference backend")

    def dsrelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("dsrelu - not implemented in reference backend")

    def dsreglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("dsreglu - not implemented in reference backend")

    def dsilu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("dsilu - not implemented in reference backend")

    def dswiglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("dsilu - not implemented in reference backend")

    def clamped_dswiglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any, limit: float = 7.0, alpha: float = 1.702) -> Any:
        raise NotImplementedError("clamped_dswiglu - not implemented in reference backend")

    def dbias_dgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError("dbias_dgelu - not implemented in reference backend")

    def dbias_dsilu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError("dbias_dsilu - not implemented in reference backend")

    def dbias_drelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError("dbias_drelu - not implemented in reference backend")

    def dbias_dqgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError("dbias_dqgelu - not implemented in reference backend")

    def dbias_dsrelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError("dbias_dsrelu - not implemented in reference backend")

    def layernorm_fwd(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        eps: float,
        ln_out: Optional[torch.Tensor],
        quantizer: Any,
        otype: torch.dtype,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError("layernorm_fwd - not implemented in reference backend")

    def layernorm_bwd(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        rsigma: torch.Tensor,
        gamma: torch.Tensor,
        sm_margin: int = 0,
        zero_centered_gamma: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError("layernorm_bwd - not implemented in reference backend")

    def rmsnorm_fwd(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        ln_out: Optional[torch.Tensor],
        quantizer: Any,
        otype: torch.dtype,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        raise NotImplementedError("rmsnorm_fwd - not implemented in reference backend")

    def rmsnorm_bwd(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        rsigma: torch.Tensor,
        gamma: torch.Tensor,
        sm_margin: int = 0,
        zero_centered_gamma: bool = False,
        eps: float = 1e-5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("rmsnorm_bwd - not implemented in reference backend")

    def rmsnorm_bwd_add(self, *args, **kwargs) -> Any:
        raise NotImplementedError("rmsnorm_bwd_add - not implemented in reference backend")

    def multi_tensor_quantize(self, tensor_list: List[torch.Tensor], quantizer_list: List[Any]) -> List[Any]:
        raise NotImplementedError("multi_tensor_quantize - not implemented in reference backend")

    def split_quantize(self, tensor: torch.Tensor, split_sections: List[int], quantizer_list: List[Any]) -> List[Any]:
        raise NotImplementedError("split_quantize - not implemented in reference backend")

    def moe_permute_fwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("moe_permute_fwd - not implemented in reference backend")

    def moe_permute_bwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("moe_permute_bwd - not implemented in reference backend")

    def moe_unpermute_fwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("moe_unpermute_fwd - not implemented in reference backend")

    def moe_unpermute_bwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("moe_unpermute_bwd - not implemented in reference backend")

    def scaled_softmax_forward(self, input: torch.Tensor, scale: float) -> torch.Tensor:
        raise NotImplementedError("scaled_softmax_forward - not implemented in reference backend")

    def scaled_softmax_backward(self, output_grad: torch.Tensor, softmax_output: torch.Tensor, scale: float) -> torch.Tensor:
        raise NotImplementedError("scaled_softmax_backward - not implemented in reference backend")

    def scaled_masked_softmax_forward(self, input: torch.Tensor, mask: torch.Tensor, scale: float) -> torch.Tensor:
        raise NotImplementedError("scaled_masked_softmax_forward - not implemented in reference backend")

    def scaled_masked_softmax_backward(self, output_grad: torch.Tensor, softmax_output: torch.Tensor, scale: float) -> torch.Tensor:
        raise NotImplementedError("scaled_masked_softmax_backward - not implemented in reference backend")

    def scaled_upper_triang_masked_softmax_forward(self, input: torch.Tensor, scale: float) -> torch.Tensor:
        raise NotImplementedError("scaled_upper_triang_masked_softmax_forward - not implemented in reference backend")

    def scaled_upper_triang_masked_softmax_backward(self, output_grad: torch.Tensor, softmax_output: torch.Tensor, scale: float) -> torch.Tensor:
        raise NotImplementedError("scaled_upper_triang_masked_softmax_backward - not implemented in reference backend")

    def scaled_aligned_causal_masked_softmax_forward(self, input: torch.Tensor, scale: float) -> torch.Tensor:
        raise NotImplementedError("scaled_aligned_causal_masked_softmax_forward - not implemented in reference backend")

    def scaled_aligned_causal_masked_softmax_backward(self, output_grad: torch.Tensor, softmax_output: torch.Tensor, scale: float) -> torch.Tensor:
        raise NotImplementedError("scaled_aligned_causal_masked_softmax_backward - not implemented in reference backend")

    def get_fused_attn_backend(self, *args, **kwargs) -> int:
        return NVTE_Fused_Attn_Backend.NVTE_No_Backend

    def fused_attn_fwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_attn_fwd - not implemented in reference backend")

    def fused_attn_bwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_attn_bwd - not implemented in reference backend")

    def fa_prepare_fwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fa_prepare_fwd - not implemented in reference backend")

    def fa_prepare_bwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fa_prepare_bwd - not implemented in reference backend")

    def copy_to_kv_cache(self, *args, **kwargs) -> Any:
        raise NotImplementedError("copy_to_kv_cache - not implemented in reference backend")

    def convert_thd_to_bshd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("convert_thd_to_bshd - not implemented in reference backend")

    def convert_bshd_to_thd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("convert_bshd_to_thd - not implemented in reference backend")

    def fused_rope_forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_rope_forward - not implemented in reference backend")

    def fused_rope_backward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_rope_backward - not implemented in reference backend")

    def fused_qkv_rope_forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_qkv_rope_forward - not implemented in reference backend")

    def fused_qkv_rope_backward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_qkv_rope_backward - not implemented in reference backend")

    def fused_topk_with_score_function_fwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_topk_with_score_function_fwd - not implemented in reference backend")

    def fused_topk_with_score_function_bwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_topk_with_score_function_bwd - not implemented in reference backend")

    def fused_score_for_moe_aux_loss_fwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_score_for_moe_aux_loss_fwd - not implemented in reference backend")

    def fused_score_for_moe_aux_loss_bwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_score_for_moe_aux_loss_bwd - not implemented in reference backend")

    def fused_moe_aux_loss_fwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_moe_aux_loss_fwd - not implemented in reference backend")

    def fused_moe_aux_loss_bwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_moe_aux_loss_bwd - not implemented in reference backend")

    def dropout_fwd(self, input: torch.Tensor, dropout_probability: float, out: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("dropout_fwd - not implemented in reference backend")

    def dropout_bwd(self, grad_output: torch.Tensor, mask: torch.Tensor, dropout_probability: float, grad_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError("dropout_bwd - not implemented in reference backend")

    def fp8_transpose(self, input: torch.Tensor, dtype: Any, *, out: torch.Tensor) -> None:
        raise NotImplementedError("fp8_transpose - not implemented in reference backend")

    def swap_first_dims(self, tensor: torch.Tensor, *, out: torch.Tensor) -> None:
        raise NotImplementedError("swap_first_dims - not implemented in reference backend")

    def compute_amax(self, input: torch.Tensor, amax: torch.Tensor) -> None:
        raise NotImplementedError("compute_amax - not implemented in reference backend")

    def fused_amax_and_scale_update_after_reduction(self, *args, **kwargs) -> None:
        raise NotImplementedError("fused_amax_and_scale_update_after_reduction - not implemented in reference backend")

    def fp8_block_scaling_compute_partial_amax(self, *args, **kwargs) -> None:
        raise NotImplementedError("fp8_block_scaling_compute_partial_amax - not implemented in reference backend")

    def fp8_block_scaling_partial_cast(self, *args, **kwargs) -> None:
        raise NotImplementedError("fp8_block_scaling_partial_cast - not implemented in reference backend")

    def fused_multi_row_padding(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_multi_row_padding - not implemented in reference backend")

    def fused_multi_row_unpadding(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_multi_row_unpadding - not implemented in reference backend")

    def get_cublasLt_version(self) -> int:
        return 0

    def get_cudnn_version(self) -> int:
        return 0

    def get_num_cublas_streams(self) -> int:
        return 0

    def thd_read_half_tensor(self, *args, **kwargs) -> Any:
        raise NotImplementedError("thd_read_half_tensor - not implemented in reference backend")

    def thd_second_half_lse_correction(self, *args, **kwargs) -> Any:
        raise NotImplementedError("thd_second_half_lse_correction - not implemented in reference backend")

    def thd_read_second_half_lse(self, *args, **kwargs) -> Any:
        raise NotImplementedError("thd_read_second_half_lse - not implemented in reference backend")

    def thd_out_correction(self, *args, **kwargs) -> Any:
        raise NotImplementedError("thd_out_correction - not implemented in reference backend")

    def thd_grad_correction(self, *args, **kwargs) -> Any:
        raise NotImplementedError("thd_grad_correction - not implemented in reference backend")

    def thd_get_partitioned_indices(self, *args, **kwargs) -> Any:
        raise NotImplementedError("thd_get_partitioned_indices - not implemented in reference backend")

    def init_nvshmem_backend(self, *args, **kwargs) -> None:
        raise NotImplementedError("init_nvshmem_backend - not implemented in reference backend")

    def create_nvshmem_tensor(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("create_nvshmem_tensor - not implemented in reference backend")

    def nvshmem_send_on_current_stream(self, *args, **kwargs) -> None:
        raise NotImplementedError("nvshmem_send_on_current_stream - not implemented in reference backend")

    def nvshmem_wait_on_current_stream(self, *args, **kwargs) -> None:
        raise NotImplementedError("nvshmem_wait_on_current_stream - not implemented in reference backend")

    def nvshmem_finalize(self) -> None:
        raise NotImplementedError("nvshmem_finalize - not implemented in reference backend")

    def multi_tensor_scale(self, chunk_size: int, noop_flag: torch.Tensor, tensor_lists: List[List[torch.Tensor]], scale: float) -> None:
        raise NotImplementedError("multi_tensor_scale - not implemented in reference backend")

    def multi_tensor_l2norm(self, chunk_size: int, noop_flag: torch.Tensor, tensor_lists: List[List[torch.Tensor]], per_tensor: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        raise NotImplementedError("multi_tensor_l2norm - not implemented in reference backend")

    def multi_tensor_unscale_l2norm(self, chunk_size: int, noop_flag: torch.Tensor, tensor_lists: List[List[torch.Tensor]], scale: torch.Tensor, per_tensor: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        raise NotImplementedError("multi_tensor_unscale_l2norm - not implemented in reference backend")

    def multi_tensor_adam(self, *args, **kwargs):
        raise NotImplementedError("multi_tensor_adam - not implemented in reference backend")

    def multi_tensor_adam_param_remainder(self, *args, **kwargs) -> None:
        raise NotImplementedError("multi_tensor_adam_param_remainder - not implemented in reference backend")

    def multi_tensor_adam_fp8(self, *args, **kwargs) -> None:
        raise NotImplementedError("multi_tensor_adam_fp8 - not implemented in reference backend")

    def multi_tensor_adam_capturable(self, *args, **kwargs) -> None:
        raise NotImplementedError("multi_tensor_adam_capturable - not implemented in reference backend")

    def multi_tensor_adam_capturable_master(self, *args, **kwargs) -> None:
        raise NotImplementedError("multi_tensor_adam_capturable_master - not implemented in reference backend")

    def multi_tensor_sgd(self, *args, **kwargs) -> None:
        raise NotImplementedError("multi_tensor_sgd - not implemented in reference backend")

    def multi_tensor_compute_scale_and_scale_inv(self, *args, **kwargs) -> None:
        raise NotImplementedError("multi_tensor_compute_scale_and_scale_inv - not implemented in reference backend")

    def bulk_overlap_ag_with_external_gemm(self, *args, **kwargs) -> Any:
        raise NotImplementedError("bulk_overlap_ag_with_external_gemm - not implemented in reference backend")

    def create_fp8_tensor_meta(self) -> FP8TensorMeta:
        return FP8TensorMeta()

    def create_comm_overlap_helper(self, *args, **kwargs) -> Any:
        raise NotImplementedError("create_comm_overlap_helper - not implemented in reference backend")

    def create_comm_overlap(self, *args, **kwargs) -> Any:
        raise NotImplementedError("create_comm_overlap - not implemented in reference backend")

    def create_comm_overlap_p2p(self, *args, **kwargs) -> Any:
        raise NotImplementedError("create_comm_overlap_p2p - not implemented in reference backend")
