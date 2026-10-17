from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.layers.attention.dsa.kpool_plan import (
    init_kpool_extend_metadata,
    init_kpool_write_plan,
    init_kpool_write_plan_capture,
    init_pooled_paged_mqa_metadata,
    update_kpool_write_plan,
    update_pooled_paged_mqa_metadata,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import (
        PrecomputedMetadata,
    )
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
    from sglang.srt.layers.attention.dsa_backend import _DSA_IMPL_T, DSAMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


@dataclass
class _KPoolForwardInputs:
    full_real_page_table: Optional[torch.Tensor] = None
    full_seqlens_expanded: Optional[torch.Tensor] = None


class DeepseekSparseAttnBackendKPoolMixin:
    """KPool-specific metadata and tail handling for the DSA backend."""

    def _check_kpool_tail_backend(
        self,
        topk_indices: Optional[torch.Tensor],
        dsa_impl: _DSA_IMPL_T,
        phase: str,
        tail_appended: bool = False,
    ) -> None:
        if (
            topk_indices is None
            or self.dsa_index_kpool <= 1
            or dsa_impl in ("fa3", "tilelang", "trtllm")
        ):
            return
        # intel_xpu is allowed only on call sites that appended the
        # always-select kpool tail to topk_indices first (tail_appended=True);
        # every other non-fused backend still raises so the tail is never
        # silently dropped.
        if dsa_impl == "intel_xpu" and tail_appended:
            return
        raise NotImplementedError(
            "index_kpool > 1 appends tail tokens to topk_indices and is "
            f"currently only supported by the FA3/TileLang/TRTLLM DSA {phase} "
            "backend."
        )

    def _append_kpool_tail_intel_xpu(
        self,
        topk_indices: Optional[torch.Tensor],
        dsa_impl: _DSA_IMPL_T,
        metadata: DSAMetadata,
        forward_batch: ForwardBatch,
    ) -> Tuple[Optional[torch.Tensor], bool]:
        """Append the always-select kpool tail for the intel_xpu (unfused) path.

        Mirrors the fused CUDA path in
        ``kpool_fp8_index.topk_from_pooled_history_logits``: the tail tokens
        (the last ``seq_len % pool_size`` tokens of each row, which are not yet
        compressed into a pool) are appended after the pooled top-k history so
        downstream sparse attention sees ``min(pool_tokens, topk) + tail``
        entries. The indices stay in request-local token positions; the caller
        maps them to physical slots exactly like the unfused PAGED/RAGGED
        transforms do.

        Returns ``(topk_indices, tail_appended)``; ``tail_appended`` is False
        when nothing was appended (other impls, unsupported speculative phases,
        or indices that already carry the tail).
        """
        if (
            topk_indices is None
            or dsa_impl != "intel_xpu"
            or self.dsa_index_kpool <= 1
        ):
            return topk_indices, False

        pool_size = self.dsa_index_kpool
        expanded_width = self.dsa_index_topk + pool_size - 1
        if topk_indices.shape[1] == expanded_width:
            # Already expanded: either a fused indexer path appended the tail,
            # or the short-sequence full-topk path widened the result with -1
            # padding (full coverage already includes the tail tokens).
            return topk_indices, True
        assert topk_indices.shape[1] == self.dsa_index_topk, (
            f"unexpected topk_indices width {topk_indices.shape[1]} for "
            f"index_topk={self.dsa_index_topk}, index_kpool={pool_size}"
        )

        from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
            append_kpool_tail_to_topk,
        )

        forward_mode = forward_batch.forward_mode
        if forward_mode.is_extend_without_speculative():
            # Per-token sequence lengths (position + 1) for the extend rows.
            plan = metadata.kpool_extend_plan
            seq_lens = (
                plan.seq_lens_expanded
                if plan is not None
                else metadata.dsa_seqlens_expanded
            )
            assert seq_lens is not None and seq_lens.ndim == 1, (
                "intel_xpu kpool prefill requires per-token sequence lengths "
                "(kpool_extend_plan.seq_lens_expanded or dsa_seqlens_expanded)"
            )
        elif forward_mode.is_decode_or_idle():
            # One query row per request; cache_seqlens includes the token
            # whose KV was just written, which the tail must select.
            seq_lens = metadata.cache_seqlens_int32[: topk_indices.shape[0]]
        elif forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2():
            # glm53-b70: one query row per draft token; dsa_seqlens_expanded holds each row's causal length
            # (position + 1), which is exactly what the tail selection needs.
            seq_lens = metadata.dsa_seqlens_expanded
            if seq_lens is None or seq_lens.ndim != 1:
                return topk_indices, False
        else:
            return topk_indices, False

        seq_lens = seq_lens.to(torch.int32)
        n_rows = topk_indices.shape[0]
        if seq_lens.shape[0] < n_rows:
            # Indexer output was padded to the padded-token count; padded
            # rows carry all -1 indices, and seq_len 0 keeps the appended
            # tail at -1 for them.
            seq_lens = torch.cat(
                [
                    seq_lens,
                    torch.zeros(
                        n_rows - seq_lens.shape[0],
                        dtype=torch.int32,
                        device=seq_lens.device,
                    ),
                ]
            )
        pool_lens = torch.div(seq_lens, pool_size, rounding_mode="floor")
        topk_indices = append_kpool_tail_to_topk(
            topk_indices,
            seq_lens=seq_lens,
            pool_lens=pool_lens,
            pool_size=pool_size,
        )
        return topk_indices, True

    def _resolve_kpool_tail_backend(
        self,
        topk_indices: Optional[torch.Tensor],
        dsa_impl: _DSA_IMPL_T,
    ) -> _DSA_IMPL_T:
        if (
            topk_indices is None
            or self.dsa_index_kpool <= 1
            or dsa_impl != "flashmla_sparse"
        ):
            return dsa_impl
        if self.device_sm_major >= 10:
            return "trtllm"
        if self.device_sm_major == 9:
            return "fa3"
        return dsa_impl

    def _kpool_slots_per_page(self) -> int:
        return getattr(self.token_to_kv_pool, "slots_per_page", self.real_page_size)

    def _build_kpool_paged_mqa_schedule_metadata(self) -> bool:
        if self.device_sm_major == 9:
            return self.num_q_heads in (32, 64)
        return True

    def _init_kpool_metadata(
        self,
        metadata: DSAMetadata,
        forward_batch: ForwardBatch,
        topk_transform_method: Optional[TopkTransformMethod] = None,
        kpool_inputs: Optional[_KPoolForwardInputs] = None,
    ) -> DSAMetadata:
        if self.dsa_index_kpool <= 1:
            return metadata

        forward_mode = forward_batch.forward_mode
        slots_per_page = self._kpool_slots_per_page()
        build_schedule_metadata = self._build_kpool_paged_mqa_schedule_metadata()
        if forward_mode.is_extend_without_speculative():
            assert topk_transform_method is not None
            assert kpool_inputs is not None
            return init_kpool_extend_metadata(
                metadata,
                forward_batch,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                slots_per_page=slots_per_page,
                topk_transform_method=topk_transform_method,
                full_real_page_table=kpool_inputs.full_real_page_table,
                full_seqlens_expanded=kpool_inputs.full_seqlens_expanded,
            )

        if forward_mode.is_decode_or_idle():
            metadata = init_pooled_paged_mqa_metadata(
                metadata,
                metadata.cache_seqlens_int32,
                forward_mode,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                slots_per_page=slots_per_page,
                build_schedule_metadata=build_schedule_metadata,
            )
            return init_kpool_write_plan(
                metadata,
                forward_batch,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                real_page_table=metadata.real_page_table,
                num_draft_tokens=1,
                write_start=(forward_batch.seq_lens - 1).to(torch.int32),
                slots_per_page=slots_per_page,
                build_schedule_metadata=build_schedule_metadata,
            )

        if forward_mode.is_target_verify():
            return init_kpool_write_plan(
                metadata,
                forward_batch,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                real_page_table=metadata.real_page_table,
                num_draft_tokens=self.speculative_num_draft_tokens,
                write_start=forward_batch.seq_lens.to(torch.int32),
                slots_per_page=slots_per_page,
                build_schedule_metadata=build_schedule_metadata,
            )

        if forward_mode.is_draft_extend_v2():
            spec_info = forward_batch.spec_info
            effective_n_per_batch = (
                spec_info.num_accept_tokens
                if spec_info is not None
                and getattr(spec_info, "num_accept_tokens", None) is not None
                else None
            )
            return init_kpool_write_plan(
                metadata,
                forward_batch,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                real_page_table=metadata.real_page_table,
                num_draft_tokens=self.speculative_num_draft_tokens,
                write_start=(
                    forward_batch.seq_lens - self.speculative_num_draft_tokens
                ).to(torch.int32),
                slots_per_page=slots_per_page,
                effective_n_per_batch=effective_n_per_batch,
                build_schedule_metadata=build_schedule_metadata,
            )

        return metadata

    def _init_kpool_metadata_capture(
        self, metadata: DSAMetadata, bs: int, forward_mode: ForwardMode
    ) -> DSAMetadata:
        if self.dsa_index_kpool <= 1:
            return metadata

        slots_per_page = self._kpool_slots_per_page()
        build_schedule_metadata = self._build_kpool_paged_mqa_schedule_metadata()
        if forward_mode.is_decode_or_idle():
            metadata = init_pooled_paged_mqa_metadata(
                metadata,
                metadata.cache_seqlens_int32,
                forward_mode,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                slots_per_page=slots_per_page,
                build_schedule_metadata=build_schedule_metadata,
            )

        if (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
        ):
            is_decode = forward_mode.is_decode_or_idle()
            is_v2 = forward_mode.is_draft_extend_v2()
            metadata = init_kpool_write_plan_capture(
                metadata,
                max_bs=bs,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                num_draft_tokens=(
                    1 if is_decode else self.speculative_num_draft_tokens
                ),
                device=self.device,
                is_verify=not is_decode,
                slots_per_page=slots_per_page,
                is_v2=is_v2,
                build_schedule_metadata=build_schedule_metadata,
            )

        return metadata

    def _update_kpool_metadata_replay(
        self,
        metadata: DSAMetadata,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        effective_n_per_batch: Optional[torch.Tensor] = None,
    ) -> None:
        if self.dsa_index_kpool <= 1:
            return

        slots_per_page = self._kpool_slots_per_page()
        build_schedule_metadata = self._build_kpool_paged_mqa_schedule_metadata()
        if forward_mode.is_decode_or_idle():
            update_pooled_paged_mqa_metadata(
                metadata,
                metadata.cache_seqlens_int32,
                forward_mode,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                slots_per_page=slots_per_page,
                build_schedule_metadata=build_schedule_metadata,
            )

        if not (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
        ):
            return

        is_decode = forward_mode.is_decode_or_idle()
        is_v2 = forward_mode.is_draft_extend_v2()
        if is_decode:
            write_start = seq_lens.to(torch.int32) - 1
        elif is_v2:
            write_start = seq_lens.to(torch.int32) - self.speculative_num_draft_tokens
        else:
            # Target verify: write_start == seq_lens exactly; the plan kernel
            # casts on load, so skip the per-replay int32 alloc + conversion.
            write_start = seq_lens
        update_kpool_write_plan(
            metadata,
            write_start=write_start,
            req_pool_indices=req_pool_indices,
            real_page_table=metadata.real_page_table,
            pool_size=self.dsa_index_kpool,
            real_page_size=self.real_page_size,
            num_draft_tokens=(1 if is_decode else self.speculative_num_draft_tokens),
            forward_mode=forward_mode,
            slots_per_page=slots_per_page,
            effective_n_per_batch=effective_n_per_batch,
        )

    def _update_kpool_metadata_from_precomputed(
        self,
        metadata: DSAMetadata,
        precomputed: PrecomputedMetadata,
        forward_mode: ForwardMode,
    ) -> None:
        if self.dsa_index_kpool <= 1:
            return

        slots_per_page = self._kpool_slots_per_page()
        build_schedule_metadata = self._build_kpool_paged_mqa_schedule_metadata()
        if forward_mode.is_decode_or_idle():
            update_pooled_paged_mqa_metadata(
                metadata,
                precomputed.cache_seqlens,
                forward_mode,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                slots_per_page=slots_per_page,
                build_schedule_metadata=build_schedule_metadata,
            )

        if not (forward_mode.is_decode_or_idle() or forward_mode.is_target_verify()):
            return

        is_verify = forward_mode.is_target_verify()
        write_start = precomputed.cache_seqlens.to(torch.int32)
        write_start = (
            write_start - self.speculative_num_draft_tokens
            if is_verify
            else write_start - 1
        )
        update_kpool_write_plan(
            metadata,
            write_start=write_start,
            req_pool_indices=precomputed.req_pool_indices,
            real_page_table=metadata.real_page_table,
            pool_size=self.dsa_index_kpool,
            real_page_size=self.real_page_size,
            num_draft_tokens=self.speculative_num_draft_tokens if is_verify else 1,
            forward_mode=forward_mode,
            slots_per_page=slots_per_page,
        )
