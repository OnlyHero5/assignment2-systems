import math

import torch

from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor

import llm

class FlashAttentionNaive(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q: Float[Tensor, "query_len d_model"],
                k: Float[Tensor, "key_len d_model"],
                v: Float[Tensor, "key_len d_model"],
                is_causal: bool = False,

                # XXX tune these or infer based on inspecting hardware?
                q_tile_size: int = 64,
                kv_tile_size: int = 64,
                ) -> Float[Tensor, "query_len d_model"]:
        assert k.shape == v.shape
        assert q.shape[-1] == k.shape[-1]

        d_model = q.shape[-1]
        scaling_factor = math.sqrt(d_model)
        query_len = q.shape[-2]
        key_len = k.shape[-2]

        n_query_tile = math.ceil(query_len / q_tile_size)
        n_kv_tile = math.ceil(key_len / kv_tile_size)

        o: Float[Tensor, "query_len d_model"] = torch.zeros(q.shape)

        # corresponds to `m` in paper spec
        row_maxes: Float[Tensor, "query_len 1"] = torch.ones(query_len, 1) * -torch.inf

        # corresponds to `l` in paper spec
        log_sum_exps: Float[Tensor, "query_len 1"] = torch.zeros(query_len, 1, dtype=torch.float32)

        ctx.save_for_backward(q, k, v, log_sum_exps, o)

        for q_tile_index in range(n_query_tile):
            q_tile_base = q_tile_index * q_tile_size
            q_tile_end = q_tile_base + q_tile_size
            q_tile: Float[Tensor, "q_tile_size d_model"] = q[q_tile_base:q_tile_end]

            for kv_tile_index in range(n_kv_tile):
                kv_tile_base = kv_tile_index * kv_tile_size
                kv_tile_end = kv_tile_base + kv_tile_size
                k_tile: Float[Tensor, "kv_tile_size d_model"] = k[kv_tile_base:kv_tile_end]
                v_tile: Float[Tensor, "kv_tile_size d_model"] = v[kv_tile_base:kv_tile_end]

                # corresponds to `S` in paper spec (line 9)
                raw_attention_tile: Float[Tensor, "q_tile_size kv_tile_size"] = einsum(
                    q_tile, k_tile, "q_tile_size d_model, kv_tile_size d_model -> q_tile_size kv_tile_size"
                ) / scaling_factor

                prev_running_row_maxes: Float[Tensor, "q_tile_size 1"] = row_maxes[q_tile_base:q_tile_end]
                # corresponds to `m~` in paper spec
                tile_row_maxes: Float[Tensor, "q_tile_size 1"] = raw_attention_tile.max(dim=-1, keepdim=True).values
                new_running_row_maxes: Float[Tensor, "q_tile_size 1"] = prev_running_row_maxes.where(prev_running_row_maxes.greater(tile_row_maxes), tile_row_maxes)
                
                # corresponds to `P~` in paper spec
                # XXX why have two separate steps to adjust by `tile_row_maxes` and rescale later on, rather than simply subtract the new running maxes?
                tile_exps: Float[Tensor, "q_tile_size kv_tile_size"] = (raw_attention_tile - tile_row_maxes).exp()

                # corresponds to `l~` in paper spec
                local_denoms: Float[Tensor, "q_tile_size 1"] = tile_exps.sum(dim=1, keepdim=True)

                # compute how much we've increased the softmax denominators in this tile so that we can scale the running sums accordingly
                prev_exp_scaling_factors: Float[Tensor, "q_tile_size 1"] = (prev_running_row_maxes - new_running_row_maxes).exp()
                tile_exp_scaling_factors: Float[Tensor, "q_tile_size 1"] = (tile_row_maxes - new_running_row_maxes).exp()

                prev_log_sum_exp: Float[Tensor, "q_tile_size 1"] = log_sum_exps[q_tile_base:q_tile_end]
                new_log_sum_exp: Float[Tensor, "q_tile_size 1"] = (prev_log_sum_exp * prev_exp_scaling_factors) + (tile_exp_scaling_factors * local_denoms)

                #tile_values: Float[Tensor, "q_tile_size d_model"] = einsum(tile_exps, v_tile, "q_tile_size kv_tile_size, kv_tile_size, d_model -> q_tile_size d_model")
                tile_values: Float[Tensor, "q_tile_size d_model"] = tile_exps @ v_tile
                
                # update accumulated output (line 12 in paper spec)
                # XXX paper illustrates the log_sum_exp adjustments at matmuls using `l` converted to diagonal matrices, but
                # this seems equivalent to just a simple elementwise multiply broadcasting the column vector `l`?
                # is this just a computational efficiency thing or am i misunderstanding the math here?
                prev_output: Float[Tensor, "q_tile_size d_model"] = o[q_tile_base:q_tile_end]
                new_output: Float[Tensor, "q_tile_size d_model"] = (
                    (prev_output * prev_log_sum_exp * prev_exp_scaling_factors) + # undo scaling from previous running denominator
                    (tile_values * tile_exp_scaling_factors) # add this tile's values with appropriate scaling
                ) / new_log_sum_exp # rescale everything based on the running denominator
                o[q_tile_base:q_tile_end] = new_output

                # write accumulated row maxes and running denominators back to global memory (line 13: update `l` and `m` in paper spec)
                row_maxes[q_tile_base:q_tile_end] = new_running_row_maxes
                log_sum_exps[q_tile_base:q_tile_end] = new_log_sum_exp

        return o

    @staticmethod
    def backward(ctx, q, k, v) -> None:
        raise NotImplementedError()

class FlashAttention2Naive(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q: Float[Tensor, "query_len d_model"],
                k: Float[Tensor, "key_len d_model"],
                v: Float[Tensor, "key_len d_model"],
                is_causal: bool = False,

                # XXX tune these or infer based on inspecting hardware?
                q_tile_size: int = 64,
                kv_tile_size: int = 64,
                ) -> Float[Tensor, "query_len d_model"]:
        assert k.shape == v.shape
        assert q.shape[-1] == k.shape[-1]

        d_model = q.shape[-1]
        scaling_factor = math.sqrt(d_model)
        query_len = q.shape[-2]
        key_len = k.shape[-2]

        n_query_tile = math.ceil(query_len / q_tile_size)
        n_kv_tile = math.ceil(key_len / kv_tile_size)

        # these are the only two global memory allocations, used for output
        o: Float[Tensor, "query_len d_model"] = torch.empty(q.shape, dtype=torch.float32)
        log_sum_exps: Float[Tensor, "query_len 1"] = torch.zeros(query_len, 1, dtype=torch.float32) # corresponds to `L` in paper spec

        ctx.save_for_backward(q, k, v, log_sum_exps, o)

        for q_tile_index in range(n_query_tile):
            q_tile_base = q_tile_index * q_tile_size
            q_tile_end = q_tile_base + q_tile_size
            q_tile: Float[Tensor, "q_tile_size d_model"] = q[q_tile_base:q_tile_end]

            # all local memory; write back to global at end of loop
            tile_output: Float[Tensor, "q_tile_size d_model"] = torch.zeros(q_tile_size, d_model)
            # corresponds to `m` in paper spec
            row_maxes: Float[Tensor, "q_tile_size 1"] = torch.ones(q_tile_size, 1) * -torch.inf
            # corresponds to `l` in paper spec
            running_denoms: Float[Tensor, "q_tile_size 1"] = torch.zeros(q_tile_size, 1)

            for kv_tile_index in range(n_kv_tile):
                kv_tile_base = kv_tile_index * kv_tile_size
                kv_tile_end = kv_tile_base + kv_tile_size
                k_tile: Float[Tensor, "kv_tile_size d_model"] = k[kv_tile_base:kv_tile_end]
                v_tile: Float[Tensor, "kv_tile_size d_model"] = v[kv_tile_base:kv_tile_end]

                raw_attention_tile: Float[Tensor, "q_tile_size kv_tile_size"] = einsum(
                    q_tile, k_tile, "q_tile_size d_model, kv_tile_size d_model -> q_tile_size kv_tile_size"
                ) / scaling_factor

                prev_running_row_maxes: Float[Tensor, "q_tile_size 1"] = row_maxes
                # corresponds to `m~` in paper spec
                tile_row_maxes: Float[Tensor, "q_tile_size 1"] = raw_attention_tile.max(dim=-1, keepdim=True).values
                row_maxes = prev_running_row_maxes.where(prev_running_row_maxes.greater(tile_row_maxes), tile_row_maxes)
                
                # corresponds to `P~` in paper spec
                # this is now scaled to the running max values
                tile_exps: Float[Tensor, "q_tile_size kv_tile_size"] = (raw_attention_tile - row_maxes).exp()

                # corresponds to `l~` in paper spec
                local_denoms: Float[Tensor, "q_tile_size 1"] = tile_exps.sum(dim=1, keepdim=True)

                # if necessary, re-scale the running denominators based on any increases in row maxes
                prev_exp_scaling_factors: Float[Tensor, "q_tile_size 1"] = (prev_running_row_maxes - row_maxes).exp()

                # the running denominators are now fully scaled based on the most recent row maxes
                running_denoms = (running_denoms * prev_exp_scaling_factors) + local_denoms

                #tile_values: Float[Tensor, "q_tile_size d_model"] = einsum(tile_exps, v_tile, "q_tile_size kv_tile_size, kv_tile_size, d_model -> q_tile_size d_model")
                tile_values: Float[Tensor, "q_tile_size d_model"] = tile_exps @ v_tile
                
                tile_output = tile_values + (tile_output * prev_exp_scaling_factors)

            # persist final values for query tile
            o[q_tile_base:q_tile_end] = tile_output / running_denoms
            log_sum_exps[q_tile_base:q_tile_end] = row_maxes + torch.log(running_denoms)

        return o

    @staticmethod
    def backward(ctx, q, k, v) -> None:
        raise NotImplementedError()


flash_attention = FlashAttentionNaive.apply
flash_attention_2 = FlashAttention2Naive.apply

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("seq_len", type=int, nargs="?", default=256)
    parser.add_argument("d_model", type=int, nargs="?", default=128)
    parser.add_argument("--tile-size")

    args = parser.parse_args()

    def create_input() -> Float[Tensor, "seq_len d_model"]:
        return torch.rand(args.seq_len, args.d_model, requires_grad=True)

    q = create_input()
    k = create_input()
    v = create_input()

    flash_kwargs = {}
    if args.tile_size:
        flash_kwargs["q_tile_size"], flash_kwargs["kv_tile_size"] = tuple(map(int, args.tile_size.split("x")))

    flash_result: Float[Tensor, "seq_len d_model"] = flash_attention_2(q, k, v) # type: ignore
    ref_result = llm.attention(q, k, v)

    assert flash_result.shape == ref_result.shape
    print(flash_result.allclose(ref_result))
    print(flash_result.isclose(ref_result))
    print(flash_result)
    print(ref_result)