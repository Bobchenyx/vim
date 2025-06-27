import warnings
import torch
from tome import (bipartite_soft_matching,
                   bipartite_hybrid_merge,
                   bipartite_hybrid_pure,
                   bipartite_hybrid_prune,
                   merge_wavg)

__all__ = [
    "prune",
    "merge"
]


def prune(values: torch.Tensor,
          score: torch.Tensor,
          num_keep_node: int,
          preserve_length: bool):
    B, N, D = values.shape
    if num_keep_node == N:
        return values
    # assert num_keep_node <= N, f"total seq_length {N} but want to keep {num_keep_node}"

    score_nocls = score.clone()
    if preserve_length:
        key_score = score_nocls.sort(dim=1, descending=True)[0][:, num_keep_node:num_keep_node + 1]
        score_mask = torch.where(score_nocls > key_score,
                                 torch.ones_like(score_nocls),
                                 torch.zeros_like(score_nocls))[..., None].expand_as(values)
        values *= score_mask
    else:
        sorted_score_key = score_nocls.sort(dim=1)[0]
        while True:
            key_score = sorted_score_key[:, num_keep_node:num_keep_node + 1]
            score_mask = torch.where(score_nocls > key_score,
                                     torch.ones_like(score_nocls),
                                     torch.zeros_like(score_nocls))[..., None].expand_as(values)
            try:
                values = values[score_mask == 0].reshape(B, -1, D)
            except RuntimeError as e:
                num_keep_node += 1
                #warnings.warn("Adjust num_keep_node to {}".format(num_keep_node))
            else:
                break

    return values


# def merge(values, score, num_keep_node, preserve_length):
#     B, N, D = values.shape
#     merge, unmerge = bipartite_soft_matching(score, r=N - num_keep_node)
#     merge_values, _ = merge_wavg(merge, values)
#     dense_shape = merge_values.shape
#     merge_values = unmerge(merge_values)

#     if not preserve_length:
#         merge_values = merge_values[merge_values.abs().sum(-1) != 0, :].reshape(dense_shape)

#     return merge_values
def merge(values, score, num_keep_node, preserve_length):
    B, N, D = values.shape
    merge_, _ = bipartite_soft_matching(score, r=N - num_keep_node)
    merge_values = merge_(values)
    # merge, unmerge = bipartite_soft_matching(score, r=N - num_keep_node)
    # merge_values, _ = merge_wavg(merge, values)
    # dense_shape = merge_values.shape
    # merge_values = unmerge(merge_values)

    # if not preserve_length:
    #     merge_values = merge_values[merge_values.abs().sum(-1) != 0, :].reshape(dense_shape)

    return merge_values


# def hybrid_prune(values: torch.Tensor,
#                  score: torch.Tensor,
#                  num_keep_node: int,
#                  preserve_length: bool):
#     B, N, D = values.shape
#     merge, unmerge = bipartite_hybrid_prune(score, r=N - num_keep_node)
#     merge_values, _ = merge_wavg(merge, values)
#     dense_shape = merge_values.shape
#     merge_values = unmerge(merge_values)
#     if not preserve_length:
#         merge_values = merge_values[merge_values.abs().sum(-1) != 0, :].reshape(dense_shape)

#     return merge_values
# def hybrid_prune(values: torch.Tensor,
#                  score: torch.Tensor,
#                  num_keep_node: int,
#                  preserve_length: bool):
#     B, N, D = values.shape
#     merge, _ = bipartite_hybrid_prune(values, score, r=N - num_keep_node)
#     merge_values = merge(values)

#     return merge_values
def hybrid_prune(values: torch.Tensor,
                 score_1: torch.Tensor,
                 score_2: torch.Tensor,
                 num_keep_node: int,
                 preserve_length: bool):
    B, N, D = values.shape
    # 1 是重要性分数 # 2 是相似度分数
    # print("score_2", score_2.shape)
    # print("score_1", score_1.shape)
    # assert(0)
    # with torch.no_grad():
    merge, _ = bipartite_hybrid_prune(values, score_2, score_1, r=N - num_keep_node)
    merge_values = merge()

    return merge_values


# def hybrid_merge(values, score, num_keep_node, preserve_length):
#     B, N, D = values.shape
#     # merge, unmerge = bipartite_hybrid_merge(score, r=N - num_keep_node)
#     merge, _ = bipartite_hybrid_merge(values, score, r=N - num_keep_node)
#     # merge_values, _ = merge_wavg(merge, values)
#     merge_values = merge(values)
#     # dense_shape = merge_values.shape
#     # merge_values = unmerge(merge_values)
#     # if not preserve_length:
#         # merge_values = merge_values[merge_values.abs().sum(-1) != 0, :].reshape(dense_shape)

#     return merge_values
def hybrid_merge(values: torch.Tensor,
                 score_1: torch.Tensor,
                 score_2: torch.Tensor,
                 num_keep_node: int,
                 preserve_length: bool):
    B, N, D = values.shape
    merge, _ = bipartite_hybrid_merge(values, score_2, score_1, r=N - num_keep_node)
    merge_values = merge()

    return merge_values


def pure_hybrid(values, score, num_keep_node, preserve_length):
    B, N, D = values.shape
    merge, unmerge = bipartite_hybrid_pure(score, r=N - num_keep_node)
    merge_values, _ = merge_wavg(merge, values)
    dense_shape = merge_values.shape
    merge_values = unmerge(merge_values)
    if not preserve_length:
        merge_values = merge_values[merge_values.abs().sum(-1) != 0, :].reshape(dense_shape)

    return merge_values