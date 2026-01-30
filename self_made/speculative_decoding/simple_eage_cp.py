import torch
import torch.nn.functional as F
import random
from typing import List, Tuple

def build_tree_attention_mask(tree_paths: List[List[int]], device):
    tree_len = len(tree_paths)

    mask = torch.zeros(tree_len, tree_len, device=device)

    for i, path in enumerate(tree_paths):
        mask[i, i] = 1
        for j, paraent in enumerate(tree_paths):
            if len(paraent) < len(path) and path[:len(paraent)] == paraent:
                mask[i, j] = 1

    return mask[None, None]



def slice_kv_cache(past_kv, indices):
    new_kv = []

    for layer_kv in past_kv:
        k, v = layer_kv
        new_k = k[:, :, indices, :]
        new_v = v[:, :, indices, :]
        new_kv.append((new_k, new_v))

    return tuple(new_kv)


def append_kv_cache(past_kv, new_kv):
    if past_kv is None:
        return new_kv
    merged = []
    for (k1, v1), (k2, v2) in zip(past_kv, new_kv):
        merged.append((torch.cat([k1, k2], dim=2),
                     (torch.cat([v1. v2], dim=2))))


@torch.no_grad()
def propose_tree(
    ea_model,
    input_ids,
    attention_mask,
    tree_paths,
    top_k=5,
):

    device = input_ids.device
    hidden, past_kv = ea_model(
        input_ids = input_ids,
        attention_mask = attention_mask,
        use_cache = True,
    )

    root_logits = ea_model.lm_head(hidden[:, -1])
    topk_probs, topk_tokens = torch.topk(
        F.softmax(root_logits, dim=-1), top_k
    )

    tree_tokens = []
    tree_probs = []

    for depth, path in enumerate(tree_paths):
        token = topk_tokens[:, path[-1]]
        prob = topk_probs[:, path[-1]]
        tree_tokens.append(token)
        tree_probs.append(prob)

    tree_tokens = torch.stack(tree_tokens, dim=1)
    tree_probs = torch.stack(tree_probs, dim=1)

    return tree_tokens, tree_probs, past_kv


@torch.no_grad()
def verify_tree(
    base_model,
    input_ids,
    tree_tokens,
    attention_mask,
    tree_mask,
    past_kv
):

    bs, tree_len = tree_tokens.shape
    device = input_ids.device

    full_input = torch.cat([input_ids. tree_tokens], dim=1)

    full_mask= torch.cat(
        [attention_mask,
        torch.ones(bs, tree_len, device=device, dtype=attention_mask.dtype)],
        dim=1
    )

    hidden, new_kv = base_model(
        input_ids = full_input,
        attention_mask = full_mask,
        past_key_values = past_kv,
        use_cache = True,
        tree_mask = tree_mask,
    )

    logits = base_model.lm_head(hidden[:, -tree_len:])

    return logits, new_kv


def evaluate_posterior(
    logits, 
    tree_tokens,
    tree_probs,
    temperature=0.0,
):

    bs, tree_len, vocab_size = logits.shape
    accept_len = torch.zeros(bs, dtype=torch.long)
    best_path = torch.zeros(bs, dtype=torch.long)

    if temperature == 0.0:
        pred = torch.argmax(logits, dim=-1)
        match = (pred == tree_tokens).int()
        cum = torch.cunprod(match, dim=-1)
        accept_len = cum.sum(dim=-1)
        best_path[:] = 0

    probs = F.softmax(logits / temperature, dim=-1)

    for b in range(bs):
        for i in range(tree_len):
            px = probs[b, i, tree_tokens[b, i]]
            qx = tree_probs[b, i]
            if random.random() <= (px / max(qx, 1e-8)):
                accept_len[b] += 1
            else:
                break

    return best_path, accept_len


@torch.no_grad()
def eagle_generate(
    base_model,
    ea_model,
    input_ids,
    attention_mask,
    tree_paths,
    max_new_tokens=128,
    temperature=0.0,
):
    device = input_ids.device

    tree_mask = build_tree_attention_mask(tree_paths, device)

    past_kv = None
    generated = input_ids.clone()

    for _ in range(max_new_tokens):

        tree_tokens, tree_probs, ea_kv = propose_tree(
            ea_model,
            generated,
            attention_mask,
            tree_paths,
        )

        logits, new_kv = verify_tree(
            base_model,
            generated,
            tree_tokens,
            attention_mask,
            tree_mask,
            past_kv
        )

        _, accept_len = evaluate_posterior(
            logits,
            tree_tokens, 
            tree_probs,
            temperature,
        )

        max_accept = accept_len.max().item()

        if max_accept == 0:
            next_token = torch.argmax(logits[:, 0], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
        else:
            accepted = tree_tokens[:, :max_accept]
            generated = torch.cat([generated, accepted], dim=1)

        past_kv = slice_kv_cache(new_kv, torch.arange(generated.shape[1]))

        if generated.shape[1] >= max_new_tokens:
            break

    return generated