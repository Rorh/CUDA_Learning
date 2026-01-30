import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, List
import lade_cuda   # <<< CUDA EXTENSION ———————————————— 关键 ✔✔✔


###########################################
# NGRAM POOL
###########################################

class NGramPool:
    def __init__(self, max_n: int, max_pool: int, device: torch.device):
        self.max_n = max_n
        self.max_pool = max_pool
        self.device = device

        # stored in CPU list; packed to CUDA tensors on demand
        self.tokens_list: List[torch.Tensor] = []
        self.lengths_list: List[int] = []
        self.scores_list: List[float] = []

    def add(self, tokens: torch.Tensor, score: float):
        if tokens.is_cuda:
            tokens = tokens.cpu()
        length = tokens.size(0)
        if length == 0:
            return
        if length > self.max_n:
            tokens = tokens[: self.max_n]
            length = self.max_n

        self.tokens_list.append(tokens.to(torch.int32))
        self.lengths_list.append(length)
        self.scores_list.append(float(score))

        if len(self.tokens_list) > self.max_pool:
            self.tokens_list.pop(0)
            self.lengths_list.pop(0)
            self.scores_list.pop(0)

    def to_cuda_tensors(self):
        """
        Convert lists → CUDA tensors (for lade_cuda kernel)
        """
        if len(self.tokens_list) == 0:
            return (
                torch.empty((0, self.max_n), dtype=torch.int32, device=self.device),
                torch.empty((0,), dtype=torch.int32, device=self.device),
                torch.empty((0,), dtype=torch.float32, device=self.device),
            )

        num = len(self.tokens_list)

        ngram_tokens = torch.full(
            (num, self.max_n),
            fill_value=-1,
            dtype=torch.int32,
            device=self.device
        )
        ngram_lengths = torch.tensor(self.lengths_list, dtype=torch.int32, device=self.device)
        ngram_scores = torch.tensor(self.scores_list, dtype=torch.float32, device=self.device)

        for i, toks in enumerate(self.tokens_list):
            ngram_tokens[i, :toks.size(0)] = toks

        return ngram_tokens, ngram_lengths, ngram_scores


###########################################
# KV-CACHE 跳跃逻辑（来自 LADE 源码）
###########################################

def apply_kv_skip(
    outputs,
    guess_tokens,
    max_hit,
    max_hit_idx,
    GUESS_SIZE,
    DIST_WORKERS=1
):
    """
    Update outputs.past_key_values according to LADE:
    - distributed mode
    - single worker KV overwrite mode

    outputs must contain:
        - past_key_values : list[(k, v)]
        - kvcache_len
        - step_len
    """

    pkv = outputs.past_key_values
    kvcache_len = outputs.kvcache_len
    step_len = outputs.step_len

    new_pkv = []

    if DIST_WORKERS > 1 and max_hit > 0:
        # parallel mode
        for (k, v) in pkv:
            new_k = k[:, :, :kvcache_len, :].contiguous()
            new_v = v[:, :, :kvcache_len, :].contiguous()
            new_pkv.append((new_k, new_v))
        outputs.past_key_values = new_pkv
        return

    # single worker mode
    offset = (step_len - len(guess_tokens) + max_hit_idx * GUESS_SIZE) if max_hit > 0 else 0

    for (k, v) in pkv:
        if max_hit > 0:
            # overwrite KV-cache
            k[:, :, kvcache_len:kvcache_len + max_hit, :] = k[:, :, offset:offset + max_hit, :]
            v[:, :, kvcache_len:kvcache_len + max_hit, :] = v[:, :, offset:offset + max_hit, :]

        new_k = k[:, :, :kvcache_len + max_hit, :].contiguous()
        new_v = v[:, :, :kvcache_len + max_hit, :].contiguous()
        new_pkv.append((new_k, new_v))

    outputs.past_key_values = new_pkv


###########################################
# GREEDY STEP
###########################################

@torch.no_grad()
def greedy_step(model, input_ids, past_key_values=None):
    out = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
    logits = out.logits[:, -1, :]   # [1, vocab]
    next_id = torch.argmax(logits, dim=-1)  # [1]
    return next_id, logits, out


###########################################
# NGRAM ROLLOUT (verification branch candidate)
###########################################

@torch.no_grad()
def generate_ngram(model, prefix_ids, start_token, n, eos_id=None, past_key_values=None):
    device = prefix_ids.device
    tokens = [start_token]
    score = 0.0

    cur_ids = torch.cat([prefix_ids, torch.tensor([[start_token]], device=device)], dim=1)
    cur_pkv = past_key_values

    for _ in range(n - 1):
        next_id, logits, out = greedy_step(model, cur_ids, cur_pkv)
        sc = logits[0, next_id.item()].item()
        score += sc
        nt = next_id.item()
        tokens.append(nt)

        cur_ids = torch.cat([cur_ids, next_id.unsqueeze(-1)], dim=1)
        cur_pkv = out.past_key_values

        if eos_id is not None and nt == eos_id:
            break

    return torch.tensor(tokens, dtype=torch.int32), score


###########################################
# LOOKAHEAD DECODE 主函数（含 CUDA + KV）
###########################################

@torch.no_grad()
def lookahead_decode(
    model,
    input_ids,
    max_new_tokens=50,
    eos_token_id=None,
    n=4,
    pool_size=64,
    topk=4,
    GUESS_SIZE=4,
    DIST_WORKERS=1
):
    device = input_ids.device
    generated = input_ids.clone()

    # 初始化 past_key_values
    next_id, logits, out = greedy_step(model, generated, None)
    past_key_values = out.past_key_values
    generated = torch.cat([generated, next_id.unsqueeze(0)], dim=1)

    # 初始化 n-gram pool
    pool = NGramPool(max_n=n, max_pool=pool_size, device=device)

    # 状态信息（如 LADE）
    class DecodeState:
        pass
    state = DecodeState()
    state.kvcache_len = generated.size(1)
    state.step_len = generated.size(1)
    state.past_key_values = past_key_values

    for step in range(max_new_tokens):

        last_token = generated[0, -1].item()

        # Step 1 — greedy logits
        next_id, logits, out = greedy_step(model, generated, state.past_key_values)
        greedy_next = next_id.item()
        greedy_score = logits[0, greedy_next].item()

        # Step 2 — expand top-k branches and populate ngram pool
        vocab_logits = logits[0]
        top_vals, top_idx = torch.topk(vocab_logits, k=topk)

        guess_tokens = []
        guess_scores = []

        for i, st in enumerate(top_idx.tolist()):
            toks, sc = generate_ngram(
                model=model,
                prefix_ids=generated,
                start_token=st,
                n=n,
                eos_id=eos_token_id,
                past_key_values=state.past_key_values
            )
            sc += vocab_logits[st].item()
            pool.add(toks, sc)
            guess_tokens.append(toks)
            guess_scores.append(sc)

        # Step 3 — CUDA: 在所有 n-gram 中找首 token = last_token 且得分最高者
        ngram_tokens, ngram_lengths, ngram_scores = pool.to_cuda_tensors()

        if ngram_tokens.size(0) == 0:
            chosen = greedy_next
            hit_len = 0
            hit_idx = -1
        else:
            best_idx = lade_cuda.find_best_ngram(
                ngram_tokens, ngram_lengths, ngram_scores, last_token
            )  # tensor([idx], cuda)

            best_i = int(best_idx.item())
            if best_i < 0:
                chosen = greedy_next
                hit_len = 0
                hit_idx = -1
            else:
                hit_len = int(ngram_lengths[best_i])
                score = float(ngram_scores[best_i])
                avg = score / max(hit_len, 1)

                if avg >= greedy_score:
                    chosen = int(ngram_tokens[best_i, 0].item())
                else:
                    chosen = greedy_next

                hit_idx = best_i

        # Step 4 — Update KV-cache using LADE skip rule
        state.kvcache_len = generated.size(1)
        state.step_len = generated.size(1)
        state.past_key_values = out.past_key_values

        if hit_len > 0:
            apply_kv_skip(
                outputs=state,
                guess_tokens=ngram_tokens[hit_idx].tolist(),
                max_hit=hit_len,
                max_hit_idx=hit_idx,
                GUESS_SIZE=GUESS_SIZE,
                DIST_WORKERS=DIST_WORKERS
            )

        # Step 5 — append chosen token
        new_tok = torch.tensor([[chosen]], device=device)
        generated = torch.cat([generated, new_tok], dim=1)

        if eos_token_id is not None and chosen == eos_token_id:
            break

    return generated
