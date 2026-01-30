import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple


# ===============================
# N-gram pool（token_map）
# ===============================

class NGramPool:
    def __init__(self, guess_size: int):
        self.pool: Dict[int, List[Tuple[int, ...]]] = {}
        self.guess_size = guess_size

    def add(self, prefix: int, cont: Tuple[int, ...]):
        if prefix not in self.pool:
            self.pool[prefix] = []
        if cont not in self.pool[prefix]:
            self.pool[prefix].append(cont)

    def get(self, prefix: int):
        return self.pool.get(prefix, None)


# ===============================
# Minimal but CORRECT LADE greedy
# ===============================

@torch.no_grad()
def lade_greedy_minimal(
    model,
    tokenizer,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 50,
    window_size: int = 8,
    guess_size: int = 4,   # = LEVEL - 1
):
    """
    This implementation is:
    - single-GPU
    - greedy
    - verification forward matches LADE layout
    """
    device = input_ids.device
    model.eval()

    # ===== init =====
    all_tokens = input_ids[0].tolist()
    token_map = NGramPool(guess_size)

    # ===== prefill =====
    outputs = model(input_ids, use_cache=True, return_dict=True)
    past_key_values = outputs.past_key_values

    steps = 0
    base_len = input_ids.size(1)

    while len(all_tokens) < base_len + max_new_tokens:
        steps += 1
        last_token = all_tokens[-1]

        # =====================================================
        # 1. Lookahead draft branch (autoregressive rollout)
        # =====================================================
        draft_tokens = []
        draft_cache = past_key_values

        cur_input = torch.tensor([[last_token]], device=device)

        for _ in range(window_size):
            out = model(
                cur_input,
                past_key_values=draft_cache,
                use_cache=True,
                return_dict=True,
            )
            next_tok = torch.argmax(out.logits[:, -1], dim=-1)
            draft_tokens.append(next_tok.item())
            cur_input = next_tok[:, None]
            draft_cache = out.past_key_values

        # ===== update n-gram pool =====
        for i in range(len(draft_tokens) - guess_size + 1):
            token_map.add(
                last_token if i == 0 else draft_tokens[i - 1],
                tuple(draft_tokens[i:i + guess_size])
            )

        # =====================================================
        # 2. Verification branch (LADE layout)
        # =====================================================
        guess_list = token_map.get(last_token)

        max_hit = 0
        hit_tokens: List[int] = []
        best_idx = -1

        if guess_list is not None and len(guess_list) > 0:
            # ===== explicit num_candidates =====
            num_candidates = len(guess_list)

            # -------------------------------------------------
            # Build verification sequence:
            #   [first greedy token] + [cand0] + [cand1] + ...
            # -------------------------------------------------
            first_tok = draft_tokens[0]
            verif_tokens: List[int] = [first_tok]
            for g in guess_list:
                verif_tokens.extend(g)

            verif_ids = torch.tensor(
                [verif_tokens], device=device, dtype=torch.long
            )

            # single forward (THIS is the key difference)
            out = model(
                verif_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            # logits shape: [1, T_verif, vocab]
            preds = torch.argmax(out.logits, dim=-1)[0]

            # -------------------------------------------------
            # verification: longest prefix match
            # -------------------------------------------------
            for i, g in enumerate(guess_list):
                hit = 0
                for j in range(guess_size):
                    # +1 offset because preds[0] is first_tok
                    if preds[1 + i * guess_size + j].item() == g[j]:
                        hit += 1
                    else:
                        break
                if hit > max_hit:
                    max_hit = hit
                    hit_tokens = list(g[:hit])
                    best_idx = i

        # =====================================================
        # 3. Fallback: normal greedy
        # =====================================================
        if max_hit == 0:
            out = model(
                torch.tensor([[last_token]], device=device),
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            next_tok = torch.argmax(out.logits[:, -1], dim=-1).item()
            hit_tokens = [next_tok]
            new_cache = out.past_key_values

        else:
            # =================================================
            # ⭐ KV CACHE COPY (STRICTLY matches LADE)
            # =================================================
            new_cache = []

            # T_verif = 1 + guess_size * num_candidates
            T_verif = out.logits.size(1)

            flat_guess_start = T_verif - guess_size * num_candidates
            src_start = flat_guess_start + best_idx * guess_size

            for layer, (k, v) in enumerate(out.past_key_values):
                # k, v: [B=1, H, T_verif_total, D]
                old_len = past_key_values[layer][0].size(2) 
                #size(2)是因为第三个维度是T_verif_total，T_verif_total其实是顺序一维排列
                #t:  0 1 2 3 4 | 5 | 6 7 8 9 | 10 11 12 13 | 14 15 16 17
                #     prefix     g   cand0       cand1           cand2


                k_new = k[:, :, :old_len + max_hit, :].clone()
                v_new = v[:, :, :old_len + max_hit, :].clone()

                k_new[:, :, -max_hit:] = k[:, :, src_start:src_start + max_hit]
                v_new[:, :, -max_hit:] = v[:, :, src_start:src_start + max_hit]

                new_cache.append((k_new, v_new))

        # ===== commit =====
        past_key_values = new_cache
        all_tokens.extend(hit_tokens)

        if tokenizer.eos_token_id in hit_tokens:
            break

    print(
        f"[LADE minimal] tokens={len(all_tokens)}, "
        f"steps={steps}, "
        f"compression={(len(all_tokens)-base_len)/steps:.2f}"
    )
    return torch.tensor([all_tokens], device=device)
