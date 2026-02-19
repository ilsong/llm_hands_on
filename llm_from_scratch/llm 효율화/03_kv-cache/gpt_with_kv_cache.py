# This file collects all the relevant code that we covered thus far
# throughout Chapters 3-4.
# This file can be run as a standalone script.
# =============================================================================
# gpt_ch04.py vs gpt_with_kv_cache.py 비교
# =============================================================================
#
# gpt_ch04.py  : KV 캐시 없는 기본 GPT 구현 (이 파일)
# gpt_with_kv_cache.py : KV 캐시를 추가한 최적화 버전
#
# -----------------------------------------------------------------------------
# 1. MultiHeadAttention.__init__() — 캐시 버퍼
# -----------------------------------------------------------------------------
# [gpt_ch04.py]    없음
#
# [gpt_with_kv_cache.py]
#   self.register_buffer("cache_k", None, persistent=False)
#   self.register_buffer("cache_v", None, persistent=False)
#   self.ptr_current_pos = 0   # 현재 시퀀스 위치 포인터
#
# -----------------------------------------------------------------------------
# 2. MultiHeadAttention.forward() — 캐시 읽기/쓰기 및 마스크 처리
# -----------------------------------------------------------------------------
# [gpt_ch04.py]
#   def forward(self, x):
#       keys = self.W_key(x)                               # 매번 전체 재계산
#       mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # 고정 마스크
#
# [gpt_with_kv_cache.py]
#   def forward(self, x, use_cache=False):                 # 파라미터 추가
#       keys_new = self.W_key(x)                           # 새 토큰만 계산
#       if use_cache:
#           self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
#           keys, values = self.cache_k, self.cache_v      # 캐시에서 참조
#       # 마스크를 위치 포인터 기반으로 동적 슬라이싱
#       mask_bool = self.mask.bool()[
#           self.ptr_current_pos : self.ptr_current_pos + num_tokens_Q,
#           :num_tokens_K
#       ]
#       self.ptr_current_pos += num_tokens_Q
#
# -----------------------------------------------------------------------------
# 3. GPTModel — 블록 컨테이너와 위치 임베딩
# -----------------------------------------------------------------------------
# 항목             | gpt_ch04.py              | gpt_with_kv_cache.py
# ------------------|--------------------------|------------------------------
# 블록 컨테이너     | nn.Sequential            | nn.ModuleList
# 이유             | 순차 자동 실행            | use_cache 인자를 각 블록에 전달
# 위치 임베딩      | arange(seq_len) 항상 0부터 | arange(current_pos, current_pos+seq_len)
# 캐시 초기화 메서드| 없음                     | reset_kv_cache() 추가
#
# [gpt_ch04.py]
#   self.trf_blocks = nn.Sequential(...)
#   pos_embeds = self.pos_emb(torch.arange(seq_len))   # 항상 [0,1,2,3,...]
#   x = self.trf_blocks(x)
#
# [gpt_with_kv_cache.py]
#   self.trf_blocks = nn.ModuleList(...)
#   # 캐시 사용 시 위치를 이어서 계산: 프롬프트 후 [4],[5],[6],...
#   pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len)
#   self.current_pos += seq_len
#   for blk in self.trf_blocks:
#       x = blk(x, use_cache=use_cache)                # 각 블록에 인자 전달
#
# -----------------------------------------------------------------------------
# 4. 생성 함수 — 핵심 동작 차이
# -----------------------------------------------------------------------------
# [gpt_ch04.py] generate_text_simple()
#   매 스텝: [Hello, I, am, a, very] 전체 → 모델 → logits  (n 토큰 계산)
#   다음 스텝: [Hello, I, am, a, very, good] 전체 → 모델 → logits (n+1 토큰)
#   → O(n²) 연산
#
# [gpt_with_kv_cache.py] generate_text_simple_cached()
#   1단계: [Hello, I, am] → 모델(캐시 저장) → logits
#   매 스텝: [새토큰 1개] → 모델(캐시 참조) → logits  (1 토큰만 계산)
#   → O(n) 연산
#
# -----------------------------------------------------------------------------
# 5. 캐시 사용 vs 미사용 성능 비교
# -----------------------------------------------------------------------------
# 항목           | 캐시 사용         | 캐시 미사용
# ---------------|-------------------|-------------------
# 모델 입력      | 새 토큰 1개       | 전체 시퀀스
# K,V 계산       | 새 토큰만         | 매번 전체 재계산
# 시간 복잡도    | O(n)              | O(n²)
# 메모리         | 캐시 저장 필요    | 추가 메모리 없음
# =============================================================================

import time
import tiktoken
import torch
import torch.nn as nn


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False
        )

        ####################################################
        # 캐시 등록
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0
        ####################################################

    def forward(self, x, use_cache=False):
        b, num_tokens, d_in = x.shape

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        # 캐시 적용
        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
            keys, values = self.cache_k, self.cache_v
        else:
            keys, values = keys_new, values_new
        ####################################################

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        ####################################################
        # mask 적용
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]

        # ptr_current_pos 0 ptr_current_pos + num_tokens_Q 4 num_tokens_Q 4 num_tokens_K 4
        # ptr_current_pos 0 ptr_current_pos + num_tokens_Q 4 num_tokens_Q 4 num_tokens_K 4
        # ptr_current_pos 0 ptr_current_pos + num_tokens_Q 4 num_tokens_Q 4 num_tokens_K 4
        # ptr_current_pos 0 ptr_current_pos + num_tokens_Q 4 num_tokens_Q 4 num_tokens_K 4
        # ptr_current_pos 0 ptr_current_pos + num_tokens_Q 4 num_tokens_Q 4 num_tokens_K 4
        # ptr_current_pos 0 ptr_current_pos + num_tokens_Q 4 num_tokens_Q 4 num_tokens_K 4
        # ptr_current_pos 0 ptr_current_pos + num_tokens_Q 4 num_tokens_Q 4 num_tokens_K 4
        # ptr_current_pos 0 ptr_current_pos + num_tokens_Q 4 num_tokens_Q 4 num_tokens_K 4
        # ptr_current_pos 0 ptr_current_pos + num_tokens_Q 4 num_tokens_Q 4 num_tokens_K 4
        # ptr_current_pos 0 ptr_current_pos + num_tokens_Q 4 num_tokens_Q 4 num_tokens_K 4
        # ptr_current_pos 0 ptr_current_pos + num_tokens_Q 4 num_tokens_Q 4 num_tokens_K 4
        # ptr_current_pos 0 ptr_current_pos + num_tokens_Q 4 num_tokens_Q 4 num_tokens_K 4
        # ptr_current_pos 4 ptr_current_pos + num_tokens_Q 5 num_tokens_Q 1 num_tokens_K 5
        # ptr_current_pos 4 ptr_current_pos + num_tokens_Q 5 num_tokens_Q 1 num_tokens_K 5
        # ptr_current_pos 4 ptr_current_pos + num_tokens_Q 5 num_tokens_Q 1 num_tokens_K 5
        # ptr_current_pos 4 ptr_current_pos + num_tokens_Q 5 num_tokens_Q 1 num_tokens_K 5
        # ptr_current_pos 4 ptr_current_pos + num_tokens_Q 5 num_tokens_Q 1 num_tokens_K 5
        # ptr_current_pos 4 ptr_current_pos + num_tokens_Q 5 num_tokens_Q 1 num_tokens_K 5
        # ptr_current_pos 4 ptr_current_pos + num_tokens_Q 5 num_tokens_Q 1 num_tokens_K 5
        # ptr_current_pos 4 ptr_current_pos + num_tokens_Q 5 num_tokens_Q 1 num_tokens_K 5
        # ptr_current_pos 4 ptr_current_pos + num_tokens_Q 5 num_tokens_Q 1 num_tokens_K 5
        # ptr_current_pos 4 ptr_current_pos + num_tokens_Q 5 num_tokens_Q 1 num_tokens_K 5
        # ptr_current_pos 4 ptr_current_pos + num_tokens_Q 5 num_tokens_Q 1 num_tokens_K 5
        # ptr_current_pos 4 ptr_current_pos + num_tokens_Q 5 num_tokens_Q 1 num_tokens_K 5
        # ptr_current_pos 5 ptr_current_pos + num_tokens_Q 6 num_tokens_Q 1 num_tokens_K 6
        if use_cache:
            print("ptr_current_pos",self.ptr_current_pos,"ptr_current_pos + num_tokens_Q",self.ptr_current_pos + num_tokens_Q,"num_tokens_Q",num_tokens_Q, "num_tokens_K",num_tokens_K)
            mask_bool = self.mask.bool()[
                self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
            ]
            self.ptr_current_pos += num_tokens_Q
        ####################################################
        # Original mask truncated to the number of tokens and converted to boolean
        else:
            mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

    ####################################################
    # NEW
    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0
    ####################################################


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)

        # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        ####################################################
        # NEW
        x = self.att(x, use_cache=use_cache)
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # self.trf_blocks = nn.Sequential(
        #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        ####################################################
        # NEW
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.current_pos = 0
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        ####################################################
        # NEW
        # 단계	seq_len	current_pos	pos_ids	설명
        # 프롬프트	4	0 → 4	[0,1,2,3]	"Hello, I am"
        # 1번째 생성	1	4 → 5	[4]	5번째 토큰
        # 2번째 생성	1	5 → 6	[5]	6번째 토큰
        # 3번째 생성	1	6 → 7	[6]	7번째 토큰
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)
        ####################################################
        # NEW
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        ####################################################

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    ####################################################
    # NEW
    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0
    ####################################################


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


####################################################
# NEW
# ┌─────────────────────────────────────────────────────────┐
# │ 1단계: 프롬프트 처리                                      │
# ├─────────────────────────────────────────────────────────┤
# │ 입력: [Hello, ,, I, am]  →  모델  →  logits (4, vocab)  │
# │                              ↓                          │
# │                         캐시에 K,V 저장                  │
# └─────────────────────────────────────────────────────────┘
#                               ↓
# ┌─────────────────────────────────────────────────────────┐
# │ 2단계: 첫 번째 토큰 생성                                  │
# ├─────────────────────────────────────────────────────────┤
# │ logits[:, -1] → argmax → "a"                            │
# │ 입력: ["a"]  →  모델(캐시 사용)  →  logits (1, vocab)    │
# └─────────────────────────────────────────────────────────┘
#                               ↓
# ┌─────────────────────────────────────────────────────────┐
# │ 3단계: 두 번째 토큰 생성                                  │
# ├─────────────────────────────────────────────────────────┤
# │ logits[:, -1] → argmax → "very"                         │
# │ 입력: ["very"]  →  모델(캐시 사용)  →  logits (1, vocab) │
# └─────────────────────────────────────────────────────────┘
#                               ↓
#                             ...

# 캐시 사용 vs 미사용 비교
# 항목|캐시 사용|캐시 미사용
# 모델 입력|새 토큰 1개|전체 시퀀스
# K,V 계산|새 토큰만|매번 전체 재계산
# 복잡도|O(n)|O(n²)
# 메모리|캐시 저장 필요|추가 메모리 없음


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
####################################################


def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    # token_ids = generate_text_simple(
    #     model=model,
    #     idx=encoded_tensor,
    #     max_new_tokens=200,
    #     context_size=GPT_CONFIG_124M["context_length"]
    # )

    ####################################################
    # NEW
    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=200,
    )
    ####################################################

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
