# Multi-Head Latent Attention (MLA) vs Multi-Head Attention (MHA)

**이 폴더는 MHA와 MLA(DeepSeek-V2에서 도입)를 비교합니다.**

---

## 개요

MLA(Multi-Head Latent Attention)는 K와 V를 각각 캐싱하는 대신,
K/V 두 개를 하나의 **저차원 latent 벡터 C**로 압축해 캐싱합니다.
추론 시에는 C에서 K와 V를 복원합니다.

이 방식으로 KV 캐시 크기를 MHA 대비 최대 **94%** 절감할 수 있어,
100K+ 토큰의 긴 컨텍스트 처리에 효과적입니다.

> DeepSeek-V2, DeepSeek-V3에서 채택된 기법입니다.

---

## 핵심 아이디어: 저차원 Latent 벡터로 K/V 공유 압축

```
MHA 방식 (캐시 2개):
  입력(768) ──→ W_key   (768→768) ──→ K(768) ──→ [캐시 K 저장]
  입력(768) ──→ W_value (768→768) ──→ V(768) ──→ [캐시 V 저장]
  토큰당 캐시: 768 + 768 = 1536 dim

MLA 방식 (캐시 1개):
  입력(768) ──→ W_DKV   (768→96)  ──→ C(96)  ──→ [캐시 C 저장]  ← 다운 프로젝션
                                          │
                         ┌────────────────┴────────────────┐
                         ▼                                 ▼
                   W_UK (96→768)                    W_UV (96→768)   ← 업 프로젝션
                         │                                 │
                      K(768)                           V(768)
  토큰당 캐시: 96 dim 만 저장!
```

---

## 파일 구성

| 파일 | 설명 |
|------|------|
| [`gpt_with_kv_mha.py`](gpt_with_kv_mha.py) | KV 캐시 + MHA (Multi-Head Attention) |
| [`gpt_with_kv_mla.py`](gpt_with_kv_mla.py) | KV 캐시 + MLA (Multi-Head Latent Attention) |

---

## 상세 코드 비교

### 1. 어텐션 클래스 및 프로젝션 레이어

**MHA** — K, V 각각 독립 프로젝션:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
        self.W_query = nn.Linear(d_in, d_out)   # 768 → 768
        self.W_key   = nn.Linear(d_in, d_out)   # 768 → 768  (K 전용)
        self.W_value = nn.Linear(d_in, d_out)   # 768 → 768  (V 전용)
        self.out_proj = nn.Linear(d_out, d_out)
        # 캐시: cache_k, cache_v (2개)
```

**MLA** — K/V를 하나의 latent로 압축:

```python
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads,
                 qkv_bias=False, latent_dim=None):
        self.latent_dim = latent_dim if latent_dim is not None else max(16, d_out // 8)
        #                                                             기본값: 768//8 = 96

        self.W_query = nn.Linear(d_in, d_out)               # 768 → 768  (Q는 동일)
        self.W_DKV   = nn.Linear(d_in, self.latent_dim)     # 768 → 96   ← 다운 프로젝션
        self.W_UK    = nn.Linear(self.latent_dim, d_out)    # 96  → 768  ← K 업 프로젝션
        self.W_UV    = nn.Linear(self.latent_dim, d_out)    # 96  → 768  ← V 업 프로젝션
        self.out_proj = nn.Linear(d_out, d_out)
        # 캐시: cache_c_kv (1개, latent만 저장)
```

---

### 2. 캐시 버퍼

**MHA** — K와 V를 각각 별도 캐시:

```python
self.register_buffer("cache_k", None, persistent=False)   # K 캐시
self.register_buffer("cache_v", None, persistent=False)   # V 캐시
self.ptr_current_pos = 0
```

**MLA** — latent 벡터 C 하나만 캐시:

```python
self.register_buffer("cache_c_kv", None, persistent=False)  # latent C 캐시 (K/V 공유)
self.ptr_current_pos = 0
```

---

### 3. forward() — 핵심 흐름 차이

**MHA** ([gpt_with_kv_mha.py:42](gpt_with_kv_mha.py)):

```python
def forward(self, x, use_cache=False):
    # 1) K, V를 입력에서 직접 계산
    keys_new   = self.W_key(x)    # (b, T, 768)
    values_new = self.W_value(x)  # (b, T, 768)
    queries    = self.W_query(x)  # (b, T, 768)

    # 2) reshape: (b, T, 768) → (b, T, num_heads, head_dim)
    keys_new   = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
    values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)

    # 3) KV 캐시 업데이트 (K와 V 각각)
    if use_cache:
        self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
        self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v

    # 4) transpose 후 어텐션 계산
    keys   = keys.transpose(1, 2)
    values = values.transpose(1, 2)
```

**MLA** ([gpt_with_kv_mla.py:144](gpt_with_kv_mla.py)):

```python
def forward(self, x, use_cache=False):
    # 1) Q는 직접, K/V는 latent로 압축
    queries_all = self.W_query(x)   # (b, T, 768)
    latent_new  = self.W_DKV(x)    # (b, T, 96)  ← 다운 프로젝션

    # 2) latent 캐시 업데이트 (단 하나의 캐시)
    if use_cache:
        latent_total = torch.cat([self.cache_c_kv, latent_new], dim=1)
        self.cache_c_kv = latent_total
    else:
        latent_total = latent_new

    # 3) latent에서 K, V를 업 프로젝션으로 복원 (추론 시 계산)
    keys_all   = self.W_UK(latent_total)  # (b, T_total, 768)
    values_all = self.W_UV(latent_total)  # (b, T_total, 768)

    # 4) reshape 후 어텐션 계산 (helper 메서드 사용)
    queries = self._reshape_to_heads(queries_all, num_heads, head_dim)
    keys    = self._reshape_to_heads(keys_all,    num_heads, head_dim)
    values  = self._reshape_to_heads(values_all,  num_heads, head_dim)
```

> **MLA의 `_reshape_to_heads` 헬퍼 메서드** (MHA에는 없음):
> ```python
> @staticmethod
> def _reshape_to_heads(x, num_heads, head_dim):
>     bsz, num_tokens, _ = x.shape
>     return x.view(bsz, num_tokens, num_heads, head_dim).transpose(1, 2).contiguous()
> ```

---

### 4. Config — `latent_dim` 추가

**MHA:**
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": False,
    # latent_dim 없음
}
```

**MLA:**
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "latent_dim": args.latent_dim,  # ← 추가 (None이면 d_out//8 = 96 자동 설정)
}
```

---

## 프로젝션 파라미터 수 비교 (emb_dim=768, latent_dim=96)

| 레이어 | MHA | MLA |
|--------|-----|-----|
| W_query | 768 × 768 = **589,824** | 768 × 768 = **589,824** |
| W_key | 768 × 768 = **589,824** | — |
| W_value | 768 × 768 = **589,824** | — |
| W_DKV (다운) | — | 768 × 96 = **73,728** |
| W_UK (K 업) | — | 96 × 768 = **73,728** |
| W_UV (V 업) | — | 96 × 768 = **73,728** |
| out_proj | 768 × 768 = **589,824** | 768 × 768 = **589,824** |
| **합계** | **2,359,296** | **1,400,832** |

K/V 관련 파라미터: MHA 1,179,648 → MLA 221,184 (**81% 절감**)

---

## KV 캐시 크기 비교 (토큰당)

| 방식 | 캐싱 내용 | 토큰당 캐시 크기 | MHA 대비 절약 |
|------|-----------|-----------------|---------------|
| MHA | K(768) + V(768) | **1,536 dim** | 기준 |
| GQA (2그룹) | K(128) + V(128) | 256 dim | 83% |
| **MLA** | **C(96) 단일 latent** | **96 dim** | **94%** |

```
캐시 크기 시각화 (1536 기준):

MHA:  ████████████████████████████████████████  (1536)
GQA:  ██████                                    (256)
MLA:  ███                                       (96)  ← 가장 작음!
```

---

## Trade-off 분석

| | MHA | MLA |
|--|-----|-----|
| KV 캐시 크기 | 크다 (1536/token) | 매우 작다 (96/token) |
| 추론 시 추가 연산 | 없음 | W_UK, W_UV 업 프로젝션 필요 |
| 긴 시퀀스 효율 | 낮음 (캐시 폭발) | 높음 (캐시 절약) |
| 코드 복잡도 | 낮음 | 중간 |
| 표현력 | 높음 | K/V가 공유 정보 기반 (약간 제약) |
| latent_dim 튜닝 | 불필요 | 필요 |

---

## MHA vs MLA 데이터 흐름 전체 비교

```
[ MHA ]
입력 x (b, T, 768)
  │
  ├─→ W_query (768→768) ──→ Q (b, T, 768) ──→ reshape (b, 12, T, 64)
  │
  ├─→ W_key   (768→768) ──→ K_new ──→ [cache_k에 누적] ──→ reshape (b, 12, T_total, 64)
  │
  └─→ W_value (768→768) ──→ V_new ──→ [cache_v에 누적] ──→ reshape (b, 12, T_total, 64)
                                                              │
                                               Q @ K^T → softmax → @ V → out_proj


[ MLA ]
입력 x (b, T, 768)
  │
  ├─→ W_query (768→768) ──→ Q (b, T, 768) ──→ reshape (b, 12, T, 64)
  │
  └─→ W_DKV   (768→96)  ──→ C_new (b, T, 96) ──→ [cache_c_kv에 누적]
                                                        │
                                         ┌──────────────┴──────────────┐
                                         ▼                             ▼
                                   W_UK (96→768)               W_UV (96→768)
                                         │                             │
                                  K (b, T_total, 768)        V (b, T_total, 768)
                                         │                             │
                                    reshape (b, 12, T_total, 64)       │
                                                              reshape (b, 12, T_total, 64)
                                                              │
                                               Q @ K^T → softmax → @ V → out_proj
```

---

## 실행 방법

```bash
# MHA 버전
python gpt_with_kv_mha.py --n_heads 12 --max_new_tokens 200

# MLA 버전 (기본 latent_dim = d_out//8 = 96)
python gpt_with_kv_mla.py --n_heads 12 --max_new_tokens 200

# MLA 버전 (latent_dim 직접 지정)
python gpt_with_kv_mla.py --n_heads 12 --latent_dim 64 --max_new_tokens 200
```

---

## 실제 모델 적용 사례

| 모델 | 어텐션 방식 | 특징 |
|------|------------|------|
| GPT-2 | MHA | 기본 표준 |
| Llama 2 7B | MHA | 표준 |
| Llama 2 70B | GQA | 8 KV 그룹 |
| **DeepSeek-V2** | **MLA** | latent 압축, 최초 상용 채택 |
| **DeepSeek-V3** | **MLA** | 개선된 MLA |

---

## 참고 자료

- [DeepSeek-V2 논문](https://arxiv.org/abs/2405.04434) — MLA 최초 제안
- [HuggingFace DeepSeek MLA 구현](https://huggingface.co/bird-of-paradise/deepseek-mla)
- [LLMs-from-scratch GitHub](https://github.com/rasbt/LLMs-from-scratch)
