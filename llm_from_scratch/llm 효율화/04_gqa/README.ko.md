# Grouped Query Attention (GQA) vs Multi-Head Attention (MHA)

**이 폴더는 MHA에 KV 캐시를 적용한 버전(MHA)과 GQA를 적용한 버전을 비교합니다.**

---

## 개요

GQA(Grouped Query Attention)는 K/V 헤드 수를 줄여 메모리와 연산량을 절약하는 어텐션 기법입니다.
MHA의 품질은 거의 유지하면서 KV 캐시 크기를 대폭 줄일 수 있어,
Llama 2, Mistral, Gemma 등 최신 LLM에서 표준으로 채택되고 있습니다.

---

## MHA vs GQA vs MQA 비교 (개념)

```
MHA (Multi-Head Attention)       GQA (Grouped Query)              MQA (Multi-Query)
Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8         Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8         Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8
 ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓           ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓           ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
K1 K2 K3 K4 K5 K6 K7 K8         K1 K1 K1 K1 K2 K2 K2 K2         K1 K1 K1 K1 K1 K1 K1 K1
V1 V2 V3 V4 V5 V6 V7 V8         V1 V1 V1 V1 V2 V2 V2 V2         V1 V1 V1 V1 V1 V1 V1 V1
   8Q, 8K, 8V                      8Q, 2K, 2V (그룹=2)               8Q, 1K, 1V
```

- **MHA**: 모든 Q 헤드가 각자 고유한 K, V를 가짐 → 표현력 최대, 메모리 최대
- **GQA**: Q 헤드를 그룹으로 묶고, 그룹 내 Q들이 같은 K, V를 공유
- **MQA**: 모든 Q가 단 하나의 K, V를 공유 → 메모리 최소, 품질 저하 가능

---

## 파일 구성

| 파일 | 설명 |
|------|------|
| [`gpt_with_kv_mha.py`](gpt_with_kv_mha.py) | KV 캐시 + MHA (Multi-Head Attention) |
| [`gpt_with_kv_gqa.py`](gpt_with_kv_gqa.py) | KV 캐시 + GQA (Grouped Query Attention) |

---

## 상세 코드 비교

### 1. 어텐션 클래스 및 생성자 파라미터

**gpt_with_kv_mha.py** — `MultiHeadAttention`:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
        self.W_key   = nn.Linear(d_in, d_out)   # 768 → 768
        self.W_value = nn.Linear(d_in, d_out)   # 768 → 768
        self.W_query = nn.Linear(d_in, d_out)   # 768 → 768
```

**gpt_with_kv_gqa.py** — `GroupedQueryAttention`:

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, num_kv_groups, dtype=None, qkv_bias=False):
        self.head_dim   = d_out // num_heads              # 768 // 12 = 64
        self.group_size = num_heads // num_kv_groups      # 12 // 2 = 6

        # K, V: 그룹 수만큼만 생성 (메모리 절약 핵심)
        self.W_key   = nn.Linear(d_in, num_kv_groups * self.head_dim)  # 768 → 128
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim)  # 768 → 128
        # Q: 전체 헤드 수 유지 (표현력 유지 핵심)
        self.W_query = nn.Linear(d_in, d_out)                          # 768 → 768
```

---

### 2. Reshape — K/V 텐서 형태 차이

**MHA** — K, V 모두 전체 헤드 수:

```python
# (b, num_tokens, d_out) → (b, num_tokens, num_heads, head_dim)
keys_new   = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)   # 12 헤드
values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim) # 12 헤드
queries    = queries.view(b, num_tokens, self.num_heads, self.head_dim)     # 12 헤드

# 이후 transpose: dim=1, dim=2 교환
keys_new = keys_new.transpose(1, 2)  # (b, 12, num_tokens, 64)
```

**GQA** — K, V는 그룹 수만큼, Q는 전체 헤드 수:

```python
# Query: 전체 12헤드 유지
queries  = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
#         (b, 12, num_tokens, 64)

# Key/Value: 2그룹만 생성 (이미 transpose된 상태로 저장)
keys_new   = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
#            (b, 2, num_tokens, 64)
```

---

### 3. KV 캐시 연결 방향 (dim 차이)

이미 transpose 전/후 시점이 달라 캐시를 붙이는 차원이 다릅니다.

**MHA** — transpose 전에 캐시 연결 → `dim=1` (num_tokens 축):

```python
# keys_new shape: (b, num_tokens, num_heads, head_dim)  ← transpose 전
self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)  # num_tokens 축
keys = self.cache_k.transpose(1, 2)  # 이후에 transpose
```

**GQA** — transpose 후에 캐시 연결 → `dim=2` (num_tokens 축):

```python
# keys_new shape: (b, num_kv_groups, num_tokens, head_dim)  ← transpose 후
self.cache_k = torch.cat([self.cache_k, keys_new], dim=2)  # num_tokens 축
keys_base = self.cache_k  # 이미 transpose 완료
```

---

### 4. GQA 핵심 — `repeat_interleave`로 K/V 확장

GQA에만 존재하는 핵심 로직으로, 그룹 수만큼이던 K/V를 Query 헤드 수로 확장합니다.

```python
# group_size = num_heads // num_kv_groups = 12 // 2 = 6
keys   = keys_base.repeat_interleave(self.group_size, dim=1)
values = values_base.repeat_interleave(self.group_size, dim=1)
```

```
repeat_interleave 동작 (group_size=6, num_kv_groups=2):

캐시의 K:  [K1,  K2]            → 2 groups
확장 후:   [K1, K1, K1, K1, K1, K1,  K2, K2, K2, K2, K2, K2]  → 12 heads
            └─── Q1~Q6 공유 ───┘   └─── Q7~Q12 공유 ───┘
```

> **`repeat` vs `repeat_interleave` 차이**
>
> ```python
> # repeat_interleave (올바른 방법): 각 요소를 연속으로 반복
> [K1, K2] → [K1, K1, K2, K2]  # group_size=2일 때
>
> # repeat (잘못된 방법): 전체 배열을 반복
> [K1, K2] → [K1, K2, K1, K2]  # 헤드-그룹 매핑이 틀어짐
> ```

---

### 5. GQA의 캐시 초기화 추가 로직

GQA는 `use_cache=False`일 때 캐시가 남아있으면 명시적으로 초기화합니다:

```python
# MHA: 별도 초기화 로직 없음

# GQA: use_cache=False인데 캐시가 남아 있으면 강제 초기화
else:
    keys_base, values_base = keys_new, values_new
    if self.cache_k is not None or self.cache_v is not None:
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0
```

---

### 6. Config — `n_kv_groups` 추가

**MHA:**
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": False,
    # n_kv_groups 없음
}
```

**GQA:**
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "n_kv_groups": 2,   # ← GQA 전용 파라미터 추가
}
```

---

## 메모리 절약 효과 (emb_dim=768, num_heads=12, num_kv_groups=2)

| 항목 | MHA (12헤드) | GQA (2그룹) | 절약률 |
|------|-------------|-------------|--------|
| `W_key` 파라미터 수 | 768 × 768 = 589,824 | 768 × 128 = 98,304 | **83%** |
| `W_value` 파라미터 수 | 768 × 768 = 589,824 | 768 × 128 = 98,304 | **83%** |
| KV 캐시 크기 | 12 × seq_len × 64 | 2 × seq_len × 64 | **83%** |
| `W_query` 파라미터 수 | 768 × 768 = 589,824 | 768 × 768 = 589,824 | 0% (동일) |

---

## 데이터 흐름 시각화

```
입력 x (b, seq_len, 768)
     │
     ├─→ W_query (768 → 768) ─→ reshape (12헤드, 64dim) ─→ Q: (b, 12, seq, 64)
     │
     ├─→ W_key   (768 → 128) ─→ reshape (2그룹,  64dim) ─→ K_raw: (b, 2, seq, 64)
     │                                                          │
     │                                          repeat_interleave(6) ↓
     │                                                     K: (b, 12, seq, 64)
     │
     └─→ W_value (768 → 128) ─→ reshape (2그룹,  64dim) ─→ V_raw: (b, 2, seq, 64)
                                                                │
                                                repeat_interleave(6) ↓
                                                           V: (b, 12, seq, 64)

Attention: Q @ K^T → softmax → @ V → out_proj
```

---

## MHA vs GQA 종합 비교

| 항목 | MHA (`gpt_with_kv_mha.py`) | GQA (`gpt_with_kv_gqa.py`) |
|------|--------------------------|--------------------------|
| 어텐션 클래스 | `MultiHeadAttention` | `GroupedQueryAttention` |
| K/V 프로젝션 출력 크기 | `d_out` (768) | `num_kv_groups × head_dim` (128) |
| Q 프로젝션 출력 크기 | `d_out` (768) | `d_out` (768) — 동일 |
| 추가 파라미터 | 없음 | `num_kv_groups` |
| K/V Reshape 헤드 수 | `num_heads` (12) | `num_kv_groups` (2) |
| Transpose 시점 | 캐시 연결 후 | 캐시 연결 전 (미리 transpose) |
| 캐시 연결 dim | `dim=1` | `dim=2` |
| K/V 확장 | 불필요 | `repeat_interleave(group_size)` |
| K/V 파라미터 절약 | 기준 | 약 83% 절약 |
| KV 캐시 절약 | 기준 | 약 83% 절약 |
| 표현력 | 최대 | MHA와 거의 동등 |

---

## 실행 방법

```bash
# MHA 버전
python gpt_with_kv_mha.py --n_heads 12 --max_new_tokens 200

# GQA 버전 (기본 2그룹)
python gpt_with_kv_gqa.py --n_heads 12 --n_kv_groups 2 --max_new_tokens 200

# GQA 버전 (4그룹 — Llama 스타일)
python gpt_with_kv_gqa.py --n_heads 12 --n_kv_groups 4 --max_new_tokens 200
```

---

## 실제 모델 적용 사례

| 모델 | 어텐션 방식 | Q 헤드 수 | KV 그룹 수 |
|------|------------|---------|----------|
| GPT-2 | MHA | 12 | 12 (=MHA) |
| Llama 2 7B | MHA | 32 | 32 (=MHA) |
| Llama 2 70B | GQA | 64 | 8 |
| Llama 3 8B | GQA | 32 | 8 |
| Mistral 7B | GQA | 32 | 8 |
| Gemma 7B | MQA | 16 | 1 (=MQA) |

---

## 참고 자료

- [GQA 논문: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [LLMs-from-scratch GitHub](https://github.com/rasbt/LLMs-from-scratch)
