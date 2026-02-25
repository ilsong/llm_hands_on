# Mixture of Experts (MoE) vs Dense FFN

**이 폴더는 일반 FFN(Dense)과 MoE(Mixture of Experts) FFN을 비교합니다.**

---

## 개요

MoE(Mixture of Experts)는 하나의 큰 FFN 대신 **여러 개의 작은 Expert**를 두고,
각 토큰마다 **Gate Network가 Top-K개의 Expert만 선택**해 연산하는 방식입니다.

**핵심 아이디어:** 파라미터는 N배 늘리되, 실제 연산은 K/N배만 수행
→ 더 큰 모델 용량을 적은 연산 비용으로 달성

---

## 파일 구성

| 파일 | 설명 |
|------|------|
| [`gpt_with_kv_ffn.py`](gpt_with_kv_ffn.py) | KV 캐시 + Dense FFN (SwiGLU) |
| [`gpt_with_kv_moe.py`](gpt_with_kv_moe.py) | KV 캐시 + MoE FFN (SwiGLU Expert × N) |

---

## Dense FFN vs MoE FFN 구조 비교

```
Dense FFN:
  입력 x (768)
       │
  [하나의 FFN]   ← 모든 토큰이 동일한 파라미터 사용
       │
     출력

MoE FFN:
  입력 x (768)
       │
  ┌────┴────┐
  │         │
  ▼         ▼
[Gate]   [Expert 1] [Expert 2] ... [Expert N]
 768→N        │           │              │
  │           └───────────┴──────────────┘
  ▼                       │
Top-K 선택 ──────────────►│
(예: K=2)                 │
  │                       ▼
  └──► 선택된 K개 Expert만 연산 → 가중합 → 출력
```

---

## FFN 클래스 상세 비교

### Dense FFN (gpt_with_kv_ffn.py) — SwiGLU

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        # 단일 FFN: 레이어 3개
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)  # 768 → 3072
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)  # 768 → 3072 (게이트)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], bias=False)  # 3072 → 768

    def forward(self, x):
        # SwiGLU: silu(fc1(x)) * fc2(x) → fc3
        return self.fc3(torch.nn.functional.silu(self.fc1(x)) * self.fc2(x))
```

### MoE FFN (gpt_with_kv_moe.py) — Gate + N개 SwiGLU Expert

```python
class MoEFeedForward(nn.Module):
    def __init__(self, cfg):
        self.num_experts = cfg["num_experts"]              # Expert 총 수 (예: 8)
        self.num_experts_per_tok = cfg["num_experts_per_tok"]  # 토큰당 선택 수 (예: 2)

        # Gate: 어떤 Expert를 쓸지 점수 계산
        self.gate = nn.Linear(cfg["emb_dim"], cfg["num_experts"], bias=False)  # 768 → 8

        # Expert마다 독립적인 SwiGLU FFN
        self.fc1 = nn.ModuleList([nn.Linear(emb_dim, hidden_dim) for _ in range(num_experts)])
        self.fc2 = nn.ModuleList([nn.Linear(emb_dim, hidden_dim) for _ in range(num_experts)])
        self.fc3 = nn.ModuleList([nn.Linear(hidden_dim, emb_dim) for _ in range(num_experts)])
```

---

## MoE forward() — 5단계 처리 흐름

### Step 1: Gate 점수 계산 및 Top-K Expert 선택

```python
scores = self.gate(x)                                        # (b, seq, num_experts)
topk_scores, topk_indices = torch.topk(scores, K, dim=-1)   # 상위 K개 선택
topk_probs = torch.softmax(topk_scores, dim=-1)             # 선택된 Expert 가중치 (합=1)
```

예시 (num_experts=8, K=2):
```
토큰 "cat"의 Gate 점수:
  Expert: [E0,  E1,  E2,   E3,  E4,  E5,   E6,  E7]
  점수:   [0.1, 0.3, 0.05, 0.8, 0.2, 0.15, 0.7, 0.1]
  Top-2 선택: E3(0.8), E6(0.7)
  softmax: [0.52, 0.48]  ← 가중치
```

### Step 2: 토큰 평탄화 (배치 처리 준비)

```python
x_flat = x.reshape(batch * seq_len, -1)              # (b*T, emb_dim)
out_flat = torch.zeros(batch * seq_len, emb_dim, ...)
unique_experts = torch.unique(topk_indices_flat)      # 실제 사용되는 Expert만
```

### Step 3: Expert별로 해당 토큰들만 모아서 처리 (Sparse Computation)

```python
for expert_id in unique_experts:
    # 이 Expert를 선택한 토큰들만 추출
    token_mask = (topk_indices_flat == expert_id).any(dim=-1)
    selected_idx = token_mask.nonzero(as_tuple=False).squeeze(-1)
    expert_input = x_flat.index_select(0, selected_idx)  # 선택된 토큰만
```

### Step 4: SwiGLU Expert 연산

```python
    hidden = silu(self.fc1[expert_id](expert_input)) * self.fc2[expert_id](expert_input)
    expert_out = self.fc3[expert_id](hidden)
```

### Step 5: 가중치를 곱해서 결과에 누적 (가중합)

```python
    out_flat.index_add_(0, selected_idx, expert_out * selected_probs.unsqueeze(-1))

# 최종 출력 = Σ (prob_i × Expert_i(x))
# 예: "cat" = 0.52 × E3("cat") + 0.48 × E6("cat")
```

---

## TransformerBlock — FFN 선택 로직

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        # num_experts > 0이면 MoE, 아니면 Dense FFN
        self.ff = MoEFeedForward(cfg) if cfg["num_experts"] > 0 else FeedForward(cfg)
```

하나의 파일(`gpt_with_kv_moe.py`)에서 `num_experts=0`으로 설정하면 Dense FFN과 동일하게 동작합니다.

---

## 파라미터 수 vs 실제 연산량 (emb_dim=768, hidden_dim=3072)

### Dense FFN (파라미터 기준)

| 레이어 | 파라미터 수 |
|--------|------------|
| fc1 (768→3072) | 2,359,296 |
| fc2 (768→3072) | 2,359,296 |
| fc3 (3072→768) | 2,359,296 |
| **합계** | **7,077,888** |

### MoE FFN (8 Expert, Top-2 기준)

| 항목 | 값 |
|------|-----|
| Expert 1개 파라미터 | 7,077,888 |
| Expert 8개 전체 파라미터 | 7,077,888 × 8 = **56,623,104** |
| Gate 파라미터 | 768 × 8 = 6,144 |
| 토큰 1개당 실제 연산 Expert 수 | **2개** (Top-2) |
| 토큰 1개당 실제 연산량 | Dense FFN × **2/8 = 25%** |

```
파라미터 수:   Dense 1× → MoE 8×  (8배 증가)
실제 연산량:   Dense 1× → MoE 0.25×  (75% 감소)

→ "파라미터는 8배지만 연산은 1/4만 한다!"
```

---

## SwiGLU 활성화 함수 (두 파일 공통)

두 파일 모두 기존 GELU 대신 **SwiGLU**를 사용합니다:

```python
# GELU (기존):
output = GELU(fc1(x))

# SwiGLU (현재):
output = silu(fc1(x)) * fc2(x)
#         └─ gate ─┘   └─ 값 ─┘

# silu(x) = x * sigmoid(x)  (Swish 활성화)
```

SwiGLU는 게이트 메커니즘으로 정보를 선택적으로 통과시켜 표현력이 더 높습니다.
MoE와 직접 비교하기 위해 Dense FFN도 SwiGLU로 통일했습니다.

---

## Config 차이

**Dense FFN (gpt_with_kv_ffn.py):**
```python
GPT_CONFIG = {
    "emb_dim": 768,
    "hidden_dim": 768 * 4,   # 3072
    "n_heads": 12,
    "n_layers": 12,
    # num_experts 없음
}
```

**MoE FFN (gpt_with_kv_moe.py):**
```python
GPT_CONFIG = {
    "emb_dim": 768,
    "hidden_dim": 768 * 4,       # 각 Expert의 hidden_dim
    "n_heads": 12,
    "n_layers": 12,
    "num_experts": 8,            # ← Expert 총 수 (0이면 Dense FFN)
    "num_experts_per_tok": 2,    # ← 토큰당 선택할 Expert 수 (Top-K)
}
```

---

## 성능 측정 (두 파일 공통)

두 파일 모두 FFN 레이어의 시간·메모리를 측정하는 프로파일링 코드가 포함되어 있습니다:

```python
# TransformerBlock.forward() 내부
start = time.perf_counter()
x = self.ff(x)                        # FFN or MoE 실행
FFN_TIME_MS.append((time.perf_counter() - start) * 1000.0)

# GPU 메모리 측정
peak_mem = torch.cuda.max_memory_allocated()
FFN_MEM_BYTES.append(peak_mem - base_mem)
```

실행 후 출력:
```
Avg FFN time/call: 0.123 ms          # Dense FFN
Avg MoE FF time/call: 0.456 ms       # MoE (Expert 순회로 인한 오버헤드)
```

---

## Dense FFN vs MoE 종합 비교

| 항목 | Dense FFN | MoE FFN (8 Expert, Top-2) |
|------|-----------|--------------------------|
| FFN 클래스 | `FeedForward` | `MoEFeedForward` |
| 활성화 함수 | SwiGLU | SwiGLU (Expert마다) |
| Gate Network | 없음 | `nn.Linear(emb_dim, num_experts)` |
| Expert 수 | 1 (전체) | N개 (num_experts) |
| 토큰당 사용 Expert | 1 (항상) | K개 (Top-K 선택) |
| 파라미터 수 | 기준 (1×) | N× (Expert 수 배) |
| 실제 연산량 | 100% | K/N × 100% = 25% |
| 토큰별 전문화 | 없음 | Gate로 Expert 선택 |
| 추가 연산 | 없음 | Gate 계산 + 토큰 라우팅 |
| 코드 복잡도 | 낮음 | 높음 |

---

## Trade-off

| 장점 | 단점 |
|------|------|
| 파라미터 대비 연산량 절감 | Expert 수만큼 파라미터 증가 (메모리) |
| 토큰마다 전문화된 처리 | Gate 학습 필요 (학습 불안정 가능) |
| 더 큰 모델 용량 확보 | Load balancing 문제 (일부 Expert 편중) |
| 동일 연산량으로 더 높은 품질 | 구현 복잡도 증가 |

---

## 실행 방법

```bash
# Dense FFN 버전
python gpt_with_kv_ffn.py --hidden_dim 3072 --max_new_tokens 200

# MoE 버전 (8 Expert, Top-2)
python gpt_with_kv_moe.py --num_experts 8 --num_experts_per_tok 2 --max_new_tokens 200

# MoE 버전 (num_experts=0이면 Dense FFN과 동일)
python gpt_with_kv_moe.py --num_experts 0 --max_new_tokens 200

# KV 캐시 없이 실행
python gpt_with_kv_moe.py --num_experts 8 --no_kv_cache
```

---

## 실제 모델 적용 사례

| 모델 | Expert 수 | Top-K | 특징 |
|------|----------|-------|------|
| Switch Transformer | N | **Top-1** | 가장 단순한 MoE |
| Mixtral 8×7B | **8** | **2** | 오픈소스 MoE 대표 |
| GPT-4 (추정) | 16 | 2 | MoE 구조 사용 추정 |
| DeepSeek-V2 | 160 | 6 | 극단적 MoE 분할 |
| Qwen1.5-MoE | 60 | 4 | 효율적 MoE |

---

## 참고 자료

- [Mixtral 8x7B 논문](https://arxiv.org/abs/2401.04088) — 대표적 오픈소스 MoE
- [Switch Transformer 논문](https://arxiv.org/abs/2101.03961) — Top-1 MoE 제안
- [SwiGLU 논문](https://arxiv.org/abs/2002.05202) — SwiGLU 활성화 함수
- [LLMs-from-scratch GitHub](https://github.com/rasbt/LLMs-from-scratch)
