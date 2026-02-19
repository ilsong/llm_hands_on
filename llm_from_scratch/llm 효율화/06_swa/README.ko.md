# Sliding Window Attention (SWA)

**각 토큰이 모든 이전 토큰을 참조하는 대신, 최근 W개 토큰만 참조합니다.**

---

## 개요

일반 Causal Attention은 시퀀스가 길어질수록 KV 캐시가 무한정 증가합니다.
SWA는 **고정된 윈도우 크기**만 유지하여 메모리를 일정하게 제한합니다.

```
일반 Attention (모든 이전 토큰 참조):
  T0  T1  T2  T3  T4  T5  T6  T7
Q7: ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ← 8개 전부 참조

Sliding Window Attention (W=4):
  T0  T1  T2  T3  T4  T5  T6  T7
Q7: ✗   ✗   ✗   ✗   ✓   ✓   ✓   ✓   ← 최근 4개만 참조
    └── 윈도우 밖 ──┘  └── 윈도우 안 ──┘
```

---

## 핵심 코드 1: KV 캐시 트리밍

MHA는 캐시를 계속 누적하지만, SWA는 윈도우 크기 초과 시 오래된 토큰을 삭제합니다.

**MHA** — 캐시를 무한정 누적:
```python
# gpt_with_kv_mha.py
self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)  # 계속 쌓임
self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
```

**SWA** — 윈도우 크기 초과 시 앞을 잘라냄:
```python
# gpt_with_kv_swa.py
self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
self.cache_v = torch.cat([self.cache_v, values_new], dim=1)

# 핵심: 윈도우 초과 시 오래된 토큰 삭제
if self.sliding_window_size is not None:
    if self.cache_k.size(1) > self.sliding_window_size:
        self.cache_k = self.cache_k[:, -self.sliding_window_size:, :, :]
        self.cache_v = self.cache_v[:, -self.sliding_window_size:, :, :]
```

```
윈도우=4 일 때 캐시 변화:
  Before: [K0, K1, K2, K3, K4, K5]  (6개)
  After:  [          K2, K3, K4, K5]  (최근 4개만)
```

---

## 핵심 코드 2: 슬라이딩 윈도우 마스크

MHA는 단순히 `q_pos < k_pos`(미래 토큰)만 마스킹하지만,
SWA는 **윈도우 밖 과거 토큰**도 추가로 마스킹합니다.

**MHA** — 미래 토큰만 마스킹:
```python
# gpt_with_kv_mha.py
mask_bool = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)
# diff < 0 인 경우만 마스킹 (미래)
```

**SWA** — 미래 + 윈도우 밖 과거 모두 마스킹:
```python
# gpt_with_kv_swa.py
diff = q_positions.unsqueeze(-1) - k_positions.unsqueeze(0)
# diff[i,j] = Q위치[i] - K위치[j]

mask_bool = (diff < 0) | (diff >= W)
#            └── 미래  ──┘  └── 윈도우 밖 과거 ──┘
```

```
예시: Q위치=7, K위치=[4,5,6,7], W=4
  diff  = [7-4, 7-5, 7-6, 7-7] = [3, 2, 1, 0]
  mask  = [F,   F,   F,   F  ]   ← 모두 윈도우 내 → 참조 가능

예시: Q위치=7, K위치=[0,1,2,3], W=4
  diff  = [7, 6, 5, 4]
  mask  = [T, T, T, T]            ← diff >= 4 → 모두 마스킹
```

> **절대 위치 보정**: 캐시를 트리밍하면 K의 시작 위치가 바뀌므로,
> 삭제된 토큰 수(`dropped`)를 계산해 K의 절대 위치를 보정합니다.
> ```python
> dropped = max(0, total_len - k_len_now)
> k_start_pos_abs = (self.ptr_current_pos - old_len) + dropped
> ```

---

## 핵심 코드 3: K:1 스케줄링

모든 레이어에 SWA를 적용하면 먼 토큰 정보가 완전히 소실됩니다.
**K개 SWA + 1개 일반 어텐션**을 반복해 정보 손실을 방지합니다.

```python
# gpt_with_kv_swa.py — GPTModel.__init__
for i in range(cfg["n_layers"]):
    blk = TransformerBlock(cfg)
    K = int(window_stride)        # sliding_window_stride 값

    group = K + 1                 # 그룹 크기 (예: K=2 → group=3)
    use_swa = (i % group) < K     # 그룹 내 처음 K개는 SWA

    blk.att.sliding_window_size = window_size if use_swa else None
```

```
sliding_window_stride=2 (2:1 스케줄):

Layer 0: SWA    (0 % 3 = 0 < 2)
Layer 1: SWA    (1 % 3 = 1 < 2)
Layer 2: 일반   (2 % 3 = 2 ≥ 2) ← 전체 컨텍스트 참조
Layer 3: SWA    (3 % 3 = 0 < 2)
Layer 4: SWA    (4 % 3 = 1 < 2)
Layer 5: 일반   (5 % 3 = 2 ≥ 2)
```

`sliding_window_size=None`이면 마스크가 `diff < 0`만 적용 → 일반 MHA와 동일하게 동작합니다.

---

## MHA vs SWA 비교

| 항목 | MHA | SWA |
|------|-----|-----|
| 어텐션 클래스 | `MultiHeadAttention` | `MultiHeadAttentionWithSWA` |
| KV 캐시 크기 | O(시퀀스 길이) — 무한 증가 | O(윈도우 크기) — 고정 |
| 참조 범위 | 모든 이전 토큰 | 최근 W개 토큰 |
| 마스킹 조건 | `q < k` (미래만) | `diff < 0 \| diff ≥ W` |
| 절대 위치 추적 | `ptr_current_pos` | `ptr_current_pos` + `dropped` 보정 |
| 레이어 스케줄 | 없음 | K:1 스케줄 (`sliding_window_stride`) |

---

## Config 차이

**MHA:**
```python
GPT_CONFIG = {
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    # sliding_window 설정 없음
}
```

**SWA:**
```python
GPT_CONFIG = {
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "sliding_window_size": 1024,    # ← 윈도우 크기 (참조할 최대 토큰 수)
    "sliding_window_stride": 2,     # ← K:1 스케줄 (K=2: 2개 SWA + 1개 일반)
}
```

---

## 메모리 효과

| 항목 | 일반 Attention | SWA (W=1024) |
|------|---------------|--------------|
| KV 캐시 크기 | O(n) — 시퀀스와 비례 | O(W) — 고정 |
| 어텐션 연산 | O(n²) | O(n × W) |

시퀀스가 아무리 길어져도 캐시는 W개로 고정 → 스트리밍/실시간 생성에 적합합니다.

---

## Trade-off

| 장점 | 단점 |
|------|------|
| KV 캐시 고정 크기 | 먼 토큰 직접 참조 불가 |
| 긴 시퀀스도 메모리 일정 | K:1 스케줄 튜닝 필요 |
| 추론 속도 향상 | 일부 장거리 의존성 손실 가능 |

---

## 실행 방법

```bash
# SWA 버전 (윈도우=1024, 2:1 스케줄)
python gpt_with_kv_swa.py --sliding_window_size 1024 --sliding_window_stride 2

# MHA 버전 (비교용)
python gpt_with_kv_mha.py --max_new_tokens 200
```

---

## 실제 모델 적용 사례

| 모델 | 윈도우 크기 | 스케줄 |
|------|------------|--------|
| Mistral 7B | 4096 | 전체 SWA |
| Gemma 2 | 4096 | 1:1 (SWA-일반 교대) |
| Phi-3 | 2048 | K:1 스케줄 |

---

## 참고 자료

- [Mistral 7B 논문](https://arxiv.org/abs/2310.06825) — SWA 대표 적용 사례
- [Longformer 논문](https://arxiv.org/abs/2004.05150) — Sliding Window Attention 제안
- [LLMs-from-scratch GitHub](https://github.com/rasbt/LLMs-from-scratch)
