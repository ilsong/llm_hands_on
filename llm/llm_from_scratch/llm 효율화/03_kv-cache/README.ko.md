# KV 캐시 (KV Cache)

**이 폴더는 GPT 모델에 KV 캐시를 추가하는 과정을 구현합니다.**

---

## 개요

KV 캐시는 추론(inference) 중 중간 계산 결과인 Key(K)와 Value(V)를 저장해 재사용하는 방법으로,
텍스트 생성 속도를 크게 향상시킵니다.

**장점:** 추론 속도 대폭 향상
**단점:** 코드 복잡도 증가, 메모리 사용량 증가, 학습(training) 중에는 사용 불가

---

## 작동 원리

LLM이 프롬프트 `"Time flies"`를 처리한다고 가정합니다.

```
Step 1: "Time flies" 입력
  → Key(K), Value(V) 계산 및 캐시에 저장

Step 2: "Time flies fast" (새 토큰 "fast" 추가)
  → "Time", "flies"의 K,V는 캐시에서 재사용
  → "fast"의 K,V만 새로 계산
```

같은 토큰의 K,V를 매번 재계산하는 것은 낭비이므로,
캐시에 저장해두고 새 토큰의 것만 계산합니다.

---

## 파일 구성

| 파일 | 설명 |
|------|------|
| [`gpt_ch04.py`](gpt_ch04.py) | KV 캐시 없는 기본 GPT 구현 |
| [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py) | KV 캐시를 추가한 최적화 버전 |

---

## gpt_ch04.py vs gpt_with_kv_cache.py 상세 비교

### 1. `MultiHeadAttention.__init__()` — 캐시 버퍼 등록

**gpt_ch04.py** — 캐시 관련 코드 없음

**gpt_with_kv_cache.py** — 캐시 버퍼 추가:

```python
self.register_buffer("cache_k", None, persistent=False)
self.register_buffer("cache_v", None, persistent=False)
self.ptr_current_pos = 0   # 현재 시퀀스 위치 포인터
```

---

### 2. `MultiHeadAttention.forward()` — 캐시 읽기/쓰기 및 마스크 처리

**gpt_ch04.py** — 매번 전체 시퀀스를 재계산:

```python
def forward(self, x):
    keys = self.W_key(x)      # 전체 시퀀스 재계산
    values = self.W_value(x)
    queries = self.W_query(x)

    # 고정 크기 마스크 슬라이싱
    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
```

**gpt_with_kv_cache.py** — 새 토큰만 계산 후 캐시에 누적:

```python
def forward(self, x, use_cache=False):   # use_cache 파라미터 추가
    keys_new = self.W_key(x)             # 새 토큰만 계산
    values_new = self.W_value(x)
    queries = self.W_query(x)

    if use_cache:
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            # 기존 캐시에 새 K,V를 이어 붙임
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v
    else:
        keys, values = keys_new, values_new

    # 마스크를 위치 포인터 기반으로 동적 슬라이싱
    if use_cache:
        mask_bool = self.mask.bool()[
            self.ptr_current_pos : self.ptr_current_pos + num_tokens_Q,
            :num_tokens_K
        ]
        self.ptr_current_pos += num_tokens_Q
    else:
        mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]
```

> **포인터(`ptr_current_pos`) 동작 예시**
>
> | 단계 | 입력 | ptr_current_pos | num_tokens_Q | num_tokens_K |
> |------|------|-----------------|-------------|-------------|
> | 프롬프트 처리 | [Hello, I, am] (3토큰) | 0 → 3 | 3 | 3 |
> | 1번째 생성 | [새토큰] (1토큰) | 3 → 4 | 1 | 4 |
> | 2번째 생성 | [새토큰] (1토큰) | 4 → 5 | 1 | 5 |

---

### 3. `GPTModel` — 블록 컨테이너와 위치 임베딩

| 항목 | gpt_ch04.py | gpt_with_kv_cache.py |
|------|-------------|----------------------|
| 블록 컨테이너 | `nn.Sequential` | `nn.ModuleList` |
| 이유 | 순차 자동 실행 | `use_cache` 인자를 각 블록에 전달해야 함 |
| 위치 임베딩 | `arange(seq_len)` — 항상 0부터 시작 | `arange(current_pos, current_pos + seq_len)` |
| 캐시 초기화 | 없음 | `reset_kv_cache()` 메서드 추가 |

**gpt_ch04.py:**

```python
self.trf_blocks = nn.Sequential(...)

def forward(self, in_idx):
    pos_embeds = self.pos_emb(torch.arange(seq_len))  # 항상 [0,1,2,3,...]
    x = self.trf_blocks(x)  # 자동 순차 실행
```

**gpt_with_kv_cache.py:**

```python
self.trf_blocks = nn.ModuleList(...)
self.current_pos = 0

def forward(self, in_idx, use_cache=False):
    # 캐시 사용 시 위치를 이어서 계산
    # 프롬프트(3토큰) 후: [3], [4], [5], ...
    if use_cache:
        pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len)
        self.current_pos += seq_len
    else:
        pos_ids = torch.arange(0, seq_len)

    for blk in self.trf_blocks:
        x = blk(x, use_cache=use_cache)  # 각 블록에 인자 전달

def reset_kv_cache(self):
    for blk in self.trf_blocks:
        blk.att.reset_cache()
    self.current_pos = 0
```

---

### 4. 생성 함수 — 핵심 동작 차이

**gpt_ch04.py** `generate_text_simple()` — 매 스텝마다 전체 시퀀스 입력:

```
Step 1: [Hello, I, am]           → 모델 → logits (3토큰 처리)
Step 2: [Hello, I, am, a]        → 모델 → logits (4토큰 처리)
Step 3: [Hello, I, am, a, very]  → 모델 → logits (5토큰 처리)
...
→ O(n²) 연산
```

**gpt_with_kv_cache.py** `generate_text_simple_cached()` — 새 토큰 1개만 입력:

```
1단계 (프롬프트 처리):
  [Hello, I, am] → 모델 (캐시에 K,V 저장) → logits

2단계 (토큰 생성 반복):
  [a]    → 모델 (캐시 참조, 1토큰만 처리) → logits
  [very] → 모델 (캐시 참조, 1토큰만 처리) → logits
  [good] → 모델 (캐시 참조, 1토큰만 처리) → logits
  ...
→ O(n) 연산
```

```python
def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    model.eval()
    with torch.no_grad():
        if use_cache:
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)  # 프롬프트 처리

            for _ in range(max_new_tokens):
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)
                logits = model(next_idx, use_cache=True)  # 새 토큰 1개만 입력
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)  # 전체 입력
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)
```

---

## 성능 비교

124M 파라미터 모델, 프롬프트 4토큰 `"Hello, I am"`, 200 토큰 생성 기준:

| | Tokens/sec |
|---|---|
| `gpt_ch04.py` (캐시 없음) | 27 |
| `gpt_with_kv_cache.py` (캐시 있음) | 144 |

약 **5배 속도 향상** (Mac Mini M4 CPU 기준)

---

## 캐시 사용 vs 미사용 종합 비교

| 항목 | 캐시 사용 | 캐시 미사용 |
|------|-----------|------------|
| 모델 입력 | 새 토큰 1개 | 전체 시퀀스 |
| K,V 계산 | 새 토큰만 | 매번 전체 재계산 |
| 시간 복잡도 | O(n) | O(n²) |
| 메모리 | 캐시 저장 필요 | 추가 메모리 없음 |
| 코드 복잡도 | 높음 | 낮음 |
| 학습 중 사용 | 불가 | 가능 |

---

## 실행 방법

```bash
# 의존성 설치
pip install tiktoken torch

# KV 캐시 없는 기본 버전
python gpt_ch04.py

# KV 캐시 적용 버전
python gpt_with_kv_cache.py
```

---

## KV 캐시 최적화 팁

현재 구현은 코드 가독성에 초점을 맞춘 교육용 버전입니다.
실전 배포 환경에서는 다음 최적화가 필요합니다.

### Tip 1: 메모리 사전 할당 (Pre-allocation)

`torch.cat`을 반복 호출하면 매번 메모리를 재할당하므로 비효율적입니다.
최대 시퀀스 길이를 기준으로 미리 텐서를 할당합니다:

```python
max_seq_len = 1024
cache_k = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
cache_v = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
# 이후 슬라이스에 값을 덮어씌우는 방식으로 사용
```

### Tip 2: 슬라이딩 윈도우 (Sliding Window)

긴 시퀀스에서 캐시가 무한정 커지는 것을 막기 위해 최근 N개 토큰만 유지합니다:

```python
window_size = 512
cache_k = cache_k[:, :, -window_size:, :]
cache_v = cache_v[:, :, -window_size:, :]
```

---

## 참고 자료

- [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) — 상세 설명 블로그 (영문)
- [LLMs-from-scratch GitHub](https://github.com/rasbt/LLMs-from-scratch)
