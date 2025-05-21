# utils.py
# ──────────────────────────────────────────────────────────────────────────────
# CLI 환경에서 사용자 입력 문장을 모델 입력으로 변환하는 유틸리티 모듈
# 주요 역할:
#  1) 토크나이저 재사용
#  2) preprocess_text 함수를 통해
#     텍스트 → 토큰 → 인덱스 → 패딩 → Tensor 변환
# ──────────────────────────────────────────────────────────────────────────────

import torch
from torchtext.data.utils import get_tokenizer

# 1) 영어 토크나이저
tokenizer = get_tokenizer('basic_english')

def preprocess_text(text, vocab, max_len=30):
    """
    사용자 입력 문자열을 모델 입력 형태의 Tensor로 변환
    - text: 입력 문자열
    - vocab: 학습 시 생성한 Vocab 객체
    - max_len: 시퀀스 최대 길이
    반환: LongTensor of shape (1, max_len)
    """
    # 토큰화
    tokens = tokenizer(text)
    # pad 인덱스
    pad_idx = vocab['<pad>']
    # 각 토큰을 인덱스로 변환 (default unk handled by vocab)
    indices = [vocab[tok] for tok in tokens]
    # max_len 넘으면 자르고, 모자라면 pad로 채우기
    padded = indices[:max_len] + [pad_idx] * max(0, max_len - len(indices))
    # (max_len,) → (1, max_len)으로 차원 확장
    return torch.tensor(padded).unsqueeze(0)