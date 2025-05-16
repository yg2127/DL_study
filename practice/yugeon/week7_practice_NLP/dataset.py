# dataset.py
# ──────────────────────────────────────────────────────────────────────────────
# AG_NEWS 데이터셋을 불러와 DataLoader로 변환하는 모듈
# 주요 역할:
#  1) 토크나이저 정의
#  2) vocab 생성을 위한 yield_tokens
#  3) load_agnews 함수로 DataLoader 반환
# ──────────────────────────────────────────────────────────────────────────────

import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer

# 1) 기본 영어 토크나이저 설정
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    """
    vocab 빌드를 위해 데이터셋 전체를 순회하며
    토큰 리스트를 생성해주는 제너레이터
    """
    for label, text in data_iter:
        yield tokenizer(text)

def load_agnews(batch_size, vocab, max_len):
    """
    AG_NEWS 학습용 DataLoader 생성 함수
    - batch_size: 배치 크기
    - vocab: build_vocab_from_iterator로 만든 Vocab 객체
    - max_len: 문장 최대 길이 (이상은 자르고, 미달은 <pad>로 채움)
    반환값: torch.utils.data.DataLoader
    """
    from torch.utils.data import DataLoader

    pad_idx = vocab['<pad>']  # 패딩 토큰 인덱스

    def collate_batch(batch):
        """
        DataLoader 내부에서 batch 단위로 호출되는 함수
        - batch: [(label, text), ...] 리스트
        반환: (texts_tensor, labels_tensor)
        """
        texts, labels = [], []
        for label, text in batch:
            tokens = tokenizer(text)
            # 토큰 → 인덱스 변환, max_len 자르기
            indices = [vocab[token] for token in tokens][:max_len]
            # 부족한 길이는 pad_idx로 채우기
            padded = indices + [pad_idx] * (max_len - len(indices))
            texts.append(torch.tensor(padded))
            labels.append(label - 1)  # 레이블 0~3 으로 맞춤
        # 배치 차원으로 쌓아서 반환
        return torch.stack(texts), torch.tensor(labels)

    # train split 로더 생성
    train_iter = AG_NEWS(split='train')
    return DataLoader(train_iter,
                      batch_size=batch_size,
                      shuffle=True,
                      collate_fn=collate_batch)