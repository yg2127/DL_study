# model.py
# ──────────────────────────────────────────────────────────────────────────────
# LSTM 기반 텍스트 분류 모델 정의 모듈
# 주요 역할:
#  1) Embedding 레이어: 토큰 인덱스 → 임베딩 벡터
#  2) LSTM 인코더: 시퀀스를 하나의 은닉 상태로 압축
#  3) FC 레이어: 은닉 상태 → 클래스 로짓
# ──────────────────────────────────────────────────────────────────────────────

import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class, pad_idx):
        """
        - vocab_size: 어휘 사전 크기
        - embed_dim: 단어 임베딩 차원
        - hidden_dim: LSTM 은닉 상태 크기
        - num_class: 예측할 클래스 개수 (4)
        - pad_idx: 패딩 토큰의 인덱스
        """
        super().__init__()
        # 1) 임베딩 레이어: pad_idx 위치 매핑 시 gradient 흐르지 않음
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )
        # 2) 단방향 LSTM 인코더
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        # 3) 최종 분류용 FC 레이어
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        """
        순전파 진행
        - x: (batch_size, seq_len) LongTensor
        반환: (batch_size, num_class) logits
        """
        # 1) 임베딩 변환 → (B, L, E)
        emb = self.embedding(x)
        # 2) LSTM 인코더 → (h_n: (1, B, H), c_n: (1, B, H))
        _, (h_n, _) = self.lstm(emb)
        # 3) 마지막 타임스텝 은닉 상태를 FC에 입력
        h_n = h_n.squeeze(0)            # (B, H)
        logits = self.fc(h_n)           # (B, num_class)
        return logits