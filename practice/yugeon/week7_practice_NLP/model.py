# model.py
# ──────────────────────────────────────────────────────────────────────────────
# 양방향 LSTM 기반 텍스트 분류 모델 정의 모듈
# 주요 역할:
#  1) Embedding 레이어: 토큰 인덱스 → 임베딩 벡터
#  2) Bidirectional LSTM 인코더: 순방향 및 역방향 시퀀스를 은닉 상태로 압축
#  3) FC 레이어: 결합된 은닉 상태 → 클래스 로짓
# ──────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class, pad_idx):
        """
        - vocab_size: 어휘 사전 크기
        - embed_dim: 단어 임베딩 차원
        - hidden_dim: LSTM 은닉 상태 크기 (각 방향별)
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
        # 2) 양방향 LSTM 인코더
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        # 3) 양방향이므로 hidden_dim * 2 크기의 FC 레이어
        self.fc = nn.Linear(hidden_dim * 2, num_class)

    def forward(self, x):
        """
        순전파 진행
        - x: (batch_size, seq_len) LongTensor
        반환: (batch_size, num_class) logits
        """
        # 1) 임베딩 변환 → (B, L, E)
        emb = self.embedding(x)
        # 2) 양방향 LSTM 인코더
        # h_n: (num_layers * num_directions, B, H)
        _, (h_n, _) = self.lstm(emb)
        # h_n.shape = (2, B, H) 일 때,
        # h_n[0] = 순방향 마지막 은닉, h_n[1] = 역방향 마지막 은닉
        h_forward = h_n[0]    # (B, H)
        h_backward = h_n[1]   # (B, H)
        # 양방향 은닉 상태 결합 → (B, 2H)
        h_cat = torch.cat((h_forward, h_backward), dim=1)
        # 3) FC 레이어에 통과
        logits = self.fc(h_cat)
        return logits