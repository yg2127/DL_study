# model.py
# ──────────────────────────────────────────────────────────────────────────────
# Transformer 기반 텍스트 분류 모델 정의 모듈
# 주요 역할:
#   1) Token Embedding + Positional Embedding
#   2) Transformer Encoder: Multi-Head Self-Attention 기반 인코더 스택
#   3) Pooling (평균 풀링) → FC 레이어 → 클래스 로짓
# ──────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn

class TransformerTextClassifier(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embed_dim: int,
            num_heads: int,
            hidden_dim_ff: int,
            num_layers: int,
            num_class: int,
            pad_idx: int,
            max_len: int = 512,
            dropout: float = 0.1
    ):
        """
        - vocab_size: 어휘 사전 크기
        - embed_dim: 토큰 임베딩/포지션 임베딩 차원 (d_model)
        - num_heads: Multi-Head Attention 헤드 수
        - hidden_dim_ff: Feed-Forward Network 내부 히든 차원 (보통 embed_dim * 4)
        - num_layers: Transformer Encoder 레이어 개수
        - num_class: 예측할 클래스 개수 (AG_News: 4)
        - pad_idx: 패딩 토큰 인덱스
        - max_len: 최대 시퀀스 길이 (포지션 인코딩 크기)
        - dropout: 드롭아웃 비율
        """
        super().__init__()

        # 1) Token Embedding: pad_idx 위치는 gradient 흐르지 않음
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        # 2) Positional Embedding: 학습 가능한 위치 임베딩
        #    [0, 1, 2, ..., max_len-1] 위치 인덱스를 embed_dim 크기로 매핑
        self.pos_embedding = nn.Embedding(
            num_embeddings=max_len,
            embedding_dim=embed_dim
        )

        # 3) Transformer Encoder 레이어 정의
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim_ff,
            dropout=dropout,
            activation='gelu',        # GELU 활성화: 논문 권장
            batch_first=True          # (batch, seq_len, embed_dim) 순서로 입력
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 4) Dropout (인코더 출력 후 풀링 전에 적용할 수도 있음)
        self.dropout = nn.Dropout(dropout)

        # 5) 최종 분류기: 평균 풀링 결과 → num_class 로짓
        self.fc = nn.Linear(embed_dim, num_class)


    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        순전파 진행
        - x: (batch_size, seq_len) LongTensor (토큰 인덱스)
        반환: (batch_size, num_class) 로짓
        """
        # 1) Token Embedding → (B, L, E)
        emb_tokens = self.token_embedding(x)
        #    emb_tokens.shape = (batch_size, seq_len, embed_dim)

        # 2) Positional Embedding
        batch_size, seq_len = x.size()
        # 위치 인덱스 0부터 seq_len-1까지 생성 → (batch_size, seq_len)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        # 위치 임베딩 → (batch_size, seq_len, embed_dim)
        emb_positions = self.pos_embedding(positions)

        # 3) 토큰 임베딩 + 위치 임베딩 합산
        #    Transformer Encoder에 들어갈 최종 입력
        #    shape = (batch_size, seq_len, embed_dim)
        encoder_input = emb_tokens + emb_positions

        # 4) Transformer Encoder
        #    출력 shape: (batch_size, seq_len, embed_dim)
        encoder_output = self.transformer_encoder(encoder_input)
        #    ※ 내부적으로
        #      (1) Multi-Head Self-Attention + Residual + LayerNorm
        #      (2) Feed-Forward + Residual + LayerNorm
        #    을 num_layers 번 반복함

        # 5) Pooling: 시퀀스 차원 평균 풀링
        #    또는 [CLS] 토큰을 사용했다면, encoder_output[:, 0, :] 를 사용
        pooled = encoder_output.mean(dim=1)    # (batch_size, embed_dim)

        # 6) Dropout (optional)
        pooled = self.dropout(pooled)

        # 7) FC 층 → 로짓 계산
        logits = self.fc(pooled)               # (batch_size, num_class)
        return logits