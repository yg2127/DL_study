# language_model/model.py

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    사인/코사인 기반 포지셔널 인코딩을 생성하여 토큰 임베딩에 더해 주는 모듈.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)               # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)     # 짝수 인덱스: sin
        pe[:, 1::2] = torch.cos(position * div_term)     # 홀수 인덱스: cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]  # (batch_size, seq_len, d_model)
        return x

class SmallTransformerLM(nn.Module):
    """
    작은 규모 Transformer 기반 언어 모델 (인과적 디코더 구조).
    """
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 512,
                 max_seq_length: int = 32, dropout: float = 0.1):
        super(SmallTransformerLM, self).__init__()
        self.d_model = d_model
        # 1) Token Embedding: (vocab_size, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 2) Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_length)
        # 3) Transformer Decoder Layer (인과적 Self-Attention 포함)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        # 여러 개 층 쌓기
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        # 4) 출력층: (d_model -> vocab_size)
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        인과적 마스크(causal mask) 생성: 미래 토큰에 접근하지 못하도록 -inf로 마스킹.
        i < j 인 원소들이 -inf이고, i >= j 인 원소는 0.
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()  # 상삼각 True
        mask = mask.float().masked_fill(mask, float('-inf'))
        return mask  # (sz, sz)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: (batch_size, seq_len) 정수 인덱스 토큰 시퀀스
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = src.size()
        # 1) 임베딩 및 스케일링: (batch_size, seq_len, d_model)
        embedded = self.embedding(src) * math.sqrt(self.d_model)
        embedded = self.pos_encoder(embedded)
        embedded = self.dropout(embedded)
        # 2) Transformer Decoder 입력 형식: (seq_len, batch_size, d_model)
        embedded = embedded.transpose(0, 1)  # → (seq_len, batch_size, d_model)

        # 3) 인과적 마스크 생성: (seq_len, seq_len)
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(src.device)

        # 4) 디코더 통과: (seq_len, batch_size, d_model)
        output = self.transformer_decoder(tgt=embedded,
                                          memory=None,
                                          tgt_mask=tgt_mask)

        # 5) (seq_len, batch_size, d_model) → (batch_size, seq_len, d_model)
        output = output.transpose(0, 1)

        # 6) Linear 출력: (batch_size, seq_len, vocab_size)
        logits = self.fc_out(output)
        return logits