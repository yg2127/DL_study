# language_model/utils.py

import torch

def create_padding_mask(token_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    토큰 시퀀스에서 PAD 위치를 마스킹하는 함수.
    Args:
        token_ids: (batch_size, seq_len) 정수 인덱스 시퀀스
        pad_token_id: PAD 토큰의 인덱스 (예: vocab['<pad>'])
    Returns:
        mask: (batch_size, seq_len) bool tensor. PAD 위치는 True.
    """
    return (token_ids == pad_token_id)

def calculate_perplexity(loss: float) -> float:
    """
    CrossEntropyLoss 값을 받아서 Perplexity를 계산하는 함수.
    Perplexity = exp(loss)
    """
    return torch.exp(torch.tensor(loss)).item()