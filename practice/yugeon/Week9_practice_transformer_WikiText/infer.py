# language_model/infer.py

import argparse
import math
import torch
import torch.nn.functional as F

from model import SmallTransformerLM, PositionalEncoding
from dataset import LMDataset


def load_model(checkpoint_path: str, vocab_size: int,
               d_model: int, nhead: int, num_layers: int,
               dim_feedforward: int, max_seq_length: int, dropout: float, device):
    """
    저장된 체크포인트를 불러와서 모델을 초기화하고, 가중치를 로드하여 반환합니다.
    """
    model = SmallTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_length=max_seq_length,
        dropout=dropout
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def generate_text(model: torch.nn.Module,
                  vocab,
                  tokenizer,
                  device,
                  prompt: str,
                  seq_len: int,
                  max_gen_len: int,
                  temperature: float = 1.0):
    """
    주어진 prompt를 바탕으로 최대 max_gen_len 토큰을 생성합니다.
    (Greedy sampling 버전)
    """
    # 1) prompt를 토크나이징 → 인덱스 리스트로 변환
    tokens = tokenizer(prompt)
    token_ids = [vocab[token] for token in tokens]

    # 2) 만약 prompt 길이가 seq_len보다 크면, 뒤의 seq_len 토큰만 사용
    if len(token_ids) > seq_len:
        token_ids = token_ids[-seq_len:]

    # 3) 현재 시퀀스를 torch.Tensor로 만들기 (batch_size=1)
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, current_len)

    generated = token_ids.copy()  # 생성된 토큰 리스트

    for _ in range(max_gen_len):
        current_len = input_ids.size(1)
        # 4) 모델이 기대하는 입력은 (batch_size, seq_len); 단, 현재 길이가 seq_len보다 짧을 수 있음
        #    → 필요 시 앞에서부터 패딩 (<pad>)을 추가하거나, PyTorch TransformerDecoder는
        #    “마스킹만 제대로 처리” 해 주면 길이가 유동적이어도 돌아갑니다.
        #    (우리 모델은 인과적 마스크만 사용하므로, 길이 그대로 넘겨도 동작)
        outputs = model(input_ids)  # (1, current_len, vocab_size)

        # 5) 마지막 토큰 위치의 로짓만 뽑아서 확률 분포 계산
        next_token_logits = outputs[0, -1, :]  # (vocab_size,)

        # 6) temperature를 곱해 주세요 (크면 분포가 샤프해지고, 작으면 평탄해짐)
        next_token_logits = next_token_logits / temperature

        # 7) 확률 분포 (softmax) → 가장 높은 값을 갖는 토큰을 선택 (Greedy)
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        next_token_id = torch.argmax(next_token_probs).item()

        # 8) 생성된 토큰을 리스트에 추가
        generated.append(next_token_id)

        # 9) input_ids를 갱신: (기존 토큰 + 방금 예측된 토큰)
        new_input = torch.tensor([[next_token_id]], dtype=torch.long, device=device)  # (1, 1)
        input_ids = torch.cat([input_ids, new_input], dim=1)  # → (1, current_len + 1)

        # 10) 만약 길이가 seq_len보다 커지면, 맨 앞 토큰을 빼고 뒤의 seq_len만 자르기 (sliding window)
        if input_ids.size(1) > seq_len:
            input_ids = input_ids[:, -seq_len:]

    # 11) 생성된 토큰 id 리스트를 다시 문자열로 변환
    generated_tokens = [vocab.get_itos()[idx] if hasattr(vocab, 'get_itos')
                        else vocab.get_itos()[idx] for idx in generated]
    # torchtext 0.15 미만 버전이라면 vocab.get_itos() 대신 vocab.get_itos() 메서드가 다를 수 있으므로
    # 사용자 환경에 맞게 get_itos() 호출 부분을 조정하세요.

    return " ".join(generated_tokens)


def main():
    parser = argparse.ArgumentParser(description="Transformer LM Inference Script")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="학습 후 저장된 모델 체크포인트 경로 (예: language_model/checkpoint.pt)")
    parser.add_argument("--prompt", type=str, default="",
                        help="텍스트 생성의 시작 문장(prompt)")
    parser.add_argument("--seq_len", type=int, default=32,
                        help="모델이 한 번에 볼 수 있는 최대 시퀀스 길이 (학습 때 설정값과 동일해야 함)")
    parser.add_argument("--max_gen_len", type=int, default=50,
                        help="prompt 이후 최대 생성할 토큰 개수")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="샘플링 온도 (기본값=1.0; 높일수록 다양성 ↑, 낮출수록 보수적 선택)")
    parser.add_argument("--d_model", type=int, default=128,
                        help="모델 내부 d_model (학습 때 설정값과 동일해야 함)")
    parser.add_argument("--nhead", type=int, default=4,
                        help="어텐션 헤드 수 (학습 때 설정값과 동일해야 함)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="디코더 레이어 개수 (학습 때 설정값과 동일해야 함)")
    parser.add_argument("--dim_feedforward", type=int, default=512,
                        help="FFN 내부 차원 (학습 때 설정값과 동일해야 함)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="드롭아웃 비율 (학습 때 설정값과 동일하게 맞출 것)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="실행할 디바이스 (예: 'cpu' 또는 'cuda')")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # 1) LMDataset을 이용해 vocab과 tokenizer 불러오기 (train split을 사용해 어휘를 구성)
    lm_dataset = LMDataset(split='train', seq_len=args.seq_len)
    vocab = lm_dataset.vocab
    tokenizer = lm_dataset.tokenizer
    vocab_size = len(vocab)

    # 2) 저장된 체크포인트로 모델 로드
    model = load_model(
        checkpoint_path=args.checkpoint,
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        max_seq_length=args.seq_len,
        dropout=args.dropout,
        device=device
    )

    # 3) 텍스트 생성 실행
    generated_text = generate_text(
        model=model,
        vocab=vocab,
        tokenizer=tokenizer,
        device=device,
        prompt=args.prompt,
        seq_len=args.seq_len,
        max_gen_len=args.max_gen_len,
        temperature=args.temperature
    )

    print("\n====== GENERATED TEXT ======")
    print(generated_text)
    print("====== END OF OUTPUT ======\n")


if __name__ == "__main__":
    main()