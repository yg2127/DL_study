import warnings
import torchtext

# torchtext deprecation 경고 끄기
torchtext.disable_torchtext_deprecation_warning()

# torchdata DataPipes 경고 끄기
warnings.filterwarnings(
    "ignore",
    message=".*The 'datapipes', 'dataloader2' modules are deprecated.*",
    module="torchdata.datapipes"
)
warnings.filterwarnings("ignore", category=UserWarning, module="torchdata.datapipes")

# train.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator

from dataset import yield_tokens, load_agnews
from model import TextClassifier

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an LSTM news classifier on AG_NEWS"
    )
    # 하이퍼파라미터 옵션
    parser.add_argument("--batch-size", type=int, default=32,
                        help="훈련 배치 크기")
    parser.add_argument("--embed-dim", type=int, default=512,
                        help="임베딩 차원")
    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="LSTM 은닉 상태 크기")
    parser.add_argument("--max-len", type=int, default=30,
                        help="입력 시퀀스 최대 길이")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="학습률")
    parser.add_argument("--epochs", type=int, default=5,
                        help="학습 에포크 수")
    parser.add_argument("--output", type=str, default="agnews_model.pt",
                        help="저장할 체크포인트 파일명")
    return parser.parse_args()

def main():
    args = parse_args()

    # 0) 디바이스 설정: MPS 우선, 없으면 CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # 1) vocab 생성
    train_iter = AG_NEWS(split='train')
    vocab = build_vocab_from_iterator(
        yield_tokens(train_iter),
        specials=['<unk>', '<pad>']
    )
    vocab.set_default_index(vocab['<unk>'])

    # 2) DataLoader 준비
    train_loader = load_agnews(args.batch_size, vocab, args.max_len)

    # 3) 모델·손실·옵티마이저 초기화
    pad_idx = vocab['<pad>']
    model = TextClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_class=4,
        pad_idx=pad_idx
    )
    model = model.to(device)

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4) 학습 루프
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        model.train()
        for texts, labels in train_loader:
            texts  = texts.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(texts)
            loss   = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{args.epochs}  Loss: {total_loss:.4f}")

    # 5) 체크포인트 저장
    torch.save({
        'model_state': model.state_dict(),
        'vocab': vocab
    }, args.output)
    print(f"Model saved to {args.output}")

if __name__ == "__main__":
    main()