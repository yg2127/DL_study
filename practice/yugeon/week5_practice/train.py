# 🔧 train.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN250514 # model.py에서 작성한 CNN
from dataset import get_dataloaders250514
from utils import accuracy, save_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train CNN on CIFAR-10')
    # 학습 옵션 추가
    parser.add_argument('--epochs', type=int, default=10, help='학습 반복 수')
    parser.add_argument('--batch_size', type=int, default=64, help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader 워커 수')
    parser.add_argument('--save_path', type=str, default='cnn.pth', help='모델 저장 경로')
    return parser.parse_args()


def train(args):
    """
    학습 흐름:
    1) 디바이스 설정 (MPS > CPU)
    2) 모델, DataLoader, 손실함수, 옵티마이저 초기화
    3) 에폭 루프: 순전파 → 손실 계산 → 역전파 → 파라미터 업데이트
    4) 에폭별 평균 손실 및 정확도 출력
    5) 최종 모델 저장
    """
    # 학습 소요시간 측정
    import time
    start_time = time.time()

    # 1) 디바이스 설정: MPS 우선, 없으면 CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # 2) 모델 초기화
    model = CNN250514().to(device)
    trainloader, testloader = get_dataloaders250514(batch_size=args.batch_size, num_workers=args.num_workers)
    # 손실함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 3) 학습 루프
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_acc = 0.0, 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += accuracy(outputs, labels)

        # 4) 에폭별 평균 계산 및 로그 출력
        avg_loss = epoch_loss / len(trainloader)
        avg_acc = epoch_acc / len(trainloader)
        print(f'Epoch [{epoch}/{args.epochs}] - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}')

    # 5) 모델 저장
    save_model(model, args.save_path)
    print(f'Model saved to {args.save_path}')

    # 6) 테스트데이터 점수
    model.eval()
    test_acc = 0.0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_acc += accuracy(outputs, labels)

    # 시간 측정 후 출력
    elapsed_time = time.time() - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")
    print(f"Test Accuracy: {test_acc / len(testloader):.4f}")


if __name__ == '__main__':
    args = parse_args()
    train(args)


