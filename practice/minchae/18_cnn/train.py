import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from mnist_classifier.trainer import Trainer
from mnist_classifier.utils import load_mnist
from mnist_classifier.utils import split_data
from mnist_classifier.utils import get_model

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--model_fn", required=True)             # 모델 저장 경로 및 파일 이름
    p.add_argument("--gpu_id", type=int, default=0 if torch.cuda.is_available() else -1)    # 사용할 GPU ID (GPU가 없으면 -1로 설정)

    p.add_argument("--train_ratio", type=float, default=.8) # 훈련 데이터와 검증 데이터의 비율 설정 (기본값 80% 훈련, 20% 검증)

    p.add_argument("--batch_size", type=int, default=256)   # 배치 크기 설정
    p.add_argument("--n_epochs", type=int, default=10)      # epoch 수 설정

    p.add_argument("--model", default="cnn")                # 사용할 모델 유형

    p.add_argument("--n_layers", type=int, default=5)       # 은닉층의 수
    p.add_argument("--use_dropout", action="store_true")    # 드롭아웃 사용 여부
    p.add_argument("--dropout_p", type=float, default=.3)   # 드롭아웃 확률

    p.add_argument("--verbose", type=int, default=1)        # 학습 중 출력되는 정보의 상세 레벨 (0: 최소, 1: 기본, 2: 상세)

    config = p.parse_args()

    return config

def main(config):
    device = torch.device("cuda") # GPU 사용

    # MNIST 데이터를 로드하고 훈련 및 검증 데이터로 분리
    x, y = load_mnist(is_train=True)
    x, y = split_data(x.to(device), y.to(device), train_ratio=config.train_ratio)

    # 데이터 크기 출력 (훈련 및 검증 데이터의 크기 확인)
    print("Train:", x[0].shape, y[0].shape)
    print("Valid:", x[1].shape, y[1].shape)

    # 모델의 입력 크기와 출력 크기 설정
    input_size = int(x[0].shape[-1])    # 입력 크기는 데이터의 마지막 차원
    output_size = int(max(y[0])) + 1    # 출력 크기는 클래스의 개수 (라벨의 최댓값 + 1)

    # 모델 생성
    model = get_model(
        input_size,
        output_size,
        config,
        device,
    ).to(device) # 모델을 GPU로 이동
    # 옵티마이저는 Adam 사용, 손실 함수는 NLLLoss 사용
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    # 학습 전 모델, 옵티마이저, 손실 함수 정보 출력
    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(model, optimizer, crit) # Trainer 객체 생성 (모델, 옵티마이저, 손실 함수 전달)
    trainer.train(train_data=(x[0], y[0]), valid_data=(x[1], y[1]), config=config) # 모델 학습

    # 학습된 모델과 옵티마이저, 설정을 파일로 저장
    torch.save({
        "model": trainer.model.state_dict(),
        "opt": optimizer.state_dict(),
        "config": config,
    }, config.model_fn)

if __name__ == "__main__":
    config = define_argparser()
    main(config)