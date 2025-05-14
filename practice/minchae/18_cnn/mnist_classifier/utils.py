import torch

from mnist_classifier.models.cnn import ConvolutionalClassifier

# MNIST 데이터셋을 로드하고 전처리하는 함수
def load_mnist(is_train=True):
    from torchvision import datasets, transforms

    # MNIST 데이터셋 로드 (다운로드 X, 이미지는 텐서로 변환)
    dataset = datasets.MNIST(
        '../data', train=is_train, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    # 이미지를 0 ~ 1 범위로 정규화
    x = dataset.data.float() / 255.
    y = dataset.targets

    return x, y

# 데이터를 훈련 데이터와 검증 데이터로 나누는 함수
def split_data(x, y, train_ratio=.8):
    # 훈련 데이터의 개수를 계산
    train_cnt = int(x.size(0) * train_ratio)
    valid_cnt = x.size(0) - train_cnt

    indices = torch.randperm(x.size(0)).to(x.device)    # 무작위로 데이터 섞기 위한 인덱스 생성
    # 데이터를 무작위로 섞은 후 훈련 데이터와 검증 데이터로 나누기
    x = torch.index_select(x, dim=0, index=indices).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(y, dim=0, index=indices).split([train_cnt, valid_cnt], dim=0)
    return x, y

# 주어진 입력 크기와 출력 크기에 맞는 은닉층 크기를 계산하는 함수
def get_hidden_sizes(input_size, output_size, n_layers):
    step_size = int((input_size - output_size) / n_layers)  # 각 은닉층의 크기 차이를 계산

    hidden_sizes = []
    current_size = input_size
    for i in range(n_layers - 1):                           # n_layers-1 만큼 은닉층 크기 계산
        hidden_sizes += [current_size - step_size]          # 현재 크기에서 step_size만큼 차감하여 은닉층 크기 계산
        current_size = hidden_sizes[-1]                     # 마지막으로 계산된 크기를 다음 층의 입력 크기로 설정

    return hidden_sizes

# 모델을 생성하는 함수, 책의 코드는 MLP와 CNN 둘의 코드가 있는 것을 전제로 코드를 작성하였지만, 저는 CNN 코드만 구현했습니다.
def get_model(input_size, output_size, config, device):
    if config.model == "cnn":
        model = ConvolutionalClassifier(output_size)
    else:
        raise NotImplementedError
    return model