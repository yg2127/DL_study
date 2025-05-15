# 🔧 dataset.py
# CIFAR-10 데이터셋을 로딩하고 DataLoader를 반환하는 함수들
import torchvision.transforms as transforms  # 이미지 전처리 도구
import torchvision.datasets as datasets      # 내장 데이터셋 모듈
from torch.utils.data import DataLoader      # 배치 처리용 DataLoader

def get_dataloaders250514(batch_size=64, num_workers=2):
    """
    CIFAR-10 학습 및 테스트 DataLoader를 생성합니다.

    Args:
        batch_size (int): 배치 크기
        num_workers (int): 데이터 로딩에 사용할 subprocess 수

    Returns:
        trainloader (DataLoader): 학습용 DataLoader
        testloader (DataLoader): 테스트용 DataLoader
    """
    # 1) Transform: Tensor 변환 및 정규화
    transform = transforms.Compose([
        transforms.ToTensor(),                                  # [0,255] 이미지 -> [0,1] Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 평균0.5, std0.5로 정규화 -> [-1,1]
    ])

    # 2) 학습용 데이터셋 로드
    trainset = datasets.CIFAR10(
        root='./data',     # 데이터 저장 디렉토리
        train=True,        # 학습 데이터 사용
        download=True,     # 없으면 다운로드
        transform=transform
    )
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,      # 에폭마다 무작위 섞기
        num_workers=num_workers
    )

    # 3) 테스트용 데이터셋 로드
    testset = datasets.CIFAR10(
        root='./data',
        train=False,       # 테스트 데이터 사용
        download=True,
        transform=transform
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,     # 순차 로딩
        num_workers=num_workers
    )

    return trainloader, testloader
