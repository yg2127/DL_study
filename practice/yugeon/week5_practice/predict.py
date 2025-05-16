# predict.py

import argparse                             # CLI 인자 파싱 라이브러리
from PIL import Image                       # 이미지 입출력 (Pillow)
import torch                                # PyTorch
import torchvision.transforms as transforms # 이미지 전처리 (Resize, Normalize)
from model import CNN250514                 # 내가 정의한 CNN 모델 클래스
from utils import load_model                # 저장된 모델 불러오는 함수
import time

# CIFAR-10 클래스 레이블 (인덱스 → 문자열 매핑)
CLASSES = [
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def parse_args():
    """
    커맨드라인 인자를 정의하고 파싱합니다.
    image_path   : 분류할 이미지 파일 경로 (필수)
    --model_path : 로드할 모델 파일 경로 (기본 'cnn.pth')
    """
    parser = argparse.ArgumentParser(
        description='Predict class of an image using trained CNN'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='분류할 이미지 파일 경로'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='cnn.pth',
        help='저장된 모델 파일 경로'
    )
    return parser.parse_args()

def predict(args):
    """
    1) 디바이스 설정
    2) 모델 로드 및 평가 모드
    3) 이미지 전처리
    4) 추론 수행
    5) 예측 결과 출력
    """
    # 예측 소요시간 측정
    start_time = time.time()

    # 1) 디바이스 설정: cuda, MPS가 가능하면 cuda or MPS, 아니면 CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # 2) 모델 로드 및 평가 모드 설정
    model = load_model(args.model_path).to(device)
    model.eval()  # 드롭아웃, 배치정규화 등을 평가 모드로 전환

    # 3) 이미지 전처리: Resize → Tensor → Normalize
    transform = transforms.Compose([
        transforms.Resize((32, 32)),               # CIFAR-10 크기에 맞춤
        transforms.ToTensor(),                     # [0,255] → [0,1]
        transforms.Normalize(                      # 평균 0.5, std 0.5로 정규화 → [-1,1]
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])
    image = Image.open(args.image_path).convert('RGB')  # RGB로 변환
    input_tensor = transform(image).unsqueeze(0).to(device)
    # unsqueeze(0): 배치 차원 추가 → (1, 3, 32, 32)

    # 4) 추론 수행 (기울기 계산 비활성화)
    with torch.no_grad():
        output = model(input_tensor)               # 모델에 입력
        pred_idx = output.argmax(dim=1).item()     # 가장 높은 점수 인덱스

    #시간 측정 후 출력
    elapsed_time = time.time() - start_time
    print(f"Prediction time: {elapsed_time:.4f} seconds")
    # 5) 결과 출력
    print(f'Predicted: {CLASSES[pred_idx]}')

if __name__ == '__main__':
    args = parse_args()  # 파싱된 인자 객체
    predict(args)        # 예측 함수 호출