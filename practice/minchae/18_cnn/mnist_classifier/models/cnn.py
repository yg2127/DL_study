import torch.nn as nn

# 하나의 합성곱 블록을 정의하는 클래스
class ConvolutionBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels      # 입력 채널 수
        self.out_channels = out_channels    # 출력 채널 수

        super().__init__()

        # 두 개의 Conv2D + ReLU + BatchNorm 블록으로 구성된 합성곱 블록
        # 첫 번째 Conv2D는 입력과 출력 채널을 유지하며 패딩으로 크기 보존
        # 두 번째 Conv2D는 stride=2로 다운샘플링

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
    
    # 순전파, 정의된 합성곱 블록을 순차적으로 적용
    def forward(self, x):
        y = self.layers(x)
        return y
    
# 전체 CNN 분류기 모델 정의
class ConvolutionalClassifier(nn.Module):

    def __init__(self, output_size):
        self.output_size = output_size  # 최종 출력 클래스의 개수

        super().__init__()

        # 5개의 ConvolutionBlock으로 구성된 피처 추출기
        # 입력은 1채널 이미지 (흑백)
        self.blocks = nn.Sequential(
            ConvolutionBlock(1, 32),
            ConvolutionBlock(32, 64),
            ConvolutionBlock(64, 128),
            ConvolutionBlock(128, 256),
            ConvolutionBlock(256, 512),
        )

        # 마지막 Conv 블록 이후의 출력을 받아 분류를 수행하는 완전 연결층 (fully connected layers)
        self.layers = nn.Sequential(
            nn.Linear(512, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, output_size),
            nn.LogSoftmax(dim=-1),  # 분류 문제에 적합한 LogSoftmax 사용
        )

    def forward(self, x):
        assert x.dim() > 2              # 입력은 최소 3차원이어야 함 (배치 X 높이 X 너비 또는 채널 포함)

        # 입력이 (B, H, W) 형태일 경우, (B, 1, H, W)로 reshape하여 2D conv에 맞춤
        if x.dim() == 3:
            x = x.view(-1, 1, x.size(-2), x.size(-1))
        z = self.blocks(x)              # 특징 추출기 통과
        y = self.layers(z.squeeze())    # 마지막 출력은 (B, 512, 1, 1)일 것이므로 squeeze하여 (B, 512)로 변환 후 분류기에 통과시켜 최종 결과 출력
        return y