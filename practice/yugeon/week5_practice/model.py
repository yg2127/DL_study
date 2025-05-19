# model.py

import torch.nn as nn
import torch.nn.functional as F

class CNN250514(nn.Module):
    def __init__(self):
        super().__init__()
        # 1st CONV block
        # 입력 채널 3(RGB), 출력 채널 32, 커널 크기 3×3, padding=1 → 출력 크기: 32×32×32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # 2nd CONV block
        # 입력 채널 32, 출력 채널 64, 커널 크기 3×3, padding=1 → 출력 크기: 32×32×64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 최대 풀링: 2×2 윈도우, 스트라이드 2 → 공간 크기 절반으로 축소
        self.pool = nn.MaxPool2d(2, 2)
        # 첫 번째 완전연결 계층: 64채널 × 8 × 8 → 256 유닛
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        # 출력 계층: 256 → 10 (CIFAR-10 클래스 수)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # --- Conv Block 1 ---
        x = self.conv1(x)        # 합성곱 연산 → (B, 32, 32, 32)
        x = F.relu(x)            # 비선형성 부여
        x = self.pool(x)         # 풀링 → (B, 32, 16, 16)

        # --- Conv Block 2 ---
        x = self.conv2(x)        # 합성곱 → (B, 64, 16, 16)
        x = F.relu(x)            # 활성화
        x = self.pool(x)         # 풀링 → (B, 64, 8, 8)

        # --- Flatten ---
        x = x.view(x.size(0), -1)  # (B, 64*8*8=4096) 형태로 변환

        # --- Fully Connected Layers ---
        x = self.fc1(x)          # 선형 변환 → (B, 256)
        x = F.relu(x)            # 활성화
        x = self.fc2(x)          # 최종 선형 변환 → (B, 10)

        return x
