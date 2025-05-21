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

# infer.py
# ──────────────────────────────────────────────────────────────────────────────
# CLI 환경에서 학습된 모델을 불러와
# 사용자가 입력한 문장을 실시간으로 분류해주는 스크립트
# 주요 역할:
#  1) 체크포인트 로드
#  2) 모델 및 Vocab 초기화
#  3) 사용자 입력 → 전처리 → 예측 → 출력
# ──────────────────────────────────────────────────────────────────────────────

import torch
from model import TextClassifier
from utils import preprocess_text

# 1) 클래스 라벨 매핑
labels = ["World", "Sports", "Business", "Sci/Tech"]

# 2) 저장된 체크포인트 로드
ckpt = torch.load("agnews_model.pt")
vocab = ckpt['vocab']

# 3) 모델 초기화 및 가중치 불러오기
pad_idx = vocab['<pad>']
model = TextClassifier(
    vocab_size=len(vocab),
    embed_dim=64,
    hidden_dim=128,
    num_class=len(labels),
    pad_idx=pad_idx
)
model.load_state_dict(ckpt['model_state'])
model.eval()  # 평가 모드로 전환

# 4) 무한 루프를 돌며 사용자 입력 처리
print("=== AG News Classifier (CLI) ===")
while True:
    text = input("\nEnter news > ").strip()
    if not text:
        continue   # 빈 문자열 입력 시 다시 받기
    # 4-1) 텍스트 전처리: 토큰화→인덱스→패딩→Tensor 변환
    x = preprocess_text(text, vocab, max_len=30)
    # 4-2) 예측
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()
    # 4-3) 결과 출력
    print(f"Predicted category: {labels[pred]}")