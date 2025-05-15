# 🧠 CIFAR-10 CNN Classifier (CLI 기반)

CIFAR-10 데이터셋을 학습하는 PyTorch 기반 CNN 분류기입니다.  
학습, 예측, 초기화 등 모든 기능은 **커맨드라인 인터페이스(CLI)** 로 실행됩니다.

---

## 📁 프로젝트 구조
.
├── model.py           # CNN250514 모델 정의
├── dataset.py         # CIFAR-10 DataLoader
├── utils.py           # 모델 저장/불러오기, 정확도 계산
├── train.py           # 학습 CLI
├── predict.py         # 이미지 예측 CLI
└── data/              # (자동 생성) CIFAR-10 다운로드 폴더

---

## 🏋️ 모델 학습: `train.py`

기본 실행:
```bash
python train.py

옵션 지정:

python train.py --epochs 20 --batch_size 128 --lr 0.0005 --save_path mymodel.pth

> ✅ 코드 내에서 PyTorch가 아래 순서로 디바이스를 자동 감지합니다:
>     1. `CUDA` (NVIDIA GPU)
>     2. `MPS` (Mac 전용 GPU)
>     3. `CPU` (기본 디바이스)

💡 모델은 cnn.pth 또는 지정한 경로로 저장됨

⸻

🔍 이미지 예측: predict.py

단일 이미지 예측:

python predict.py /path/to/image.png

모델 경로 지정:

python predict.py /path/to/image.png --model_path mymodel.pth

📌 결과는 Predicted: cat 등으로 터미널에 출력됩니다.
🧽 이미지 크기가 달라도 자동으로 32x32 리사이징 처리됩니다.

⸻

📂 이미지 폴더 일괄 예측

for img in ~/Desktop/cifar10/*.{jpg,png}; do
  python predict.py "$img"
done


⸻

♻️ 모델 초기화 저장: reset_model.py

초기화된 가중치 저장 (훈련 전 상태):

python reset_model.py --output cnn_initial.pth

👉 이후 train.py에서 해당 모델을 불러오면 깨끗한 상태에서 재학습 가능

⸻

⚠️ 주의사항
	•	data/ 폴더는 .gitignore에 추가되어야 합니다.
	•	GitHub에 푸시 시, CIFAR-10 원본 압축 파일은 LFS 또는 제외 필요
	•	cnn.pth 등 학습된 모델은 .gitignore 또는 .gitattributes로 관리 권장

⸻

👨‍💻 개발환경
	•	macOS (Apple Silicon M1/M2/M3) 최적화
	•	Python 3.8 이상
	•	PyTorch (MPS 지원 버전)

conda install pytorch torchvision torchaudio -c pytorch


⸻

✨ 예측 클래스

['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


⸻

🙌 Author
	•	유건 (Yugeon)
	•	세종대학교 인공지능학과
	•	GitHub: yg2127

---

필요하면 실제 프로젝트에 맞게 사용자 이름, 파일 이름(CNN 클래스명 등), 사용 환경 등을 수정해 드릴 수 있어요.  
`README.md` 파일로 저장해 넣으면 바로 사용 가능합니다!