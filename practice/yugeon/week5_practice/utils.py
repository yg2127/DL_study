
# 🔧 utils.py
import torch

from model import CNN250514


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """출력과 타겟을 비교하여 정확도를 계산"""
    preds = output.argmax(dim=1)
    return (preds == target).sum().item() / target.size(0)


def save_model(model: torch.nn.Module, path: str = 'cnn.pth') -> None:
    """모델의 state_dict를 지정 경로에 저장"""
    torch.save(model.state_dict(), path)


def load_model(path: str = 'cnn.pth') -> torch.nn.Module:
    """저장된 모델을 로드하여 반환"""
    model = CNN250514()
    model.load_state_dict(torch.load(path))
    return model
