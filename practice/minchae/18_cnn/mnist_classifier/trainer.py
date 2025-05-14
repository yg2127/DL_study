from copy import deepcopy
import numpy as np
import torch

# 모델 학습을 위한 클래스
class Trainer():

    def __init__(self, model, optimizer, crit):
        self.model = model          # 모델
        self.optimizer = optimizer  # 옵티마이저
        self.crit = crit            # 손실 함수

        super().__init__()

    # 데이터를 배치 단위로 나누는 함수
    def _batchify(self, x, y, batch_size, random_split=True):
        # random_split이 True일 경우, 데이터를 무작위로 섞음
        if random_split:
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices)
            y = torch.index_select(y, dim=0, index=indices)

        # x와 y를 배치 크기로 나누어 반환
        x = x.split(batch_size, dim=0)
        y = y.split(batch_size, dim=0)
        return x, y

    # 모델을 학습하는 함수
    def _train(self, x, y, config):
        self.model.train()                              # 모델을 학습 모드로 설정

        x, y = self._batchify(x, y, config.batch_size)  # 배치로 나누기
        total_loss = 0                                  # 전체 손실 값을 초기화

        # 배치 단위로 학습
        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)                   # 모델의 예측값 계산
            loss_i = self.crit(y_hat_i, y_i.squeeze())  # 손실 함수 계산

            self.optimizer.zero_grad()                  # 기울기 초기화
            loss_i.backward()                           # 역전파
            self.optimizer.step()                       # 가중치 업데이트

            # verbose가 2 이상이면, 각 배치마다 손실 출력
            if config.verbose >= 2:
                print("Train Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))
            total_loss += float(loss_i)                 # 손실 누적
        return total_loss / len(x)                      # 전체 배치에 대한 평균 손실값 반환

    # 모델의 성능을 검수하는 함수
    def _validate(self, x, y, config):
        self.model.eval()   # 모델을 평가 모드로 설정

        # 메모리 절약을 위해 기울기 계산 X
        with torch.no_grad():
            x, y = self._batchify(x, y, config.batch_size, random_split=False)  # 배치로 나누기
            total_loss = 0                                                      # 전체 손실 값을 초기화

            # 배치 단위로 검증
            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)                                       # 모델의 예측값 계산
                loss_i = self.crit(y_hat_i, y_i.squeeze())                      # 손실값 계산

                # verbose가 2 이상이면, 각 배치마다 손실 출력
                if config.verbose >= 2:
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))
                total_loss += float(loss_i)                                     # 손실 누적
            return total_loss / len(x)                                          # 전체 배치에 대한 평균 손실값 반환

    # 모델 학습을 위한 전체 트레이닝 루프
    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf    # 가장 낮은 검증 손실 값을 초기화
        best_model = None       # 가장 성능이 좋은 모델의 상태를 저장할 변수

        # 설정된 epoch 수만큼 반복
        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)      # 학습 과정
            valid_loss = self._validate(valid_data[0], valid_data[1], config)   # 검증 과정

            # 검증 손실이 가장 낮을 때 모델을 저장
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())
            # 학습 및 검증 손실 출력
            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (epoch_index + 1, config.n_epochs, train_loss, valid_loss, lowest_loss))

        self.model.load_state_dict(best_model)  # 학습이 끝난 후, 가장 성능이 좋았던 모델로 로드