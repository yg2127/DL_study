from google.colab import drive
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

drive.mount('/content/drive')

data_orignal = pd.read_csv('/content/drive/MyDrive/dataset/titanic.csv')

data_orignal.isnull().sum() #결측치 0개
print(data_orignal.columns)

#데이터 전처리
data = data_orignal.drop(columns=['Name', 'Ticket'])
data = pd.get_dummies(data)

'''
pclass: 객실등급, name: 이름, ticket:티켓번호, sibsp:함께 탑승한 형제자매, 배우자 수, parch:함께 탑승한 부모
embarked: 탑승 항구
'''

#독립, 종속변수 나누기
X = data.drop(columns=['Survived'])
Y = data['Survived']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#dataset 클래스 정의
class TitanicDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


train_dataset = TitanicDataset(X_train, Y_train)
test_dataset = TitanicDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


#모델 정의
class TitanicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(X.shape[1], 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


model = TitanicModel()

#학습 설정
criterion = nn.BCELoss() #binary cross entropy
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#학습 루프
for epoch in range(100):
    model.train()
    total_loss = 0
    #xb는 입력데이터, yb는 출력데이터
    for xb, yb in train_loader: # 데이터셋을 배치 단위로 나눠서 가져오는 역할
        pred = model(xb).squeeze()
        loss = criterion(pred, yb) #손실계산
        optimizer.zero_grad()#손실계산이 끝나고 다음 배치를 위해 가중치 모두 0 초기화
        loss.backward() #역전파 계산
        optimizer.step()# 계산된 gradient 바탕으로 파라미터 업데이트
        total_loss += loss.item() #손실더하고 .item은 텐서를 숫자로 바꾸는 함수

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

#모델 평가
model.eval()
correct = 0
total = 0

with torch.no_grad(): # 테스트할때는 역전파도 필요없고, 메모리 아껴야 하므로 gradient를 계산하지 않게 함
    for xb, yb in test_loader:
        pred = model(xb).squeeze()
        predicted = (pred > 0.5).float() # 0.5보다 크면 1 즉 생존, 작으면 0 (이진분류기 때문)
        correct += (predicted == yb).sum().item() #예측값이 실제값 yb와 같은지 비교
        total += yb.size(0)

print(f"\n Accuracy: {correct / total:.4f}")
