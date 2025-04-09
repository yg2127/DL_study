import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data_orignal.isnull().sum() #결측치 0개
print(data_orignal.columns)

#데이터 전처리
data_2 = data_orignal.drop(columns=['Name', 'Ticket'])
data_2 = pd.get_dummies(data)


# 특성과 정답 분리
X = data_2.drop(columns=['Survived']).values
y = data_2['Survived'].values

# 데이터 정규화
scaler = StandardScaler()
X = scaler.fit_transform(X)

# train/test 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tensor로 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# 로지스틱 회귀 모델 정의
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegressionModel(X_train.shape[1])

# 손실 함수와 최적화기
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 학습
for epoch in range(100):
    model.train()
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 평가
model.eval()
with torch.no_grad():
    outputs = model(X_test).squeeze()
    preds = (outputs > 0.5).float()
    acc = (preds == y_test).sum() / len(y_test)
    print(f"Test Accuracy: {acc:.4f}")