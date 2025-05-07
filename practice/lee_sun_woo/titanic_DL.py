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

data_orignal.isnull().sum() #����ġ 0��
print(data_orignal.columns)

#������ ��ó��
data = data_orignal.drop(columns=['Name', 'Ticket'])
data = pd.get_dummies(data)

'''
pclass: ���ǵ��, name: �̸�, ticket:Ƽ�Ϲ�ȣ, sibsp:�Բ� ž���� �����ڸ�, ����� ��, parch:�Բ� ž���� �θ�
embarked: ž�� �ױ�
'''

#����, ���Ӻ��� ������
X = data.drop(columns=['Survived'])
Y = data['Survived']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#dataset Ŭ���� ����
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


#�� ����
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

#�н� ����
criterion = nn.BCELoss() #binary cross entropy
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#�н� ����
for epoch in range(100):
    model.train()
    total_loss = 0
    #xb�� �Էµ�����, yb�� ��µ�����
    for xb, yb in train_loader: # �����ͼ��� ��ġ ������ ������ �������� ����
        pred = model(xb).squeeze()
        loss = criterion(pred, yb) #�սǰ��
        optimizer.zero_grad()#�սǰ���� ������ ���� ��ġ�� ���� ����ġ ��� 0 �ʱ�ȭ
        loss.backward() #������ ���
        optimizer.step()# ���� gradient �������� �Ķ���� ������Ʈ
        total_loss += loss.item() #�սǴ��ϰ� .item�� �ټ��� ���ڷ� �ٲٴ� �Լ�

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

#�� ��
model.eval()
correct = 0
total = 0

with torch.no_grad(): # �׽�Ʈ�Ҷ��� �����ĵ� �ʿ����, �޸� �Ʋ��� �ϹǷ� gradient�� ������� �ʰ� ��
    for xb, yb in test_loader:
        pred = model(xb).squeeze()
        predicted = (pred > 0.5).float() # 0.5���� ũ�� 1 �� ����, ������ 0 (�����з��� ����)
        correct += (predicted == yb).sum().item() #�������� ������ yb�� ������ ��
        total += yb.size(0)

print(f"\n Accuracy: {correct / total:.4f}")
