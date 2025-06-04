# language_model/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import LMDataset
from model import SmallTransformerLM
from utils import calculate_perplexity

# 하이퍼파라미터 설정
SEQ_LEN = 32
BATCH_SIZE = 64
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
LR = 1e-4
NUM_EPOCHS = 5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) 데이터셋 및 DataLoader 생성
#    train, valid split에서 각각 Dataset 생성
train_dataset = LMDataset(split='train', seq_len=SEQ_LEN)
valid_dataset = LMDataset(split='valid',
                          vocab=train_dataset.vocab,
                          tokenizer=train_dataset.tokenizer,
                          seq_len=SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

VOCAB_SIZE = len(train_dataset.vocab)
PAD_ID = train_dataset.vocab['<pad>']

# 2) 모델 초기화
model = SmallTransformerLM(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    max_seq_length=SEQ_LEN,
    dropout=DROPOUT
).to(DEVICE)

# 3) 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = optim.AdamW(model.parameters(), lr=LR)

# 4) 학습 루프
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    for batch_inputs, batch_targets in train_loader:
        batch_inputs  = batch_inputs.to(DEVICE)    # (batch_size, seq_len)
        batch_targets = batch_targets.to(DEVICE)   # (batch_size, seq_len)

        optimizer.zero_grad()
        outputs = model(batch_inputs)              # (batch_size, seq_len, vocab_size)
        loss = criterion(outputs.view(-1, VOCAB_SIZE),
                         batch_targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_ppl = calculate_perplexity(avg_train_loss)

    # 검증 단계
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_targets in valid_loader:
            val_inputs  = val_inputs.to(DEVICE)
            val_targets = val_targets.to(DEVICE)
            val_outputs = model(val_inputs)
            loss_v = criterion(val_outputs.view(-1, VOCAB_SIZE),
                               val_targets.view(-1))
            val_loss += loss_v.item()

    avg_val_loss = val_loss / len(valid_loader)
    val_ppl = calculate_perplexity(avg_val_loss)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} — "
          f"Train Loss: {avg_train_loss:.4f}, Train PPL: {train_ppl:.2f} | "
          f"Valid Loss: {avg_val_loss:.4f}, Valid PPL: {val_ppl:.2f}")