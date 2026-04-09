import time
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from config import *
from torch.utils.tensorboard import SummaryWriter
from dataset import get_dataloader


def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    model.to(device)
    total_loss = 0
    for batch in tqdm(data_loader, desc="训练"):
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    return total_loss / len(data_loader)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu"))

    # 加载数据
    data_loader = get_dataloader(True)

    # 创建模型
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter(LOG_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    best_loss = float('inf')

    # 模型训练
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, data_loader, optimizer, device)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f}")
        writer.add_scalar("train_loss", train_loss, epoch)

        if train_loss < best_loss:
            best_loss = train_loss
            model.save_pretrained(MODEL_DIR)


if __name__ == '__main__':
    train()
