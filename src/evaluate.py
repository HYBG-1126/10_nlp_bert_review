import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from config import *
from dataset import get_dataloader
from predict import predict_batch


def evaluate(model, dataloader, device):
    correct_count = 0
    total_count = 0

    with torch.no_grad():
        # 按批次前向传播
        for batch in tqdm(dataloader, desc="评估"):
            for key, value in batch.items():
                batch[key] = value.to(device)
            # 前向传播，得到预测概率
            batch_results = predict_batch(model, batch)
            # 做拉链
            for target, result in zip(batch['labels'], batch_results):
                total_count += 1
                # 判断预测概率是否大于 0.5 转成预测标签
                result = 1 if result > 0.5 else 0
                if result == target:
                    correct_count += 1

    return correct_count / total_count


def run_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 分词器

    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    print("模型加载成功")

    # 获取数据集
    test_loader = get_dataloader(False)

    # 调用评估逻辑
    acc = evaluate(model, test_loader, device)
    print("准确率：", acc)


if __name__ == '__main__':
    run_evaluate()
