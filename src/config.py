from pathlib import Path

# ROOT_DIR
ROOT_DIR = Path(__file__).parent.parent

# DATA_DIR
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

#
RAW_FILE = RAW_DIR / "online_shopping_10_cats.csv"

# MODEL_DIR
MODEL_DIR = ROOT_DIR / "models"

BERT_MODEL_NAME = "google-bert/bert-base-chinese"

# LOG_DIR
LOG_DIR = ROOT_DIR / "logs"

# 超参数
EMBEDDING_SIZE = 128
MAX_LEN = 128  # 最长序列长度
BATCH_SIZE = 64  # 批次大小
EPOCHS = 1  # 训练轮数
LEARNING_RATE = 1e-5  # 学习率
SEED = 42  # 随机数种子
HIDDEN_SIZE = 256  # 隐藏层大小
