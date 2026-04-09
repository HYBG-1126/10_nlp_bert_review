from datasets import load_dataset, ClassLabel
from pydantic.v1.utils import truncate
from rich import padding
from transformers import AutoTokenizer

from config import *


def preprocess():
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    # 加载数据
    dataset = load_dataset('csv', data_files={"train": str(RAW_FILE)})['train']
    # 删除列
    dataset = dataset.remove_columns(['cat'])
    # 过滤数据
    dataset = dataset.filter(lambda x: x["review"] is not None)
    # 分割数据,分层抽样
    dataset = dataset.cast_column('label', ClassLabel(names=['0', '1']))
    dataset_dict = dataset.train_test_split(test_size=0.2, stratify_by_column='label')
    # 处理数据
    dataset_dict = dataset_dict.map(
        lambda x: tokenizer(x['review'], padding='max_length', truncation=True, max_length=MAX_LEN), batched=True,
        remove_columns=['review'])
    dataset_dict = dataset_dict.rename_columns({"label": "labels"})
    print(dataset_dict)
    # 保存数据
    dataset_dict.save_to_disk(PROCESSED_DIR)


if __name__ == '__main__':
    preprocess()
