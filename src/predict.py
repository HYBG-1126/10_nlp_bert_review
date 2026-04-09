import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import BERT_MODEL_NAME, MAX_LEN, MODEL_DIR


def predict(text):
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
    result_batch = predict_batch(model, inputs)
    return result_batch[0]


def predict_batch(model, inputs):
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    result_batch = torch.softmax(outputs.logits, dim=-1)
    return result_batch[:, 1].tolist()


if __name__ == '__main__':
    result = predict("说的什么玩意")
    if result > 0.5:
        print(f"positive,置信度：{result}")
    else:
        print(f"negative,置信度：{1 - result}")
