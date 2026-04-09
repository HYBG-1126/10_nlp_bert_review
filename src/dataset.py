from datasets import load_from_disk
from torch.utils.data import DataLoader

from config import PROCESSED_DIR, BATCH_SIZE


def get_dataloader(train: bool):
    path = str(PROCESSED_DIR / ("train" if train else "test"))
    dataset = load_from_disk(path)
    dataset.set_format(type='torch')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


if __name__ == '__main__':
    train_loader = get_dataloader(train=True)
    for batch in train_loader:
        for key, value in batch.items():
            print(key, value.shape)
        break
