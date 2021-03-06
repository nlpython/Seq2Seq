from torch.utils.data import Dataset
from datetime import datetime
import numpy as np
import torch

PAD_ID = 0


class DateDataset(Dataset):
    def __init__(self, n):
        np.random.seed(1)
        self.date_cn = []
        self.date_en = []
        for timestamp in np.random.randint(143835585, 2043835585, n):
            date = datetime.fromtimestamp(timestamp)
            self.date_cn.append(date.strftime("%y-%m-%d"))
            self.date_en.append(date.strftime("%d/%b/%Y"))
        self.vocab = set(
            [str(i) for i in range(0, 10)] + ["-", "/", "<GO>", "<EOS>"] + [i.split("/")[1] for i in self.date_en]
        )
        self.v2i = {v: i for i, v in enumerate(sorted(list(self.vocab)), start=1)}
        self.v2i["<PAD>"] = PAD_ID
        self.vocab.add("<PAD>")
        self.i2v = {i: v for v, i in self.v2i.items()}
        self.x, self.y = [], []
        for cn, en in zip(self.date_cn, self.date_en):
            self.x.append([self.v2i[v] for v in cn])
            self.y.append([self.v2i["<GO>"], ] + [self.v2i[v] for v in en[:3]] + [
                self.v2i[en[3:6]]] + [self.v2i[v] for v in en[6:]] + [self.v2i["<EOS>"], ])
        self.x, self.y = np.array(self.x), np.array(self.y)
        self.start_token = self.v2i["<GO>"]
        self.end_token = self.v2i["<EOS>"]

    def __len__(self):
        return len(self.x)

    @property
    def num_word(self):
        return len(self.vocab)

    def __getitem__(self, index):
        # return self.x[index], self.y[index][:-1], self.y[index][1:], len(self.y[index]) - 1
        return {
            'enc_input': torch.tensor(self.x[index], dtype=torch.long),
            'dec_input': torch.tensor(self.y[index][:-1], dtype=torch.long),
            'dec_output': torch.tensor(self.y[index][1:], dtype=torch.long),
            'length': torch.tensor(len(self.y[index]) - 1)
        }

    def idx2str(self, idx):
        x = []
        if not isinstance(idx, list):
            idx = idx.tolist()
        for i in idx:
            if i == self.end_token:
                break
            x.append(self.i2v[i])
        return "".join(x)

    def str2idx(self, str):
        return [self.v2i[i] for i in str]
