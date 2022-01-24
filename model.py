import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.data import SequentialSampler
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings
from ophooks import register_ophooks_recursively, MemTracerOpHook

class Encoder(torch.nn.Module):
    def __init__(self, hidden_dim, is_ckp=False):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 4 * hidden_dim),
            torch.nn.Linear(4 * hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

        self.linear3 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim),
              torch.nn.Linear(hidden_dim, hidden_dim),
              torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.is_ckp = is_ckp

    def forward(self, x):
        h2 = self.linear1(x)
        if self.is_ckp:
            h3 = checkpoint(self.linear3, h2)
        else:
            h3 = self.linear3(h2)
        return h3


def get_data_loader(
    batch_size,
    total_samples,
    hidden_dim,
    device,
    data_type=torch.float,
    is_distrbuted=False,
):
    train_data = torch.randn(total_samples, hidden_dim, device=device, dtype=data_type)
    train_label = torch.empty(total_samples, dtype=torch.long, device=device).random_(
        hidden_dim
    )
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    if is_distrbuted:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = SequentialSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler
    )
    return train_loader


def get_bert_data_loader(
    batch_size, total_samples, sequence_length, device, is_distrbuted=False
):
    train_data = torch.randint(
        low=0,
        high=10,
        size=(total_samples, sequence_length),
        device=device,
        dtype=torch.long,
    )
    train_label = torch.zeros(total_samples, dtype=torch.long, device=device)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    if is_distrbuted:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = SequentialSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler
    )
    return train_loader


class SimpleModel(torch.nn.Module):
    def __init__(self, hidden_dim, seq_len, is_ckp=False, is_share_param=False):
        super(SimpleModel, self).__init__()
        config = BertConfig()
        config.vocab_size = 25
        config.max_position_embeddings = seq_len
        config.hidden_size = hidden_dim
        self.embeddings_1 = BertEmbeddings(config)

        self._is_share_param = is_share_param
        if is_share_param:
            self.embeddings_2 = self.embeddings_1
        else:
            self.embeddings_2 = BertEmbeddings(config)
        self.encoder = Encoder(hidden_dim, is_ckp)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        h1 = self.embeddings_1(x)
        h2 = self.embeddings_2(x)
        h3 = h1 + h2
        h3 = self.encoder(h3)
        return self.cross_entropy_loss(h3[:, 0], y)