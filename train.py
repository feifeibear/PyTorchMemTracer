import logging
import torch

from model import SimpleModel, get_bert_data_loader
# from pytorchmemtracer.ophooks import register_ophooks_recursively, MemTracerOpHook
from pytorchmemtracer import memtracer_wrapper

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

BATCH_SIZE = 8
HIDDEN_DIM = 128
SEQ_LEN = 128


def model_func():
    return SimpleModel(
        hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN, is_ckp=False, is_share_param=True
    )

LR = 5e-5
BETAS = (0.9, 0.999)
EPS = 1e-6
WEIGHT_DECAY = 0

config = {
    # The same format as optimizer config of DeepSpeed
    # https://www.deepspeed.ai/docs/config-json/#optimizer-parameters
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": LR,
            "betas": BETAS,
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "use_hybrid_adam": True,
        },
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 2 ** 3,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "default_chunk_size": 1024,
    "use_fake_dist": False,
    "use_cpu_embedding": False,
}

torch.manual_seed(0)
model = model_func()

optim = torch.optim.Adam(
    model.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY
)
model.cuda()

train_loader = get_bert_data_loader(BATCH_SIZE, 10000, 128, device, False)

# add this line for mem tracing
model = memtracer_wrapper(model)

for i, batch in enumerate(train_loader):
    optim.zero_grad()
    input_ids, labels = batch

    # change the backward API
    # loss = model(input_ids, labels)
    model.backward(loss)
    loss.backward()
    optim.zero_grad()
    optim.step()
    print(i, loss.item())
    if i == 10:
        break

# ophook_list[0].save_results("memstats.pkl")
# ophook_list[0].show_mem_stats()