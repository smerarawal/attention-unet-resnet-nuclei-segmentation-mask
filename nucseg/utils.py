import json
import os
import random
import time

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # fixed-size crops -> faster; eval shapes vary but that's fine


def worker_init_fn(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast(device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


class Timer:
    def __init__(self):
        self.start = self.last = time.time()

    @property
    def minutes(self):
        return (time.time() - self.start) / 60

    def lap(self):
        t = time.time()
        d, self.last = t - self.last, t
        return d


def nanmean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p
