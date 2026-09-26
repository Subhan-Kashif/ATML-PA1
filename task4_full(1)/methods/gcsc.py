
from methods.vanilla import train as _train

def train_gcsc(cfg):
    assert cfg.get("use_randaugment", False), "gcsc.yaml must set use_randaugment: true"
    return _train(cfg)
