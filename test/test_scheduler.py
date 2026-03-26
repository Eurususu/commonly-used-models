import torch
from models import build_model
from optim import build_optimizer
from scheduler import build_scheduler, list_schedulers
import yaml

cfg = yaml.safe_load(open("config/model/resnet18.yaml"))

model = build_model(**cfg["model"])
print(model.get_model_info())

optimizer = build_optimizer(model.parameters(), **cfg["optim"])
print(optimizer)

scheduler = build_scheduler(optimizer, cfg["scheduler"]["name"], **cfg["scheduler"]["kwargs"])
print(scheduler)

print(list_schedulers())





