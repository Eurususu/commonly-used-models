import torch
from models import build_model
import yaml
with open("./config/model/dynamic_model.yaml") as f:
    layers_cfg = yaml.safe_load(f)

model = build_model("dynamic_graph_model", layers_cfg=layers_cfg["model"]["kwargs"]["layers_cfg"])

print(model)

result = model(torch.randn(1, 3, 224, 224))
print(result.shape)
