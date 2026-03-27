from dataset import list_datasets
from models import list_models
from optim import list_optimizers
from loss import list_losses
from scheduler import list_schedulers



print("Available datasets:",list_datasets())
print("Available models:",list_models())
print("Available losses:",list_losses())
print("Available optimizers:",list_optimizers())
print("Available schedulers:",list_schedulers())