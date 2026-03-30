import torch.optim as optim
from ._optimRegistry import register_optimizer

__all__ = []

register_optimizer("SGD")(optim.SGD)
register_optimizer("Adam")(optim.Adam)
register_optimizer("RMSprop")(optim.RMSprop)
register_optimizer("AdamW")(optim.AdamW)
register_optimizer("Adagrad")(optim.Adagrad)
register_optimizer("Adadelta")(optim.Adadelta)
register_optimizer("Adamax")(optim.Adamax)
register_optimizer("ASGD")(optim.ASGD)
register_optimizer("LBFGS")(optim.LBFGS)
register_optimizer("NAdam")(optim.NAdam)
register_optimizer("RAdam")(optim.RAdam)
register_optimizer("SparseAdam")(optim.SparseAdam)



