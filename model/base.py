import torch


class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
    
    @property
    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
