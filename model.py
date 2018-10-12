import torch.nn as nn
import torch.nn.functional as F

#The model we'll be training
#It seems to be doing much better without bias
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 16, bias=False),
            nn.Linear(16, 1, bias=False)
        )

    def forward(self, x):
        return self.model(x)