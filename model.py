import torch.nn as nn

class AttritionNN(nn.Module):
    def __init__(self, input_dim):
        super(AttritionNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # binary output
            nn.Sigmoid()        # probability
        )
    def forward(self, x):
        return self.model(x)