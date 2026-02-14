
import torch
import torch.nn as nn

class ActorGaussian(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(64, action_dim)

        # Learnable log-std (one per action dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, x):
        h = self.body(x)
        mu = self.mu_head(h)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std

class Critic(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
