from torch import nn

# neural network
class DQN(nn.Module):
    def __init__(self, hidden_state, states, actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(states, hidden_state, ),
            nn.ReLU(),
            nn.Linear(hidden_state, hidden_state),
            nn.ReLU(),
            nn.Linear(hidden_state, actions)

        )

    def forward(self, x):
        return self.net(x.float())
