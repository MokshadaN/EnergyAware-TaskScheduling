import torch
import torch.nn as nn
import torch.optim as optim

class SchedulingModel(nn.Module):
    def __init__(self, n_input: int, n_actions: int, learning_rate: float = 0.001, device: str = 'cpu'):
        super(SchedulingModel, self).__init__()
        self.n_input = n_input
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.device = device
        self.fc1 = nn.Linear(n_input, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, n_actions)
        self.dropout = nn.Dropout(0.2)
        self._initialize_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.9)
        self.loss_fn = nn.SmoothL1Loss()
        self.dq1 = torch.tensor(0.0, device=device)
        self.dq2 = torch.tensor(0.0, device=device)
        
    def _initialize_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(state))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        values = self.fc5(x)
        return values
    
    def update_correction_factors(self, dq1: float, dq2: float):
        self.dq1 = torch.tensor(dq1, device=self.device)
        self.dq2 = torch.tensor(dq2, device=self.device)
