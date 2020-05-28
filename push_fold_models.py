import torch
import torch.nn as nn
import torch.nn.functional as F

class Pusher(nn.Module):
    def __init__(self):
        super(Pusher, self).__init__()
        self.FC1 = nn.Linear(30, 10) # Card 1 = 13 ranks, Card 2 = 13 ranks, Suited/Unsuited/Paired = 3, Stack size
        self.FC2 = nn.Linear(10, 5)
        self.FC3 = nn.Linear(5, 1)

    def forward(self, card1, card2, suited, stacksize):
        x = torch.cat((card1, card2, suited, stacksize), 1)
        x = F.relu(self.FC1(x))
        x = F.relu(self.FC2(x))
        x = torch.sigmoid(self.FC3(x))
        return x

class Caller(nn.Module):
    def __init__(self):
        super(Caller, self).__init__()
        self.FC1 = nn.Linear(30, 10) # Card 1 = 13 ranks, Card 2 = 13 ranks, Suited/Unsuited/Paired = 3, Stack size
        self.FC2 = nn.Linear(10, 5)
        self.FC3 = nn.Linear(5, 1)

    def forward(self, card1, card2, suited, stacksize):
        x = torch.cat((card1, card2, suited, stacksize), 1)
        x = F.relu(self.FC1(x))
        x = F.relu(self.FC2(x))
        x = torch.sigmoid(self.FC3(x))
        return x