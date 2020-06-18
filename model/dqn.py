import torch
import torch.nn as nn

class DQN(nn.Module):
    
    def __init__(self, input_channels, num_actions, img_size=84):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=8, stride=4)
        conv1_output_size = int((img_size - 8) / 4) + 1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        conv2_output_size = int((conv1_output_size - 4) / 2) + 1
        self.fc = nn.Linear(32 * (conv2_output_size ** 2), 256)
        self.output = nn.Linear(256, num_actions)
        
    def forward(self, state):
        out = self.conv1(state)
        out = torch.relu(out)
        out = self.conv2(out)
        out = torch.relu(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        out = torch.relu(out)
        out = self.output(out)
        return out
        