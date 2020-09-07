import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam, SGD

"""The global CNN and Linear layers of the Network""" 

class Net(nn.Module):
    def __init__(self, env):
        """Builds the CNN's until the output reduces to 2x2"""
        super(Net, self).__init__()

        # Shape of the input H x W x O (shape(H x W) x objects)
        objects = len(env._value_mapping)
        shape = env.observation_spec()['board'].shape

        # Filter depths for the CNN
        filters =[objects] + [16, 32, 64, 64] 
        self.conv_layers = []

        for i in range(4):
            if shape[0] > 3 and shape[1] > 3:
              self.conv_layers.append(nn.Conv2d(filters[i], filters[i+1], kernel_size = 3, stride = 1))
              self.conv_layers.append(nn.ReLU(inplace=True))
              shape = shape[0] - 2, shape[1] - 2

            elif shape[0] == 2 and shape[1] == 2:
              break
            
            else:
              #Calculating the padding to be added to reduce it to 2x2 
              padx = 1 if shape[0] == 3 else (shape[0] == 2)*2
              pady = 1 if shape[1] == 3 else (shape[1] == 2)*2

              shape = shape[0] - 2 + padx, shape[1] - 2 + pady

              self.conv_layers.append(nn.ZeroPad2d((pady//2,pady - pady//2, padx//2, padx - padx//2)))
              self.conv_layers.append(nn.Conv2d(filters[i], filters[i+1], kernel_size = 3, stride = 1))
              self.conv_layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*self.conv_layers)
        self.fc1 = nn.Linear(filters[i]*4, 64)
        self.fc2 = nn.Linear(64, 4)
        
            
    def forward(self, x):
      x = self.conv(x)
      x = torch.flatten(x, 1)
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)

      return x


"""The Global and Local layers of the Network"""
class AgentNet(nn.Module):
    def __init__(self, env):
        super(AgentNet, self).__init__()
        self.global_net = Net(env)
        inp = 4
        self.local_net = nn.Sequential(nn.Linear(inp, 64),
                                       nn.Linear(64, 4)
                                       )
    
    def forward(self, state_global, state_local):
        state_global = self.global_net(state_global)
        state_local = self.local_net(state_local)
        
        combined = (state_global+state_local)/2
        return combined