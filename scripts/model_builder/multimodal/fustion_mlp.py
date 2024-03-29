# from typing import List

import torch
import torch.nn as nn




class FusionMLP(nn.Module):
    def __init__(
        self,        
    ):

        super().__init__()

        self.linear1 = nn.Linear(745471, 512)
        self.linear2 = nn.Linear(490581,256)
        self.linear3 = nn.Linear(124344,128)
        self.linear4 = nn.Linear(3*128,64)
        self.linear5 = nn.Linear(64,32)
        self.linear6 = nn.Linear(32,16)
        self.linear7 = nn.Linear(16,2)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(32)
        self.bn6 = nn.BatchNorm1d(16)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.act4 = nn.ReLU()
        self.act5 = nn.ReLU()
        self.act6 = nn.ReLU()


    def forward(self, input_l1, input_l2, input_l3, goal, prev_cmd_vel):
        
        x = self.bn1(self.linear1(input_l1))
        x = self.act1(x)        
       

        x = torch.cat([input_l2,x], dim=-1)
        x = self.bn2(self.linear2(x))
        x = self.act2(x)          

        x = torch.cat([input_l3,x], dim=-1)
        x = self.bn3(self.linear3(x))
        x = self.act3(x)   
      
        # print(x.shape,goal.shape, prev_cmd_vel.shape)
        x = torch.cat([x, goal, prev_cmd_vel], dim=-1)
        x = self.bn4(self.linear4(x))
        x = self.act4(x)  

        x = self.bn5(self.linear5(x))
        x = self.act5(x)  
        
        x = self.bn6(self.linear6(x))
        x = self.act6(x)  
        
        x = self.linear7(x)
        
        return x




