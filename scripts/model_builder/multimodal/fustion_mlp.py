# from typing import List

import torch
import torch.nn as nn
from .custom_activation import CustomActivation




class FusionMLP(nn.Module):
    def __init__(
        self,        
    ):

        super().__init__()

        self.linear1 = nn.Linear(745471, 512)
        self.linear2 = nn.Linear(490581, 2 * 256)
        self.linear3 = nn.Linear(124344 + (1*256), 512)
        self.linear4 = nn.Linear(512+2*128,256)
        # self.linear5 = nn.Linear(512,512)
        self.linear6 = nn.GRU(256,32,2, bidirectional = True)
        # self.linear7 = nn.Linear(128,64)
        self.linear8 = nn.GRU(64, 32)
        self.linear9 = nn.Linear(32, 2)


        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(2 * 256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        # self.bn5 = nn.BatchNorm1d(128)
        # self.bn6 = nn.Tanh(64)
        # self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(16)
        # self.bn7 = nn.BatchNorm1d(64)
        
        

        self.act1 = nn.LeakyReLU()
        self.act2 = nn.LeakyReLU()
        self.act3 = nn.LeakyReLU()
        self.act4 = nn.ReLU()
        self.act5 = nn.ReLU()
        self.act6 = nn.Tanh()
        self.act7 = nn.ReLU()
        self.act8 = nn.Tanh()
        self.act9 = CustomActivation()


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

        # x = self.bn5(self.linear5(x))
        # x = self.act5(x)  

        x, h = self.linear6(x)
        x = self.act6(x)
        
        
        # x = self.bn7(self.linear7(x))
        # x = self.act7(x)
        
        x, h = self.linear8(x)
        x = self.act8(x)

        x = self.linear9(x)

        return x




