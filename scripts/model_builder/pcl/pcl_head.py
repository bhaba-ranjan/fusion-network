# from typing import List

import torch
import torch.nn as nn
from ..image.backbone import make_mlp
from .pointnet_backbone import PclBackbone



class PclMLP(nn.Module):
    def __init__(
        self,        
    ):

        super().__init__()

        self.backbone_pcl = PclBackbone().float()

        self.lstm_input = 158400
        self.hidden_state_dim = 1024
        self.num_layers = 4

        # self.rnn = nn.RNN(self.lstm_input, self.hidden_state_dim, self.num_layers, nonlinearity='relu',batch_first=True)

        self.featur_ext = nn.Sequential(
            nn.Linear(158400,2048),
            nn.ELU(),
            nn.Linear(2048,1024),
            nn.ELU()                                             
        )

        self.after_rnn = nn.Sequential(
            nn.Linear(1024+64,512),
            nn.ELU()                              
        )        

        self.featt_util = nn.Sequential(
            nn.Linear(1024,128),
            nn.ELU(),
            nn.Linear(128,22)                                          
        )        

        self.predict = nn.Linear(512,22)            

        self.goal_encoder = make_mlp( [2, 128, 64], 'relu', False, False, 0.0)
                

    def forward(self, input, goal):
        
        # batch_size = input.size()[0]

        # h0 = torch.zeros(self.num_layers, 1, self.hidden_state_dim, device='cuda')
        # c0 = torch.zeros(self.num_layers, 1, self.hidden_state_dim,device='cuda')

        point_cloud_feat = self.backbone_pcl(input.float())        
        goal = self.goal_encoder(goal)            
        
        # pcl_goal_concat = point_cloud_feat.unsqueeze(0)

        # rnn_out, _ = self.rnn(pcl_goal_concat, h0)

        # print(f'rnn output: {rnn_out.shape}')

        rnn_out = self.featur_ext(point_cloud_feat)

        rnn_goal = torch.cat([rnn_out, goal],dim=-1)

        final_feat = self.after_rnn(rnn_goal)
        
        prediction = self.predict(final_feat)

        utility = self.featt_util(rnn_out)

        return utility, prediction



