# from typing import List

import torch
import torch.nn as nn
from .backbone_fusion import ImageFusionModel
from .backbone import make_mlp



class ImageHeadMLP(nn.Module):
    def __init__(
        self,        
    ):

        super().__init__()
        self.backbone = ImageFusionModel()        
        self.goal_encoder = make_mlp( [2, 128, 64], 'relu', False, False, 0.0)      

        self.feat_util = nn.Sequential(
            nn.Linear(36864,256),            
            nn.LeakyReLU(),
            nn.Linear(256,22)            
        )

        self.feat_ext = nn.Sequential(
            nn.Linear(36864,512),            
            nn.LeakyReLU()
        )

        self.after_rnn = nn.Sequential(
            nn.Linear(512+64,256),            
            nn.LeakyReLU()
        )

        self.predict = nn.Linear(256,22)

    def forward(self, input, goal):
        
        image_features = self.backbone(input)                
        goal = self.goal_encoder(goal)

        # utility = self.feat_util(image_features)
        img_feat = self.feat_ext(image_features)

        img_feat_with_goal = torch.cat([img_feat, goal],dim=-1)

        final_feat = self.after_rnn(img_feat_with_goal)
        
        prediction = self.predict(final_feat)
          
        return img_feat, prediction




