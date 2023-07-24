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
        self.goal_encoder = make_mlp( [2, 64, 128, 64], 'relu', False, False, 0.0)


        self.common = nn.Sequential(
            nn.Linear(512,128),            
            nn.LeakyReLU(),
            nn.Linear(128,128),            
            nn.ReLU(),
        )

        self.concat_goal = nn.Sequential(
            nn.Linear(64+128,128),            
            nn.LeakyReLU(),
            nn.Linear(128,64),            
            nn.LeakyReLU(),
        )

        self.way_pts = nn.Linear(64,10)

    def forward(self, input, goal):
        
        image_features, _ = self.backbone(input)                
        goal = self.goal_encoder(goal)

        feat_shared_common = self.common(image_features)

        img_feat_with_goal = torch.cat([feat_shared_common, goal],dim=-1)

        img_goal_feat = self.concat_goal(img_feat_with_goal)

        # predict = self.way_pts(img_goal_feat)
          
        return image_features, feat_shared_common, img_goal_feat



