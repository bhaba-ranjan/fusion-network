# from typing import List

import torch
import torch.nn as nn
from ..pcl.pcl_head import PclMLP
from ..image.image_head import ImageHeadMLP
# from .tf_model import CustomTransformerModel

def set_trainable_false(model):
    for param in model.parameters():
        param.requires_grad = False    

def torch_load_weights(path):
    check_point = torch.load(path)
    model_weights = check_point['model_state_dict']
    return model_weights

class MultiModalVAE(nn.Module):
    def __init__(self):

        super().__init__()
        
        self.image =  ImageHeadMLP()        
        self.pcl =  PclMLP()

        self.pcl_weights = torch_load_weights('/home/ranjan/Workspace/my_works/fusion-network/scripts/tf_pcl_full_ann_utility_sep_70_0.07738654711255934.pth')
        self.image_weights = torch_load_weights('/home/ranjan/Workspace/my_works/fusion-network/scripts/image_model_60_0.08670703555086381.pth')
        
        self.image.load_state_dict(self.image_weights, strict=False)
        self.pcl.load_state_dict(self.pcl_weights, strict=False)

        set_trainable_false(self.image)
        set_trainable_false(self.pcl)

        self.global_path_encoder = nn.Sequential(
            nn.Linear(44,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU()
        )

        self.feature_encoder = nn.Sequential(
            nn.Linear(1024+512,2304),
            nn.ELU(),
            nn.Linear(2304,1024),
            nn.ELU()
        )
        
        self.fuse_gp_and_perception = nn.Sequential(
            nn.Linear(1024+64,1024),
            nn.ELU()
        )

        self.fc_mu = nn.Linear(1024, 128)
        self.fc_logvar = nn.Linear(1024, 128)
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )


    def encode(self, x):
        encoder_output = self.encoder(x)
        return encoder_output
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)


    def forward(self, stacked_images, pcl, local_goal):

        image_feat, image_gp = self.image(stacked_images, local_goal)
        pcl_feat, pcl_gp = self.pcl(pcl, local_goal)

        backbone_feats = torch.cat([pcl_feat, image_feat], dim=-1)
        fusion_feat_encoding = self.feature_encoder(backbone_feats)        
        

        global_path_features = torch.cat([pcl_gp,image_gp], dim=-1)
        global_path_encoding = self.global_path_encoder(global_path_features)

        feat_encoding_concat = torch.cat([fusion_feat_encoding,global_path_encoding], dim=-1)
        fused_feat_encoding = self.fuse_gp_and_perception(feat_encoding_concat)          

        mu = self.fc_mu(fused_feat_encoding)
        logvar = self.fc_logvar(fused_feat_encoding)
        
        z = self.reparameterize(mu, logvar)
        
        output = self.decode(z)

        return output    
