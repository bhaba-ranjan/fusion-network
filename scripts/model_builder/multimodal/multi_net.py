# from typing import List

import torch
import torch.nn as nn
from ..pcl.pcl_head import PclMLP
from ..image.image_head import ImageHeadMLP
from .tf_model import CustomTransformerModel
from .tf_img import CustomTransformerModelImg
from .tf_pcl import CustomTransformerModelPcl
from ..image.backbone import make_mlp

def set_trainable_false(model):
    for param in model.parameters():
        param.requires_grad = False    

def torch_load_weights(path):
    check_point = torch.load(path)
    model_weights = check_point['model_state_dict']
    return model_weights

class MultiModalNet(nn.Module):
    def __init__(self):

        super().__init__()
        
        self.image =  ImageHeadMLP()        
        self.pcl =  PclMLP()
        self.transformer = CustomTransformerModel()
        self.transformer_img = CustomTransformerModelImg()
        self.transformer_pcl = CustomTransformerModelPcl()
        self.goal_encoder = make_mlp( [2, 128, 64], 'relu', False, False, 0.0)

        self.multimodal_feat_compress = nn.Sequential(
            nn.Linear(256+512+64,1024),
            nn.ELU()           
        )        

        self.global_path_predictor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512,8),            
        )

        self.global_path_encoder = nn.Sequential(
            nn.Linear(8, 256),
            nn.ELU(),
            nn.Linear(256,128),
            nn.ELU()
        )

        # self.global_path_fusion = nn.Sequential(
        #     nn.Linear(8+8, 256),
        #     nn.ELU(),
        #     nn.Linear(256,128),
        #     nn.ELU()
        # )

        self.joint_perception_path_feautres = nn.Sequential(
            nn.Linear(128+1024,512),
            nn.ELU()
        )

        self.predict_vel = nn.Linear(512,2)        

    def forward(self, stacked_images, pcl, local_goal):
        

        rnn_img_out = self.image(stacked_images, local_goal)
        rnn_pcl_out = self.pcl(pcl, local_goal)

        rnn_img_input = rnn_img_out.unsqueeze(0)
        rnn_pcl_input = rnn_pcl_out.unsqueeze(0)

        tf_img_out = self.transformer_img(rnn_img_input)
        tf_pcl_out = self.transformer_pcl(rnn_pcl_input)

        tf_img_out = tf_img_out.squeeze(0)
        tf_pcl_out = tf_pcl_out.squeeze(0)

        encoded_goal = self.goal_encoder(local_goal)

        fsn_feat_with_lc_goal = torch.cat([tf_img_out, tf_pcl_out, encoded_goal], dim=-1).unsqueeze(0)
        tf_multimodal_out = self.transformer(fsn_feat_with_lc_goal)

        tf_multimodal_out = tf_multimodal_out.squeeze(0)

        multimodal_feat_reduced = self.multimodal_feat_compress(tf_multimodal_out)

        fsn_global_path = self.global_path_predictor(multimodal_feat_reduced)

        encoded_global_path = self.global_path_encoder(fsn_global_path)
        

        final_features_concat = torch.cat([multimodal_feat_reduced, encoded_global_path], dim=-1)
                
        fustion_perception_path = self.joint_perception_path_feautres(final_features_concat)

        fusion_vel = self.predict_vel(fustion_perception_path)

        return fsn_global_path, fusion_vel
