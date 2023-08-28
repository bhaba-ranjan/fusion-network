import torch
import torch.nn as nn

class CustomTransformerModelImg(nn.Module):
    def __init__(self, hidden_dim = 256, num_layers = 2, nhead=2):
        super(CustomTransformerModelImg, self).__init__()                
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
    

    def forward(self, x):
        # linear_output = self.linear(x)
        transformer_output = self.transformer_encoder(x)
        # output = self.fc(transformer_output)
        return transformer_output
