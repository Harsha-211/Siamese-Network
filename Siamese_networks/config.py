import torch 
import torch.nn as nn
import torch.nn.functional as F

class ConstructiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ConstructiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self , feat1, feat2 , label):
        dist = F.pairwise_distance(feat1,feat2)
        loss = (1-label)*torch.pow(dist,2)+label*torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        return loss.mean()