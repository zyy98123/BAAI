import torch
import torch.nn as nn


class CosSimLoss(nn.Module):
    def __init__(self):
        super(CosSimLoss, self).__init__()

    def forward(self, vector_feature1, vector_feature2, label):
        loss = (vector_feature1 * vector_feature2).sum(dim=1)/(torch.norm(vector_feature1, dim=1) * torch.norm(vector_feature2, dim=1) + 1e-6)
        loss = (loss - label) ** 2.0
        # print(loss.size(), loss)
        loss = loss.mean()
        return loss


class MarginLoss(nn.Module):

    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, vector_feature1, vector_feature2, label):
        distance = torch.sqrt(torch.sum(((vector_feature1 - vector_feature2) ** 2.0), dim=1))
        loss_contrastive = torch.mean(
            torch.clamp(label, min=0) * torch.pow(distance, 2) +
            (1 - torch.clamp(label, min=0)) * torch.pow(torch.clamp(1 - distance, min=0.0), 2)
        )
        return loss_contrastive

