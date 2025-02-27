import torch
import torch.nn as nn


class CosSimLoss(nn.Module):
    def __init__(self):
        super(CosSimLoss, self).__init__()

    def forward(self, vector_feature1, vector_feature2, label):
        loss = (vector_feature1 * vector_feature2).sum(dim=1) / (
                    torch.norm(vector_feature1, dim=1) * torch.norm(vector_feature2, dim=1) + 1e-6)
        cosine_similarity = loss
        loss = (loss - label) ** 2.0
        # print(loss.size(), loss)
        loss = loss.mean()
        prediction = torch.where(cosine_similarity >= 0, torch.tensor(1), torch.tensor(-1))

        return prediction, loss

# validation loss
class MarginLoss(nn.Module):

    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, vector_feature1, vector_feature2, label):
        label = (label + 1) / 2
        distance = torch.sqrt(torch.sum((vector_feature1 - vector_feature2) ** 2, dim=1) + 1e-9)
        loss_contrastive = torch.mean(
            label * torch.pow(distance, 2) +
            (1 - label) * torch.pow(torch.clamp(1 - distance, min=0.0), 2)
        )
        prediction = torch.where(distance >= 0.5, torch.tensor(1), torch.tensor(0))

        return prediction, loss_contrastive