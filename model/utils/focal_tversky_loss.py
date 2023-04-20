import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np

class TverskyLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6, alpha=0.7, beta=0.3):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.nn.functional.softmax(inputs,dim=1)

        categorical = targets.long()#.cuda()
        categorical = F.one_hot(categorical, 3)
        #print(categorical.size())

        targets = categorical.permute(0, -1, 1,2).float()

        TP = torch.sum((inputs * targets), (1, 2, 3))
        FP = torch.sum((1. - targets) * inputs, (1, 2, 3))
        FN = torch.sum((targets * (1. - inputs)), (1, 2, 3))

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return torch.mean(1 - Tversky)

class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6, alpha=0.7, beta=0.3, gamma=0.75):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.nn.functional.softmax(inputs,dim=1)

        #categorical = torch.from_numpy(np.array(targets)).long()
        categorical = targets.long()  # .cuda()
        categorical = F.one_hot(categorical, 3)  # 此处是类别数nclass=3
        # print(categorical.size())

        targets = categorical.permute(0, -1, 1, 2).float()

        # flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = torch.sum((inputs * targets), (1, 2, 3))
        FP = torch.sum((1. - targets) * inputs, (1, 2, 3))
        FN = torch.sum((targets * (1. - inputs)), (1, 2, 3))

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        return torch.mean(FocalTversky)




if __name__ == "__main__":
    tversky = TverskyLoss()
    focal_tversky = FocalTverskyLoss()
    a = torch.rand(1, 1, 256, 256)  # .cuda()
    b = torch.rand(1, 256, 256)  # .cuda()
    print(a)
    print(b)

    print(tversky(a, b).item())
    print(focal_tversky(a, b).item())

