import torch
import torch.nn as nn


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'dice':
            return self.Diceloss
        elif mode == 'iou':
            return self.SoftIoULoss
        elif mode == 'dice_focal':
            return self.DiceLoss_Focal
        elif mode == 'dice_ce':
            return self.DiceLoss_CE
        elif mode == 'FocalTversky':
            return self.FocalTversky
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def Diceloss(self, logit, target, smooth=1e-5):
        logit = torch.nn.functional.sigmoid(logit)
        n, h, w = target.size()
        target_one_hot = torch.zeros(n, 3, h, w)
        target_one_hot = target_one_hot.cuda()
        target = target_one_hot.scatter_(1, target.long().view(n, 1, h, w), 1)
        #print(target)
        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        totalLoss = 0

        for i in range(C):
            logit1 = logit[:, i]
            # print(logit1)
            target1 = target[:, i]
            # print(target1)
            N = target1.size(0)

            input_flat = logit1.view(N, -1)
            target_flat = target1.view(N, -1)

            intersection = input_flat * target_flat

            loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
            diceLoss = 1 - loss.sum() / N
            # print(diceLoss)
            totalLoss += diceLoss

        return totalLoss

    def SoftIoULoss(self, input, target, n_classes=3):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)
        pred = torch.nn.functional.softmax(input, dim=1)

        n, h, w = target.size()
        target_onehot = torch.zeros(n, n_classes, h, w)
        target_onehot = target_onehot.cuda()
        target_onehot = target_onehot.scatter_(1, target.long().view(n, 1, h, w), 1)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return 1 - loss.mean()

    def DiceLoss_Focal(self, logit, target):
        loss_dice = SegmentationLosses.Diceloss(self, logit, target)
        loss_focal = SegmentationLosses.FocalLoss(self, logit, target)
        return loss_dice + loss_focal

    def DiceLoss_CE(self, logit, target):
        loss_dice = SegmentationLosses.Diceloss(self, logit, target)
        loss_ce = SegmentationLosses.CrossEntropyLoss(self, logit, target)
        return loss_dice + loss_ce

    def FocalTversky(self, inputs, targets, smooth=1e-6, alpha=0.7, beta=0.3, gamma=0.75):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.nn.functional.sigmoid(inputs)
        n, h, w = targets.size()
        targets_one_hot = torch.zeros(n, 3, h, w)
        targets_one_hot = targets_one_hot#.cuda()
        targets = targets_one_hot.scatter_(1, targets.long().view(n, 1, h, w), 1)
        print(inputs.size())
        print(targets.size())

        TP = torch.sum((inputs * targets), (1, 2, 3))
        FP = torch.sum((1. - targets) * inputs, (1, 2, 3))
        FN = torch.sum((targets * (1. - inputs)), (1, 2, 3))

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        return torch.mean(FocalTversky)



if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7)  # .cuda()
    b = torch.rand(1, 7, 7)  # .cuda()
    print(a)
    print(b)


    print(loss.FocalTversky(a, b).item())
    # print(loss.CrossEntropyLoss(a, b).item())
    # print(loss.Diceloss(a, b).item())
    # print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    # print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
    # print(loss.SoftIoULoss(a, b).item())
    # print(loss.DiceLoss_Focal(a, b).item())
