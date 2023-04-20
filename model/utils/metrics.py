import numpy as np
import torch


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Accuracy_3(self):
         Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
         Acc1,Acc2,Acc3=Acc[0],Acc[1],Acc[2]
         return Acc1,Acc2,Acc3

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def F1_score(self):
        #  precision =  TP  / (TP + FP )
        #  recall    =  TP  / (TP + FN )
        precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        precision = np.nanmean(precision)
        recall = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        recall = np.nanmean(recall)
        f1_score= 2 * precision * recall / (precision + recall)
        return f1_score

    def F1_score_ma(self): #
        #  precision =  TP  / (TP + FP )
        #  recall    =  TP  / (TP + FN )
        precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        recall = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)

        f1_score1 = 2 * precision[0] * recall[0] / (precision[0] + recall[0])
        f1_score2 = 2 * precision[1] * recall[1] / (precision[1] + recall[1])
        f1_score3 = 2 * precision[2] * recall[2] / (precision[2] + recall[2])
        f1_score= (f1_score1 + f1_score2 + f1_score3) / 3
        return f1_score

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)








    def target_Proportion(self):  ###标签各类颜色占比
        pro = self.confusionMatrix.sum(axis=1) / self.confusionMatrix.sum()
        # pro = np.nanmean(pro)
        return pro[0], pro[1], pro[2]

    def predict_Proportion(self):  ####预测输出各类颜色占比
        pro = self.confusionMatrix.sum(axis=0) / self.confusionMatrix.sum()
        # pro = np.nanmean(pro)
        return pro[0], pro[1], pro[2]

    def Dice_coeff(self, logit, target, smooth=1e-5):

        logit = torch.nn.functional.sigmoid(logit)
        n, h, w = target.size()
        target_one_hot = torch.zeros(n, 3, h, w)
        target_one_hot = target_one_hot.cuda()
        target = target_one_hot.scatter_(1, target.long().view(n, 1, h, w), 1)
        # print(target)
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
            diceLoss = loss.sum() / N
            # print(diceLoss)
            totalLoss += diceLoss

        return totalLoss.numpy()








