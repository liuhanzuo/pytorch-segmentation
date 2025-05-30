import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils import class_weight 
from utils.lovasz_losses import lovasz_softmax

def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    #cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        target = target.clone().detach()
        
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        target = target.clone().detach()
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()

class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
    
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss

class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
    
    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss
    
class HybridLoss(nn.Module):
    def __init__(self, 
                 ignore_index=255, 
                 focal_weight=None, 
                 focal_gamma=2.0,
                 lovasz_classes='present',
                 dice_smooth=1.0,
                 weights=[0.5, 0.3, 0.2]):
        super().__init__()
        self.ignore_index = ignore_index
        
        self.focal = FocalLoss(gamma=focal_gamma, 
                             alpha=focal_weight,
                             ignore_index=ignore_index)
        
        self.lovasz = LovaszSoftmax(classes=lovasz_classes,
                                  ignore_index=ignore_index)
        
        self.dice = DiceLoss(smooth=dice_smooth,
                            ignore_index=ignore_index)
        
        assert len(weights) == 3, "weights need three keywords[focal, lovasz, dice]"
        self.weights = weights

    def forward(self, output, target):

        focal_loss = self.focal(output, target)
        lovasz_loss = self.lovasz(output, target)
        dice_loss = self.dice(output, target)
        
        total_loss = (self.weights[0] * focal_loss + 
                     self.weights[1] * lovasz_loss + 
                     self.weights[2] * dice_loss)
        
        return total_loss

    @staticmethod
    def calculate_class_weights(dataloader, n_classes):
        """计算类别权重（示例方法）"""
        class_counts = torch.zeros(n_classes)
        for _, labels in dataloader:
            mask = (labels != 255)  # 忽略255
            hist = torch.histc(labels[mask].float(), 
                              bins=n_classes, 
                              min=0, 
                              max=n_classes-1)
            class_counts += hist
        
        weights = 1.0 / (class_counts + 1e-6)  
        weights /= weights.sum()  
        
        return weights.cuda()
