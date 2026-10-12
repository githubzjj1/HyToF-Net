import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, gamma=2.0, alpha=None, reduction='mean'):

        super(FocalLabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        
        logprobs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(logprobs)
        
        pt = probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        
        confidence = 1. - self.smoothing
        nll_loss = -logprobs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        ls_loss = confidence * nll_loss + self.smoothing * smooth_loss
        
        if self.alpha is not None:
            
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_weight = self.alpha.gather(0, targets)
            ls_loss = ls_loss * alpha_weight
            
        
        focal_weight = (1. - pt) ** self.gamma
        combined_loss = focal_weight * ls_loss
        
        
        if self.reduction == 'mean':
            return combined_loss.mean()
        elif self.reduction == 'sum':
            return combined_loss.sum()
        else:
            return combined_loss
        