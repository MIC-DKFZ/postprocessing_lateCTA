import torch
from torch import nn
import torch.nn.functional as F
from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
import numpy as np
import os,sys


class TimeDistL1_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, weight_dice=1, weight_dist=1, weight_time=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        :param weight_dist:
        :param weight_time:
        """
        super(TimeDistL1_loss, self).__init__()

        self.weight_dice = weight_dice
        self.ignore_label = ignore_label

        self.weight_dist = weight_dist
        self.weight_time = weight_time

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, brain_mask: torch.Tensor, vessel_mask: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=2
        :param net_output:
        :param target:
        :param brain_mask:
        :param vessel_mask:
        :return:
        """

        # pred_vessels = (net_output[:,0] <= 0).float()
        #pred_vessels = net_output[:,-1] 
        #pred_vessels = F.relu(-net_output[:,0])

        # TODO: deactivate this!!
        dc_loss = self.dc(net_output[:,-1], vessel_mask.squeeze()) \
            if self.weight_dice != 0 else 0
        

        n_brain_voxels = brain_mask.sum()
        n_vessel_voxels = vessel_mask.sum()

        #net_output[:,1] = F.sigmoid(net_output[:,1])

        dist_loss = F.l1_loss(net_output[:,0]*brain_mask.squeeze(), 
                              target[:,0]*brain_mask.squeeze(), 
                              reduction="sum")
        tta_loss = F.l1_loss(net_output[:,1]*vessel_mask.squeeze(), 
                             target[:,1]*vessel_mask.squeeze(), 
                             reduction="sum")
        
       # fft_loss_fn = FourierLoss(loss_type='l1', weight_high_freq=True, dim=3)  # for 3D volumes
       # fft_loss = fft_loss_fn(net_output[:,0].unsqueeze(1).float(), 
       #                    target[:,0].unsqueeze(1).float())

        dist_loss /= (n_brain_voxels + np.finfo(float).eps)
        tta_loss /= (n_vessel_voxels + np.finfo(float).eps)

        sobel_loss_fn = MaskedSobelEdgeLoss3D()
        sobel_loss = sobel_loss_fn(net_output[:,0].unsqueeze(1),
                                   target[:,0].unsqueeze(1),
                                   brain_mask)


        
        result = self.weight_dice * dc_loss + self.weight_dist*dist_loss + self.weight_time*tta_loss + 0.1*sobel_loss

       # print(fft_loss, sobel_loss)

        return result
    


class TimeDistL2_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, weight_dice=1, weight_dist=1, weight_time=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        :param weight_dist:
        :param weight_time:
        """
        super(TimeDistL2_loss, self).__init__()

        self.weight_dice = weight_dice
        self.ignore_label = ignore_label

        self.weight_dist = weight_dist
        self.weight_time = weight_time

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, brain_mask: torch.Tensor, vessel_mask: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=2
        :param net_output:
        :param target:
        :param brain_mask:
        :param vessel_mask:
        :return:
        """
        # pred_vessels = (net_output[:,0] <= 0).float()
        #pred_vessels = net_output[:,-1]
        #pred_vessels = F.relu(-net_output[:,0])

        # TODO: deactivate this!!
        dc_loss = self.dc(net_output[:,-1], vessel_mask.squeeze()) \
            if self.weight_dice != 0 else 0
        

        n_brain_voxels = brain_mask.sum()
        n_vessel_voxels = vessel_mask.sum()

        #net_output[:,1] = F.sigmoid(net_output[:,1])

        dist_loss = F.mse_loss(net_output[:,0]*brain_mask.squeeze(), 
                              target[:,0]*brain_mask.squeeze(), 
                              reduction="sum")
        tta_loss = F.mse_loss(net_output[:,1]*vessel_mask.squeeze(), 
                             target[:,1]*vessel_mask.squeeze(), 
                             reduction="sum")


        dist_loss /= (n_brain_voxels + np.finfo(float).eps)
        tta_loss /= (n_vessel_voxels + np.finfo(float).eps)

        result = self.weight_dice * dc_loss + self.weight_dist*dist_loss + self.weight_time*tta_loss
        
        return result
    


class FourierLoss(nn.Module):
    def __init__(self, loss_type='l1', weight_high_freq=True, dim=2):
        super().__init__()
        self.loss_type = loss_type
        self.weight_high_freq = weight_high_freq
        self.dim = dim  # 2 for 2D images, 3 for volumes

    def forward(self, pred, target):
        # pred, target: (B, C, H, W) or (B, C, D, H, W)
        assert pred.shape == target.shape

        # Compute FFT
        fft_fn = torch.fft.fft2 if self.dim == 2 else torch.fft.fftn
        fft_pred = fft_fn(pred, norm='ortho')
        fft_target = fft_fn(target, norm='ortho')

        # Compute magnitude spectra
        mag_pred = torch.abs(fft_pred)
        mag_target = torch.abs(fft_target)

        # Optionally apply high-frequency emphasis
        if self.weight_high_freq:
            weight = self._high_freq_weight(pred.shape[-self.dim:], 
                                            device=pred.device)
            mag_pred = mag_pred * weight
            mag_target = mag_target * weight

        if self.loss_type == 'l1':
            return torch.mean(torch.abs(mag_pred - mag_target))
        elif self.loss_type == 'l2':
            return torch.mean((mag_pred - mag_target) ** 2)
        else:
            raise ValueError("loss_type must be 'l1' or 'l2'")

    def _high_freq_weight(self, shape, device):
        """
        Returns a frequency weighting mask that emphasizes high frequencies.
        shape: spatial dimensions (H, W) or (D, H, W)
        """
        grids = torch.meshgrid([torch.fft.fftfreq(s) for s in shape], indexing='ij')
        freq_mag = torch.sqrt(sum(g ** 2 for g in grids)).to(torch.float32)
        freq_mag /= freq_mag.max() + 1e-8  # normalize to [0, 1]
        # weight = freq_mag.to(next(self.parameters()).device)
        weight = freq_mag.to(device)
        return weight.unsqueeze(0).unsqueeze(0)
    

class MaskedSobelEdgeLoss3D(nn.Module):
    def __init__(self, loss_type='l1'):
        super().__init__()
        self.loss_type = loss_type
        self.gx, self.gy, self.gz = self._get_sobel_kernels()

    def _get_sobel_kernels(self):
        gx = torch.tensor([
            [[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]],
            [[-3, 0, 3], [-6, 0, 6], [-3, 0, 3]],
            [[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]
        ], dtype=torch.float32).view(1, 1, 3, 3, 3)

        gy = torch.tensor([
            [[-1, -3, -1], [0, 0, 0], [1, 3, 1]],
            [[-3, -6, -3], [0, 0, 0], [3, 6, 3]],
            [[-1, -3, -1], [0, 0, 0], [1, 3, 1]]
        ], dtype=torch.float32).view(1, 1, 3, 3, 3)

        gz = torch.tensor([
            [[-1, -3, -1], [-3, -6, -3], [-1, -3, -1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 3, 1], [3, 6, 3], [1, 3, 1]]
        ], dtype=torch.float32).view(1, 1, 3, 3, 3)

        return gx, gy, gz

    def _compute_edges(self, x):
        if x.shape[1] > 1:  # grayscale conversion
            x = x.mean(dim=1, keepdim=True)

        gx = self.gx.to(x.device, x.dtype)
        gy = self.gy.to(x.device, x.dtype)
        gz = self.gz.to(x.device, x.dtype)

        grad_x = F.conv3d(x, gx, padding=1)
        grad_y = F.conv3d(x, gy, padding=1)
        grad_z = F.conv3d(x, gz, padding=1)

        edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2 + 1e-8)
        return edge

    def forward(self, pred, target, mask):
        """
        pred, target: (B, C, D, H, W)
        mask: (B, 1, D, H, W), binary
        """
        edge_pred = self._compute_edges(pred)
        edge_target = self._compute_edges(target)

        diff = torch.abs(edge_pred - edge_target) if self.loss_type == 'l1' else (edge_pred - edge_target) ** 2

        # Apply mask
        masked_diff = diff * mask
        denom = mask.sum() + 1e-8  # avoid div by 0

        return masked_diff.sum() / denom
