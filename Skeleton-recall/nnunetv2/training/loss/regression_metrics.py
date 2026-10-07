import os,sys
import torch
import numpy as np


def masked_psnr(pred : torch.tensor, target : torch.tensor, mask : torch.tensor = None):
    """
    Compute masked PSNR between tensors of shape
    [H, W, D]
    
    Params
    ------
    pred : predicted tensor by network
    target : target tensor
    mask : mask with coordinates where to compute PSNR
        (default : None)


    Returns
    -------
    psnr : PSNR metric
    
    """

    if not(isinstance(pred, torch.Tensor)):
        if isinstance(pred, list) and len(pred) == 1: 
            pred = torch.from_numpy(pred)
        else:
            pred = torch.from_numpy(pred)
    if not(isinstance(target, torch.Tensor)):
        if isinstance(target, list) and len(target) == 1: 
            target = torch.from_numpy(target)
        else:
            target = torch.from_numpy(target)
    if mask is not None:
        if not(isinstance(mask, torch.Tensor)):
            if isinstance(mask, list) and len(mask) == 1: 
                mask = torch.from_numpy(mask[0])
            else:
                mask = torch.from_numpy(mask)

    if mask is not None:
        mask = mask.squeeze()
        pred, target = pred[mask > 0], target[mask > 0]

    pred, target = pred.flatten(), target.flatten()

    psnr = torch.tensor([0.])

    if (pred.shape[0] > 0) and (target.shape[0] > 0):
        mse = torch.mean((pred-target)**2)

        psnr = 20*torch.log10(torch.max(target))-10*torch.log10(mse + np.finfo(float).eps)

    # Avoid NaNs
    psnr = psnr.item()
    if np.isnan(psnr):
        psnr = 0.0

    return psnr


def masked_mae(pred : torch.tensor, target : torch.tensor, mask : torch.tensor = None):
    """
    Compute masked MAE between tensors of shape
    [H, W, D]
    
    Params
    ------
    pred : predicted tensor by network
    target : target tensor
    mask : mask with coordinates where to compute MAE
        (default : None)


    Returns
    -------
    mae : MAE metric
    
    """

    if not(isinstance(pred, torch.Tensor)):
        if isinstance(pred, list) and len(pred) == 1: 
            pred = torch.from_numpy(pred)
        else:
            pred = torch.from_numpy(pred)
    if not(isinstance(target, torch.Tensor)):
        if isinstance(target, list) and len(target) == 1: 
            target = torch.from_numpy(target)
        else:
            target = torch.from_numpy(target)
    if mask is not None:
        if not(isinstance(mask, torch.Tensor)):
            if isinstance(mask, list) and len(mask) == 1: 
                mask = torch.from_numpy(mask[0])
            else:
                mask = torch.from_numpy(mask)

    if mask is not None:
        mask = mask.squeeze()
        pred, target = pred[mask > 0], target[mask > 0]

    pred, target = pred.flatten(), target.flatten()
    mae = torch.mean(torch.abs(pred-target))

    return float(mae.item())




def masked_corr(pred : torch.tensor, target : torch.tensor, mask : torch.tensor = None):
    """
    Compute masked correlation between tensors of shape
    [H, W, D]
    
    Params
    ------
    pred : predicted tensor by network
    target : target tensor
    mask : mask with coordinates where to compute PSNR
        (default : None)


    Returns
    -------
    r : correlation metric
    
    """

    if not(isinstance(pred, torch.Tensor)):
        pred = torch.from_numpy(pred)
    if not(isinstance(target, torch.Tensor)):
        target = torch.from_numpy(target)
    if mask is not None:
        if not(isinstance(mask, torch.Tensor)):
            mask = torch.from_numpy(mask)


    if mask is not None:
        mask = mask.squeeze()
        pred, target = pred[mask > 0], target[mask > 0]

    pred, target = pred.flatten(), target.flatten()

    r = torch.tensor([0.])

    if (pred.shape[0] > 0) and (target.shape[0] > 0):
        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        cov = (pred_centered*target_centered).mean()
        pred_std, target_std = pred_centered.std(), target_centered.std()
        r = cov / (pred_std * target_std + np.finfo(float).eps)

    # Avoid NaNs
    r = r.item()
    if np.isnan(r):
        r = 0.0

    return r


def dice_coefficient_3d(pred, target, epsilon=1e-6):
    """
    Computes the Dice coefficient for 3D volumes.
    
    Args:
        pred (torch.Tensor): Predicted tensor (N, C, D, H, W) — one-hot or binarized.
        target (torch.Tensor): Ground truth tensor (same shape as pred).
        epsilon (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Dice score per class (C,)
    """
    assert pred.shape == target.shape, "Shape mismatch between prediction and target"
    
    # Flatten spatial dims
    pred_flat = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
    target_flat = target.contiguous().view(target.shape[0], target.shape[1], -1)

    intersection = (pred_flat * target_flat).sum(-1)  # (N, C)
    union = pred_flat.sum(-1) + target_flat.sum(-1)   # (N, C)

    dice = (2 * intersection + epsilon) / (union + epsilon)  # (N, C)
    return float(dice.mean(dim=0).item())  # mean over batch → (C,)



def dice_coefficient_3d_numpy(pred, target, epsilon=1e-6):
    """
    Computes the Dice coefficient for 3D volumes using NumPy.

    Args:
        pred (np.ndarray): Predicted mask. Shape (D, H, W) or (C, D, H, W)
        target (np.ndarray): Ground truth mask (same shape as pred)
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float or np.ndarray: Dice score. Float for binary, (C,) array for multiclass.
    """
    assert pred.shape == target.shape, "Shape mismatch between prediction and target"

    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat)
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return float(dice)