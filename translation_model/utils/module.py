import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from dynamic_network_architectures.architectures.unet import PlainConvUNet

class DualLossUNet(pl.LightningModule):
    def __init__(self, arch_kwargs, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model = PlainConvUNet(**arch_kwargs)
        self.cfg = cfg
        self.lr = cfg["initial_lr"]

        # Loss functions
        self.loss_dist = nn.L1Loss()
        self.loss_tta = nn.L1Loss()  # replace with PSNR or custom if needed

    def forward(self, x):
        return self.model(x)

    def compute_dual_loss(self, preds : torch.tensor, dist : torch.tensor, t : torch.tensor, brain_mask : torch.tensor, vessel_mask : torch.tensor):
        """
        Compute MAE loss between predicted and ground truth
        distance and time maps

        Params
        ------
        preds : predicted maps
        dist : distance maps
        t : time maps
        brain_mask : brain mask
        vessel_mask : vessel mask

        Returns
        -------
        Combined and individual losses
        
        """
        pred_dist, pred_tta = preds[:, 0], preds[:, 1]

        # Apply masks
        loss_dist = self.loss_dist(pred_dist[brain_mask], dist[brain_mask])
        loss_tta = self.loss_tta(pred_tta[vessel_mask], t[vessel_mask])

        return loss_dist + loss_tta, loss_dist, loss_tta
    


    def training_step(self, batch, batch_idx):
        x, dist, t, brain_mask, vessel_mask = batch
        preds = self(x)
        loss, loss_d, loss_t = self.compute_dual_loss(preds, dist, t, brain_mask, vessel_mask)
        self.log("train/loss", loss)
        self.log("train/loss distance", loss_d)
        self.log("train/loss time", loss_t)
        return loss


    def validation_step(self, batch, batch_idx):
        x, dist, t, brain_mask, vessel_mask = batch
        preds = self(x)
        loss, loss_d, loss_t = self.compute_dual_loss(preds, dist, t, brain_mask, vessel_mask)
        psnr_dist = -10 * torch.log10(F.mse_loss(preds[:,0][brain_mask], t[brain_mask]))
        psnr_time = -10 * torch.log10(F.mse_loss(preds[:,1][vessel_mask], t[vessel_mask]))
        psnr = (psnr_dist + psnr_time)*0.5
        self.log_dict({
            "val/loss": loss,
            "val/loss distance" : loss_d,
            "val/loss time" : loss_t,
            "val/psnr dist": psnr_dist,
            "val/psnr time": psnr_time,
            "val/psnr" : psnr
        }, prog_bar=True)
        return psnr

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), 
                                    lr=self.lr, 
                                    momentum=self.cfg["sgd_momentum"],
                                    nesterov=bool(self.cfg["sgd_nesterov"]),
                                    weight_decay=self.cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            patience=self.cfg["scheduler_patience"], 
            factor=self.cfg["scheduler_factor"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/psnr",
                "interval": "epoch",
                "frequency": 1
            }
        }