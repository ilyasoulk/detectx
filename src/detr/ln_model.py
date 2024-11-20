import lightning as L
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import DeTr
from loss import HungarianLoss

class LightningDETR(L.LightningModule):
    def __init__(
        self,
        backbone="resnet50",
        hidden_dim=256,
        input_shape=(3, 512, 512),
        fc_dim=2048,
        num_heads=8,
        activ_fn="relu",
        num_encoder=6,
        num_decoder=6,
        num_obj=100,
        d=256,
        num_cls=21,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_epochs=10,
        max_epochs=300,
        lambda_l1=1.0,
        lambda_iou=1.0
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize DETR model
        self.model = DeTr(
            backbone=backbone,
            hidden_dim=hidden_dim,
            input_shape=input_shape,
            fc_dim=fc_dim,
            num_heads=num_heads,
            activ_fn=activ_fn,
            num_encoder=num_encoder,
            num_decoder=num_decoder,
            num_obj=num_obj,
            d=d,
            num_cls=num_cls
        )
        
        # Initialize loss
        self.criterion = HungarianLoss(lambda_l1=lambda_l1, lambda_iou=lambda_iou)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        pred_classes, pred_boxes = self(images)
        loss = self.criterion((pred_classes, pred_boxes), targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss.mean()

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        pred_classes, pred_boxes = self(images)
        loss = self.criterion((pred_classes, pred_boxes), targets)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss.mean()

    def configure_optimizers(self):
        # Separate backbone parameters for different learning rates
        backbone_params = [p for n, p in self.named_parameters() if "backbone" in n]
        other_params = [p for n, p in self.named_parameters() if "backbone" not in n]

        param_dicts = [
            {"params": backbone_params, "lr": self.hparams.learning_rate * 0.1},
            {"params": other_params, "lr": self.hparams.learning_rate}
        ]

        optimizer = AdamW(
            param_dicts,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.learning_rate * 0.01
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
