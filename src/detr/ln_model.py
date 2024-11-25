import lightning as L
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import DeTr
from loss import HungarianLoss, HungarianMatcher


class LightningDETR(L.LightningModule):
    def __init__(
        self,
        backbone="resnet18",
        hidden_dim=128,
        input_shape=(3, 512, 512),
        fc_dim=512,
        num_heads=8,
        activ_fn="relu",
        num_encoder=2,
        num_decoder=2,
        num_obj=30,
        num_cls=21,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_epochs=10,
        max_epochs=50,
        lambda_l1=5.0,
        lambda_iou=2.0,
        lambda_cls=1.0,
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
            num_cls=num_cls,
        )

        # Initialize loss
        matcher = HungarianMatcher(lambda_l1, lambda_iou, lambda_cls)
        self.criterion = HungarianLoss(matcher, num_classes=num_cls)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        pred_classes, pred_boxes = self(images)
        losses = self.criterion((pred_classes, pred_boxes), targets)

        # Log all losses - now using the new loss dictionary structure
        self.log(
            "train/loss_ce",
            losses["loss_ce"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/loss_bbox",
            losses["loss_bbox"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train/loss_giou",
            losses["loss_giou"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/total_loss",
            losses["total_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return losses["total_loss"]

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        pred_classes, pred_boxes = self(images)
        losses = self.criterion((pred_classes, pred_boxes), targets)

        # Log all losses
        self.log("val/loss_ce", losses["loss_ce"], on_epoch=True, prog_bar=False)
        self.log("val/loss_bbox", losses["loss_bbox"], on_epoch=True, prog_bar=False)
        self.log("val/loss_giou", losses["loss_giou"], on_epoch=True, prog_bar=False)
        self.log("val/total_loss", losses["total_loss"], on_epoch=True, prog_bar=True)

        return losses["total_loss"]

    def configure_optimizers(self):
        # Separate backbone parameters for different learning rates
        backbone_params = [p for n, p in self.named_parameters() if "backbone" in n]
        other_params = [p for n, p in self.named_parameters() if "backbone" not in n]

        learning_rate = self.hparams.learning_rate * 0.1
        backbone_lr = learning_rate * 0.1

        param_dicts = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": other_params, "lr": learning_rate},
        ]

        optimizer = AdamW(param_dicts, weight_decay=self.hparams.weight_decay)

        return {
            "optimizer": optimizer,
        }
