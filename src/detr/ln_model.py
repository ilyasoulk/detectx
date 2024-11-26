import lightning as L
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import DeTr
from loss import HungarianLoss, HungarianMatcher
from lightning.pytorch.loggers import WandbLogger
import wandb


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
        project_name="detr",
        run_name=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Add these lines to track running averages
        self.training_step_outputs = []
        self.validation_step_outputs = []

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
        batch_size = images.size(0)

        # Store the loss values
        self.training_step_outputs.append(losses)

        # Log metrics for wandb
        self.log_dict(
            {
                "train/loss_ce": losses["loss_ce"],
                "train/loss_bbox": losses["loss_bbox"],
                "train/loss_giou": losses["loss_giou"],
                "train/total_loss": losses["total_loss"],
            },
            on_step=True,  # Changed to True to see per-step metrics
            on_epoch=True,  # Also aggregate for epoch
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        # Optional: Log learning rate
        opt = self.optimizers()
        if opt is not None:
            current_lr = opt.param_groups[0]["lr"]
            self.log("learning_rate", current_lr, on_step=True, on_epoch=False)

        return losses["total_loss"]

    def on_train_epoch_end(self):
        # Calculate epoch-level metrics
        epoch_losses = {
            "train/epoch_loss_ce": torch.stack(
                [x["loss_ce"] for x in self.training_step_outputs]
            ).mean(),
            "train/epoch_loss_bbox": torch.stack(
                [x["loss_bbox"] for x in self.training_step_outputs]
            ).mean(),
            "train/epoch_loss_giou": torch.stack(
                [x["loss_giou"] for x in self.training_step_outputs]
            ).mean(),
            "train/epoch_total_loss": torch.stack(
                [x["total_loss"] for x in self.training_step_outputs]
            ).mean(),
        }

        # Log epoch-level metrics
        self.log_dict(epoch_losses)

        # Clear the list for next epoch
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        pred_classes, pred_boxes = self(images)
        losses = self.criterion((pred_classes, pred_boxes), targets)
        batch_size = images.size(0)

        # Store validation losses
        self.validation_step_outputs.append(losses)

        # Log with sync_dist=True
        self.log(
            "val/loss_ce",
            losses["loss_ce"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/loss_bbox",
            losses["loss_bbox"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/loss_giou",
            losses["loss_giou"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/total_loss",
            losses["total_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        return losses["total_loss"]

    def on_validation_epoch_end(self):
        # Calculate epoch-level validation metrics
        epoch_losses = {
            "loss_ce": torch.stack(
                [x["loss_ce"] for x in self.validation_step_outputs]
            ).mean(),
            "loss_bbox": torch.stack(
                [x["loss_bbox"] for x in self.validation_step_outputs]
            ).mean(),
            "loss_giou": torch.stack(
                [x["loss_giou"] for x in self.validation_step_outputs]
            ).mean(),
            "total_loss": torch.stack(
                [x["total_loss"] for x in self.validation_step_outputs]
            ).mean(),
        }

        # Log epoch-level metrics
        self.log("val/epoch_loss_ce", epoch_losses["loss_ce"])
        self.log("val/epoch_loss_bbox", epoch_losses["loss_bbox"])
        self.log("val/epoch_loss_giou", epoch_losses["loss_giou"])
        self.log("val/epoch_total_loss", epoch_losses["total_loss"])

        # Clear the list for next epoch
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # Separate backbone parameters for different learning rates
        backbone_params = [p for n, p in self.named_parameters() if "backbone" in n]
        other_params = [p for n, p in self.named_parameters() if "backbone" not in n]

        learning_rate = self.hparams.learning_rate
        backbone_lr = learning_rate * 0.1

        param_dicts = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": other_params, "lr": learning_rate},
        ]

        optimizer = AdamW(param_dicts, weight_decay=self.hparams.weight_decay)

        return {
            "optimizer": optimizer,
        }

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=0.1, gradient_clip_algorithm="norm"
    ):
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )
