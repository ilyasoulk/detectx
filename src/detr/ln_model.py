import lightning as L
from torch.optim import AdamW
from model import DeTr
from loss import HungarianLoss, HungarianMatcher


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

        # Core model components
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

        # Initialize matcher and criterion
        matcher = HungarianMatcher(lambda_l1, lambda_iou, lambda_cls)
        self.criterion = HungarianLoss(matcher, num_classes=num_cls)

    def forward(self, x):
        return self.model(x)

    def _compute_losses(self, batch):
        images, targets = batch
        pred_classes, pred_boxes = self(images)
        return self.criterion((pred_classes, pred_boxes), targets)

    def training_step(self, batch, batch_idx):
        losses = self._compute_losses(batch)

        # Log all losses
        self.log_dict(
            {
                "train/loss_ce": losses["loss_ce"],
                "train/loss_bbox": losses["loss_bbox"],
                "train/loss_iou": losses["loss_iou"],
                "train/total_loss": losses["total_loss"],
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return losses["total_loss"]

    def validation_step(self, batch, batch_idx):
        losses = self._compute_losses(batch)

        # Log all losses
        self.log_dict(
            {
                "val/loss_ce": losses["loss_ce"],
                "val/loss_bbox": losses["loss_bbox"],
                "val/loss_iou": losses["loss_iou"],
                "val/total_loss": losses["total_loss"],
            },
            on_epoch=True,
            prog_bar=True,
        )

        return losses["total_loss"]

    def configure_optimizers(self):
        params = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n],
                "lr": self.hparams.learning_rate * 0.1,
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if "backbone" not in n
                ],
                "lr": self.hparams.learning_rate,
            },
        ]
        return AdamW(params, weight_decay=self.hparams.weight_decay)
