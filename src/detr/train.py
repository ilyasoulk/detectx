from ln_model import LightningDETR
from dataset import PascalVOCDataModule
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import wandb


# In your training script
def main():
    # Initialize wandb first
    wandb.init(
        project="detr",
        name="experiment_1",
        config={
            "architecture": "DETR",
            "backbone": "resnet18",
            "dataset": "PASCAL VOC",
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 5e-5,
        },
    )

    data_module = PascalVOCDataModule(
        data_dir="../../data", batch_size=8, num_workers=4
    )

    wandb_logger = WandbLogger(
        project="detr",
        log_model=True,
        save_dir="logs/",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/total_loss",  # Monitor validation loss
        dirpath="checkpoints",  # Directory to save checkpoints
        filename="detr-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,  # Save the top 3 models
        mode="min",  # Lower validation loss is better
        save_last=True,  # Additionally save the last model
        verbose=True,  # Print information about savings
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    model = LightningDETR(
        backbone="resnet18",
        hidden_dim=128,
        fc_dim=512,
        num_heads=8,
        activ_fn="gelu",
        num_encoder=3,
        num_decoder=3,
        num_obj=30,
        num_cls=21,
        learning_rate=1e-3,
    )

    trainer = L.Trainer(
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=50,
        accelerator="mps",
        devices=1,
        precision="16-mixed",
        # gradient_clip_val=0.1,
        # gradient_clip_algorithm="norm",
        logger=wandb_logger,
        log_every_n_steps=1,  # Add this to control logging frequency
    )

    wandb_logger.watch(model, log="all", log_freq=100)

    trainer.fit(model, data_module)

    wandb.finish()


if __name__ == "__main__":
    main()
