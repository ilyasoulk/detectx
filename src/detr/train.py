from ln_model import LightningDETR
from dataset import PascalVOCDataModule
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint


# In your training script
def main():
    data_module = PascalVOCDataModule(
        data_dir="../../data",
        batch_size=2,
        num_workers=4
    )


    checkpoint_callback = ModelCheckpoint(
        monitor='val/total_loss',          # Monitor validation loss
        dirpath='checkpoints',       # Directory to save checkpoints
        filename='detr-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,               # Save the top 3 models
        mode='min',                 # Lower validation loss is better
        save_last=True,             # Additionally save the last model
        verbose=True                # Print information about savings
    )


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
    )

    trainer = L.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=50,
        accelerator='mps', devices=1,
        precision="16-mixed"
    )

    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()
