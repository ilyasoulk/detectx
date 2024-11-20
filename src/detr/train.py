from ln_model import LightningDETR
from dataset import PascalVOCDataModule
import lightning as L

# In your training script
def main():
    data_module = PascalVOCDataModule(
        data_dir="../../data",
        batch_size=2,
        num_workers=4
    )

    model = LightningDETR()

    trainer = L.Trainer(
        max_epochs=300,
        accelerator='mps', devices=1,
        precision="16-mixed"
    )

    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()
