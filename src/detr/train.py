from engine import build, train_one_epoch, evaluate

from dataset_voc import MyVOCDetection, collate_fn
from torch.utils.data import DataLoader
import torch


def main():
    weight_dict = {"ce": 5, "bbox": 2, "giou": 1}
    batch_size = 4
    device = "mps"
    model, criterion, optimizer, scheduler = build(
        weight_dict,
        backbone="resnet18",
        hidden_dim=128,
        num_heads=4,
        num_encoder=2,
        num_decoder=2,
        num_cls=92,
        learning_rate=1e-4,
        weight_decay=1e-4,
    )
    model = model.to(device)
    criterion = criterion.to(device)
    print(sum([param.numel() for param in model.parameters()]))

    # Initialize best mAP tracking instead of best loss
    best_map = 0.0  # Start from 0 since mAP ranges from 0 to 1

    train_ds = MyVOCDetection(
        root="./data_voc", year="2012", image_set="train", download=True
    )

    val_ds = MyVOCDetection(
        root="./data_voc", year="2012", image_set="val", download=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    epochs = 1000
    val_results = []
    train_results = []
    for epoch in range(epochs):
        train_loss, train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, max_norm=0.1
        )
        scheduler.step()

        train_results.append(train_metrics)

        val_loss, val_metrics = evaluate(model, criterion, val_loader, device)
        val_results.append(val_metrics)
        # Step the scheduler with validation mAP

        print(f"\nEpoch {epoch}")
        print(f"mAP train : {train_metrics}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"mAP val : {val_metrics}")
        print(f"Validation Loss: {val_loss:.4f}")

        # Save if this is the best model based on validation mAP
        current_map = val_metrics["mAP"]
        if current_map > best_map:
            best_map = current_map
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "best_map": best_map,
            }
            torch.save(checkpoint, "best_model.pth")
            print(f"Saved new best model with validation mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()
