from torchvision import transforms as T
import cv2  # OpenCV for webcam capture
import torch
from dataset_voc import CLASSES
from torchvision import ops
import torch.nn.functional as F
from engine import build
from dataset_voc import revert_normalization


def preprocess_frame(frame_rgb, edge=512):
    """
    Preprocess a webcam frame to match training preprocessing exactly.
    Returns preprocessed tensor and scale factors for bbox adjustment
    """
    h, w = frame_rgb.shape[:2]

    # Store original dimensions for scaling
    scale_w = w / edge
    scale_h = h / edge

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize(
                (edge, edge), antialias=True
            ),  # Direct resize to square like in training
        ]
    )

    tensor_frame = transform(frame_rgb)

    return tensor_frame, (scale_w, scale_h)


def visualize_predictions_webcam(model, device, criterion, conf_threshold=0.9):
    model = model.to(device)
    model.eval()

    # Initialize webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with torch.no_grad():
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Convert frame (OpenCV BGR to RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height, frame_width = frame.shape[:2]

            # Calculate aspect ratio parameters
            if frame_height > frame_width:
                scale_factor = frame_width / frame_height
                effective_width = int(512 * scale_factor)
                effective_height = 512
                offset_x = (512 - effective_width) // 2
                offset_y = 0
            else:
                scale_factor = frame_height / frame_width
                effective_height = int(512 * scale_factor)
                effective_width = 512
                offset_x = 0
                offset_y = (512 - effective_height) // 2

            # Preprocess frame
            image, (scale_w, scale_h) = preprocess_frame(frame_rgb)
            image = image.to(device).unsqueeze(0)

            # Get model predictions
            outputs = model(image)
            classes_pred, boxes_pred = outputs

            # Move predictions to CPU for further processing
            classes_pred = classes_pred.cpu()
            boxes_pred = boxes_pred.cpu()

            # Process predictions
            probs = F.softmax(classes_pred[0], dim=-1)
            boxes = boxes_pred[0]
            boxes_xyxy = ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

            # Filter predictions
            max_probs, pred_classes = probs.max(dim=1)
            keep = (max_probs > conf_threshold) & (pred_classes != (len(CLASSES) - 1))
            filtered_boxes = boxes_xyxy[keep]
            filtered_probs = probs[keep]
            filtered_classes = pred_classes[keep]

            # Draw predictions on the frame
            for box, prob, cls in zip(filtered_boxes, filtered_probs, filtered_classes):
                # Get coordinates (already in [0,1] range)
                x1, y1, x2, y2 = box.numpy()

                # Scale directly to frame dimensions
                x1 = int(x1 * frame_width)
                x2 = int(x2 * frame_width)
                y1 = int(y1 * frame_height)
                y2 = int(y2 * frame_height)

                # Ensure coordinates are within frame boundaries
                x1 = max(0, min(x1, frame_width))
                x2 = max(0, min(x2, frame_width))
                y1 = max(0, min(y1, frame_height))
                y2 = max(0, min(y2, frame_height))

                # Only draw if box has valid dimensions
                if x2 > x1 and y2 > y1:
                    label = f"{CLASSES[cls]}: {prob.max().item():.2f}"

                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Add label background for better visibility
                    label_size = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )[0]
                    cv2.rectangle(
                        frame,
                        (x1, max(y1 - label_size[1] - 10, 0)),
                        (x1 + label_size[0], max(y1, label_size[1])),
                        (0, 255, 0),
                        -1,
                    )

                    # Draw text
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),  # Black text on green background
                        2,
                    )

            # Display the frame with bounding boxes
            cv2.imshow("Webcam - Object Detection", frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    weight_dict = {"ce": 5, "bbox": 2, "giou": 1}
    device = "mps"
    model, criterion, _, _ = build(
        weight_dict,
        backbone="resnet18",
        hidden_dim=128,
        num_heads=4,
        num_encoder=2,
        num_decoder=2,
        num_cls=22,
    )

    checkpoint = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    visualize_predictions_webcam(model, device, criterion)
