import torch
import cv2

# Load the YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s.pt")  # default


# Function to load an image
def load_image(img_path):
    img = cv2.imread(img_path)
    return img


# Function to perform object detection
def detect_objects(model, img):
    results = model(img)
    # Results contain bounding boxes, confidences, and class names
    return results


# Function to draw bounding boxes with labels and confidence, and crop the detected objects
def draw_boxes_and_crop(results, img):
    # Get class names
    names = results.names
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    n = len(labels)
    cropped_images = []
    for i in range(n):
        row = cord[i]
        # If you have more than one batch, you'll need to adjust this
        x1, y1, x2, y2, conf = (
            int(row[0] * img.shape[1]),
            int(row[1] * img.shape[0]),
            int(row[2] * img.shape[1]),
            int(row[3] * img.shape[0]),
            row[4],
        )
        if conf > 0.25:  # Confidence threshold
            label = names[int(labels[i])]
            conf_text = f"{label} {conf:.2f}"
            color = (255, 255, 255)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            img = cv2.putText(
                img, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            crop_img = img[y1:y2, x1:x2]
            cropped_images.append((label, conf, crop_img))
            cv2.imshow(f"Cropped image {i}", crop_img)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cropped_images
