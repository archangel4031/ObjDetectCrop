import torch
from PIL import Image, ImageDraw, ImageFont
import random

# Load the YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s.pt")  # default


# Function to load an image
def load_image(image_path):
    return Image.open(image_path)


def generate_random_color():
    # Generate three random numbers between 0 and 255
    r = random.randint(70, 255)
    g = random.randint(70, 255)
    b = random.randint(70, 255)

    # Return the color as a tuple
    return (r, g, b)


# Function to perform object detection
def detect_objects(model, img):
    results = model(img)
    # Results contain bounding boxes, confidences, and class names
    return results


# Function to draw bounding boxes with labels and confidence, and crop the detected objects
def draw_boxes_and_crop(results, pil_img, conf_threshold=0.25):
    draw = ImageDraw.Draw(pil_img)
    names = results.names
    predictions = results.pred[0]
    boxes = predictions[:, :4]
    score = predictions[:, 4]
    categories = predictions[:, 5]
    cropped_images = []

    # Extract bounding box coordinates, label, and confidence from result
    # x1, y1, x2, y2, label, conf = ...  # Implement according to your result format
    objects_detected = len(predictions)
    for p in range(objects_detected):
        x1 = boxes[p][0]
        y1 = boxes[p][1]
        x2 = boxes[p][2]
        y2 = boxes[p][3]

        label = names[int(categories[p])]
        conf = score[p]

        if conf > conf_threshold:  # Confidence threshold
            color = generate_random_color()
            # Draw the rectangle on the image
            draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=2)
            # Optionally draw label and confidence
            font = ImageFont.truetype("arial.ttf", 24)
            draw.text((x1, y1), f"{label} {conf:.2f}", fill=color, font=font)

            # # Crop the image
            # Assuming 'coords' is a PyTorch tensor with the crop coordinates
            coords = torch.tensor([x1, y1, x2, y2])

            # Convert tensor to a list of Python integers
            coords = [int(coord.item()) for coord in coords]
            crop_img = pil_img.crop(coords)
            cropped_images.append((label, conf, crop_img))

    return pil_img, cropped_images


def process_and_show(image, conf_threshold=0.25):
    # Detect objects using YOLO
    results = detect_objects(model, image)
    # Draw bounding boxes and crop the detected objects
    pil_img, cropped_images = draw_boxes_and_crop(results, image, conf_threshold)
    # Extract PIL Image for Gradio Output
    pil_cropped_images = [pil_img for _, _, pil_img in cropped_images]
    return pil_img, pil_cropped_images
