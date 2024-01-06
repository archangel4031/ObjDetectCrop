import torch
import cv2
import random
import tkinter as tk
from tkinter import messagebox


def ask_to_view_cropped_images():
    show_cropped = True
    # Create a root window and hide it
    root = tk.Tk()
    root.withdraw()

    # Ask user if they want to view the cropped images
    response = messagebox.askyesno(
        "View Cropped Images",
        "Do you want to view cropped images in a separate window?",
    )

    # The response will be True if the user clicked 'Yes' and False otherwise
    if response:
        print("User chose to view cropped images.")
        # Add your logic here to open the images in a separate window
        show_cropped = True
    else:
        print("User chose not to view cropped images.")
        # Add your logic here to not open the images in a separate window
        show_cropped = False

    # Destroy the root window after the dialog is closed
    root.destroy()

    return show_cropped


# Load the YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s.pt")  # default


def generate_random_color():
    # Generate three random numbers between 0 and 255
    r = random.randint(70, 255)
    g = random.randint(70, 255)
    b = random.randint(70, 255)

    # Return the color as a tuple
    return (r, g, b)


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
def draw_boxes_and_crop(results, img, conf_threshold=0.25):
    show_cropped = ask_to_view_cropped_images()
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
        if conf > conf_threshold:  # Confidence threshold
            label = names[int(labels[i])]
            conf_text = f"{label} {conf:.2f}"
            color = generate_random_color()
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            img = cv2.putText(
                img, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            crop_img = img[y1:y2, x1:x2]
            cropped_images.append((label, conf, crop_img))
            if show_cropped:
                cv2.imshow(f"Cropped image {i} - {label}: {conf:.2f}", crop_img)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cropped_images


def get_predictions(results, conf_threshold=0.25):
    names = results.names
    preds = []
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    n = len(labels)
    for i in range(n):
        row = cord[i]
        # If you have more than one batch, you'll need to adjust this
        conf = row[4]
        if conf > conf_threshold:  # Confidence threshold
            label = names[int(labels[i])]
            preds.append((label, conf))
    return preds
