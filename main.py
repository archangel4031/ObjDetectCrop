from components import (
    load_image,
    detect_objects,
    draw_boxes_and_crop,
    model,
    get_predictions,
)
import argparse
import os
import requests
from urllib.parse import urlparse
import tkinter as tk
from tkinter import messagebox


def download_image(image_url, download_path):
    # Parse the name of the image from the URL
    parsed_url = urlparse(image_url)
    image_name = os.path.basename(parsed_url.path)
    # Create the downloads directory if it does not exist
    os.makedirs(download_path, exist_ok=True)
    # Download the image
    response = requests.get(image_url)
    if response.status_code == 200:
        image_path = os.path.join(download_path, image_name)
        with open(image_path, "wb") as f:
            f.write(response.content)
        return image_path
    else:
        raise ValueError(
            f"Failed to download the image. Status code: {response.status_code}"
        )


def check_confidence(value):
    fvalue = float(value)
    if fvalue < 0 or fvalue > 1:
        raise argparse.ArgumentTypeError(
            "Confidence threshold must be a floating-point number between 0 and 1"
        )
    return fvalue


def check_image_path_or_url(path_or_url):
    if os.path.isfile(path_or_url):
        return path_or_url
    elif path_or_url.startswith(("http://", "https://")):
        # Assume it is a URL and try to download the image
        try:
            return download_image(path_or_url, "./downloads")
        except Exception as e:
            raise argparse.ArgumentTypeError(
                f"Could not download the image from the URL provided: {e}"
            )
    else:
        raise argparse.ArgumentTypeError(
            f"The provided path '{path_or_url}' is neither a valid file nor a URL."
        )


# Main function
def main(image_path, conf_threshold=0.25):
    # Your main function that processes the image
    print(f"Processing image: {image_path} with confidence threshold: {conf_threshold}")
    img = load_image(image_path)
    results = detect_objects(model, img)
    cropped_images = draw_boxes_and_crop(results, img, conf_threshold)


def test(image_path="./examples/example (1).jpg", conf_threshold=0.25):
    main(image_path, conf_threshold)


def prediction(image_path, conf_threshold=0.25):
    img = load_image(image_path)
    results = detect_objects(model, img)
    preds = get_predictions(results, conf_threshold)
    # Format the results into a string message
    message = "\n".join(
        [f"Label: {label}, Confidence: {confidence:.2f}" for label, confidence in preds]
    )
    # Initialize Tkinter root
    root = tk.Tk()
    root.withdraw()  # We don't need a full GUI, so keep the root window from appearing

    # Show the message box with the results
    messagebox.showinfo("Image Processing Results", message)

    # Destroy the root after the message box is closed
    root.destroy()


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Process an image or download from a URL with a confidence threshold."
    )
    parser.add_argument(
        "-i",
        "--image",
        type=check_image_path_or_url,
        default="examples\example (1).jpg",
        help='Path to the image file or URL to download the image. Defaults to "examples\example (1).jpg"',
    )
    parser.add_argument(
        "-c",
        "--confidence",
        type=check_confidence,
        default=0.25,
        help="Confidence threshold value between 0 and 1. Defaults to 0.25",
    )
    parser.add_argument(
        "-p",
        "--predict",
        action="store_true",
        help="Run prediction function instead of test.",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    print("Press any key to exit (while image window is focused)...")

    # Conditionally run the test or prediction function based on the presence of the -p flag
    if args.predict:
        prediction(args.image, args.confidence)
    else:
        test(args.image, args.confidence)
