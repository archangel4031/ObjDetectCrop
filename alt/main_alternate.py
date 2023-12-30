from components_alternate import load_image, detect_objects, draw_boxes_and_crop, model


# Main function
def main(image_path):
    img = load_image(image_path)
    results = detect_objects(model, img)
    cropped_images = draw_boxes_and_crop(results, img)


if __name__ == "__main__":
    image_path = "./examples/example (1).jpg"  # replace with your image path
    main(image_path)
