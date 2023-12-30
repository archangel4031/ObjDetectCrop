from grInterface import *
import gradio as gr
import glob


def main():
    # List all JPEG and PNG images in the ./examples directory
    image_paths = glob.glob("./examples/*.jpg") + glob.glob("./examples/*.png")

    # Define Gradio interface
    iface = gr.Interface(
        fn=process_and_show,
        inputs=[
            gr.inputs.Image(type="pil"),
            gr.inputs.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                default=0.25,
                label="Confidence Threshold",
            ),
        ],
        outputs=[
            gr.outputs.Image(type="pil", label="Processed Image"),
            gr.outputs.components.Gallery(
                label="Cropped Images", type="pil", show_flags=False
            ).style(height="400px", grid=2),
        ],
        title="Object Detection",
        description="Identify objects in an image using YOLOv5",
        examples=[[path] for path in image_paths],  # Optional: Provide example images
        allow_flagging="never",
        theme="gradio/monochrome",
    )

    # Launch the Gradio app
    iface.launch(show_api=False)


if __name__ == "__main__":
    main()
