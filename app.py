import PIL.Image as Image
import gradio as gr
import cv2
import numpy as np

from ultralytics import ASSETS, YOLO

model = YOLO("best.pt")

def preprocess_image(img):
    # Convert to grayscale
    img_gray = img.convert("L")
    
    img_resized = img_gray.resize((640,640))
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(np.array(img_resized))
    
    
    return Image.fromarray(img_clahe)

def predict_image(img, conf_threshold, iou_threshold):
    # Preprocess the image
    img_preprocessed = preprocess_image(img)
    
    results = model.predict(
        source=img_preprocessed,
        conf=conf_threshold,
        iou=0.45,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im

iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.7, label="Confidence threshold"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    allow_flagging="never",
    description="""<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Ultralytics Gradio</title>
</head>
<body>
    <center>
        <div class="jumbotron jumbotron-fluid" style="background-image:url('/file=Jumbotron.png')">
            <div class="container-fluid">
                <img src="/file=logo.png" alt="" style="height: 40px;">
                <p style="color: white; text-align: center; font-size: 50px; font-weight: bold; font-family: Verdana, Geneva, Tahoma, sans-serif;">Anoa-Detection</p>
                <p style="color: white; text-align: center; font-size: 30px; font-family: Verdana, Geneva, Tahoma, sans-serif">Identifikasi Individu Anoa di Taman Nasional Bogani Nani Wartabone</p>
            </div>
        </div>
    </center>
</body>
</html>""",
)

if __name__ == '__main__':
    iface.launch(share=True, allowed_paths=['jumbotron.png','logo.png'])
