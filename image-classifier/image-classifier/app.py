import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image


model = tf.keras.models.load_model('meu_modelo.h5')

def predict_image(img):
    
    img = np.array(img)

    
    img = tf.image.resize(img, (224, 224))

    # MobileNetV2:
    
    img = img / 127.5 - 1

    
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    
    if prediction < 0.5:
        result = {"ai": float(1 - prediction[0][0]), "human": float(prediction[0][0])}
    else:
        result = {"human": float(prediction[0][0]), "ai": float(1 - prediction[0][0])}

    return result


exemplos = [
    'vangoghai.jpg',
    'vangoghhuman.jpg'
]

#gradio
image_input = gr.Image()
label_output = gr.Label()

# Gradio Interface
interface = gr.Interface(
    fn=predict_image,
    inputs=image_input,
    outputs=label_output,
    examples=exemplos,
    title="Image-Classifier-AIvsHuman",
    description="Upload an image and the output will tell you whether it's potentially AI-generated or human-generated."
)

interface.launch(debug=True)


