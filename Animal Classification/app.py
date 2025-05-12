import tensorflow as tf
from tensorflow import keras
import gradio as gr

saved_model = keras.models.load_model('animal_classifier_model.h5')

# Class names (ensure these match your dataset classes)
class_names = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 
                'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion', 
                'Panda', 'Tiger', 'Zebra']
def predict(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, 0)

    prediction = saved_model.predict(img_array)
    predicted_class = class_names[tf.argmax(prediction[0])]
    confidence = round(100 * tf.reduce_max(prediction).numpy(), 2)

    return f"Predicted: {predicted_class} (Confidence: {confidence}%)"

# Gradio Interface
gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs='text',
    title="🐾 Animal Classifier - Identify Your Wildlife Companion! 🐾"
).launch(share=True)
