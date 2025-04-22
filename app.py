import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

@st.cache_resource
def load_model():
    try:
        # Try loading with absolute path first
        model_path = 'models/quantized_distilled_model.tflite'
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except ValueError as e:
        st.error(f"Model loading failed: {str(e)}")
        st.error("Possible solutions:")
        st.error("1. Ensure model file exists at correct path")
        st.error("2. Check TensorFlow version matches model version")
        st.error("3. Try re-converting your model with TensorFlow 2.15.0")
        st.stop()
