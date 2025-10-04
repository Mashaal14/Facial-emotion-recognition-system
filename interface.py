import os
import numpy as np
import tkinter as tk
from tkinter import Label, Button, filedialog
from PIL import Image, ImageTk
import tensorflow as tf

# -----------------------
# Load the trained model
# -----------------------
model = tf.keras.models.load_model("fer2013_cnn_model.keras")

# Class labels (same order as training)
class_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# -----------------------
# GUI Application
# -----------------------
root = tk.Tk()
root.title("FER-2013 Emotion Recognition")
root.geometry("600x500")
root.configure(bg="#f4f6f9")

# Title
title_label = Label(root, text="Emotion Recognition (FER-2013)", 
                    font=("Helvetica", 18, "bold"), bg="#f4f6f9", fg="#333")
title_label.pack(pady=20)

# Image display
image_label = Label(root, bg="#f4f6f9")
image_label.pack(pady=20)

# Prediction result
result_label = Label(root, text="", font=("Helvetica", 16, "bold"), bg="#f4f6f9")
result_label.pack(pady=10)


def browse_image():
    """Browse image from computer and display prediction"""
    global img_display

    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:  # If no file selected
        return

    # Load image
    img = Image.open(file_path).convert("L")  # grayscale
    img_resized = img.resize((200, 200))
    img_display = ImageTk.PhotoImage(img_resized)

    # Show image
    image_label.config(image=img_display)

    # Preprocess for model
    img_array = img.resize((48, 48))
    img_array = np.array(img_array).reshape(1, 48, 48, 1) / 255.0

    # Predict
    prediction = model.predict(img_array, verbose=0)
    predicted_class = class_labels[np.argmax(prediction)]

    # Show result
    result_label.config(text=f"Predicted Emotion: {predicted_class}", fg="#2c3e50")


# Button
browse_button = Button(
    root,
    text="Browse Image",
    font=("Helvetica", 14),
    bg="#3498db",
    fg="white",
    activebackground="#2980b9",
    activeforeground="white",
    relief="flat",
    padx=10,
    pady=5,
    command=browse_image
)
browse_button.pack(pady=20)

# Run App
root.mainloop()
