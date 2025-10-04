# Facial-emotion-recognition-system
Built a facial emotion recognition system that identified emotions images of human faces. Used a labeled dataset: FER-2013 from Kaagle which contained label images of facial expressions. I preprocessed the images, trained a convolutional neural network (CNN), and evaluate the model's accuracy in classifying emotions. 

# Facial Emotion Recognition using CNN 🎭🤖  

This project implements a **Facial Emotion Recognition (FER) system using a Convolutional Neural Network (CNN) trained on the FER-2013 dataset. The model classifies grayscale facial images (48x48) into seven emotion categories**:  

- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral  

---

## 📂 Dataset  
- Source: FER-2013 (ICML 2013 Challenges in Representation Learning)  
- Classes: 7  
- Structure: `train/`, `val/`, `test/`  
- Images: Grayscale, 48x48 pixels  

---

## ⚙️ Features  
- Data Preprocessing: Normalization, augmentation (rotation, shifts, flips, zoom).  
- Model: Deep CNN with Conv2D, BatchNorm, MaxPooling, Dropout, Dense layers.  
- Training: 50 epochs, Adam optimizer, categorical crossentropy.  
- Evaluation: Classification Report + Confusion Matrix.  
- Accuracy: ~68% test accuracy.  
- Interface: Tkinter GUI to upload an image and predict emotion.  

---

## 📊 Results  
- Training/Validation Curves: Accuracy and Loss plotted.  
- Confusion Matrix: Visual evaluation of class predictions.  
- Classification Report:  
- Precision & recall across 7 classes.  
- Weighted avg accuracy: **68%**.  

---

## 🚀 How to Run  

### 1. Clone Repository  
```bash
git clone https://github.com/your-username/facial-emotion-recognition.git
cd facial-emotion-recognition
2. Install Requirements
bash
Copy code
pip install -r requirements.txt
3. Train Model
bash
Copy code
python building_CNN.py
4. Run Interface (GUI)
bash
Copy code
python interface.py
Upload any face image (JPG/PNG) and get Predicted Emotion instantly.
