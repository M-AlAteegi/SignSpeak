![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Mediapipe](https://img.shields.io/badge/Mediapipe-CV-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# ✋ SignSpeak

**SignSpeak** is a real-time **Sign Language Recognition** system that combines **Computer Vision (Mediapipe)**, **Deep Learning (TensorFlow/Keras)**, and **Natural Language Processing (N-grams)** with a user-friendly **Tkinter GUI**.  

It captures hand gestures through a webcam, classifies them into letters, builds words and sentences, and intelligently suggests the **next word** using an N-gram language model — bridging the gap between signers and non-signers.

---
## 🛠 System Overview
The system captures hand gestures using **MediaPipe**, classifies them with a trained **MLP model**, and enhances fluency with an **N-gram language model**.  
Users interact with the system through a **Tkinter GUI** for real-time translation.

![System Flow](assets/system_flow.png)

---

## 🚀 Features
- **Real-time gesture recognition** using Mediapipe hand landmarks.  
- **Deep Learning classifier** (MLP model) trained on ASL alphabet.  
- **N-gram language model** for next-word prediction (trained on 4M+ Twitter dataset).  
- **Interactive Tkinter GUI** with:
  - Live video feed with letter overlays.  
  - Word and sentence builder.  
  - Top-3 intelligent word suggestions with one-click insertion.  
  - ASL reference chart for beginners.  
- **Robust error handling** (missing models, missing datasets, webcam access).  
- Modular codebase with separate components for:
  - Data collection (`dataset.py`)  
  - Model training (`Modeltraining.py`)  
  - N-gram training & evaluation (`TrainingNGrams.py`, `NGramsPlots.py`)  
  - GUI (`GUI.py`)  

---

## 📂 Folder Structure
SignSpeak/ <br>
│── assets/ # Images & static assets (ASL guide, etc.) <br>
│ └── ASL_image.png <br>
│ <br>
│── models/ # (empty on GitHub, filled locally when user trains) <br>
│ <br>
│── other/ # Supporting scripts (training, testing, dataset collection) <br>
│ └── asl_dataset.csv <br>
│ └── dataset.py <br>
│ └── Modeltraining.py <br>
│ └── TrainingNGrams.py <br>
│ <br>
│── utils/ # Utility classes & helpers <br>
│ └── NgramAutoComplete.py <br>
│ <br>
│── GUI.py # Main application with Tkinter GUI <br>
│── Mediapipe.py # Hand detection + ASL model prediction <br>
│── requirements.txt # Dependencies <br>
│── .gitignore # Ignored files (large models, checkpoints, etc.) <br>

---

## ⚙️ Installation


### 1. Clone the repository
```bash
git clone https://github.com/M-AlAteegi/SignSpeak.git
cd SignSpeak
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset

The project uses the SwiftKey dataset (Twitter subset) for N-gram training. <br> 
📥 [Download from Kaggle: Tweets-Blogs-News Dataset (4M)](https://www.kaggle.com/datasets/crmercado/tweets-blogs-news-swiftkey-dataset-4million?resource=download)

Place en_US.twitter.txt inside the project root folder. <br> <br>

---

## ▶️ Usage
### 1. Train ASL Recognition Model
python other/`Modeltraining.py`


This generates:

- asl_mlp_model_new.h5

- label_encoder.pkl

- scaler.pkl <br>

### 2. Train N-gram Model
python other/`TrainingNGrams.py`


This generates:

- en_counts.pkl

- vocab.pkl <br>

### 3. Launch the Application
python `GUI.py`


✅ The GUI will open with:

- Live webcam feed.

- Recognized letters, words, and sentence builder.

- Top-3 word suggestions with confidence scores.

- An ASL chart for quick reference. <br> <br>

---

## 📊 Example Workflow

1. Show ASL signs → letters appear live.

2. Press space → word is committed.

3. Suggestions box shows top-3 likely next words.

4. Click on a suggestion → inserted into the sentence.

5. Build full sentences naturally, assisted by NLP. <br> <br>

---

## 🔮 Future Improvements

- Add support for dynamic ASL gestures (not just static letters).

- Replace N-grams with a Transformer-based LM (Hugging Face integration).

- Export final sentences to speech (TTS) for accessibility.

- Mobile app version for broader use. <br> <br>

---

## 📜 License <br>
This project is licensed under the [MIT License](LICENSE) – free to use and modify. <br> <br>

---

## 👨‍💻 Author <br>
Developed by **Mohammed AlAteegi** <br>
📧 m7mdateegi@gmail.com
