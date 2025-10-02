"""
Modeltraining.py
----------------
Trains the ASL gesture recognition model (multi-layer perceptron).

Process:
- Loads dataset from /other/asl_dataset.csv (generated using dataset.py).
- Normalizes landmarks and encodes labels.
- Trains an MLP model with TensorFlow/Keras.
- Evaluates model performance with accuracy, confusion matrix, and classification report.
- Saves:
  - 'asl_mlp_model_new.h5' (trained model)
  - 'scaler.pkl' (for normalization)
  - 'label_encoder.pkl' (for mapping labels back to letters)

Usage:
Run once to generate model artifacts needed for Mediapipe.py and GUI.py.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("Sign_Language_Complete/other/asl_dataset.csv")

X = df.drop(columns=['label']).values
y = df['label'].values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

with open("Sign_Language_Complete/models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Scaler saved successfully!")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

with open("Sign_Language_Complete/models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

y_train_cat = keras.utils.to_categorical(y_train, num_classes=len(label_encoder.classes_))
y_test_cat = keras.utils.to_categorical(y_test, num_classes=len(label_encoder.classes_))

model = keras.Sequential([
    keras.layers.Input(shape=(42,)),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(len(label_encoder.classes_), activation="softmax")
])

optimizer = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train_cat, epochs=100, batch_size=16, validation_data=(X_test, y_test_cat))

model.save("asl_mlp_model_new.h5")
print("MLP Model trained and saved successfully!")

loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_classes)
class_names = label_encoder.classes_

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

report = classification_report(y_test, y_pred_classes, target_names=class_names, output_dict=True)
df_report = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Classification Report")
plt.tight_layout()
plt.show()


per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(12, 5))
plt.bar(class_names, per_class_accuracy * 100)
plt.title("Per-Class Accuracy (%)")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
