"""
GUI.py
------
Main application file for the Sign Language Sentence Builder.

Features:
- Captures webcam feed and detects ASL hand gestures (via Mediapipe + trained MLP model).
- Converts detected gestures into letters and builds words in real-time.
- Uses an N-gram language model to suggest next words with confidence scores.
- Provides a simple Tkinter GUI with live video, recognized text, and suggestions.
- Displays an ASL alphabet reference image for beginners.

Requirements:
- Requires trained N-gram model (run TrainingNGrams.py first).
- Requires trained ASL model (run Modeltraining.py first).
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import time
import os

from Mediapipe import detect_and_draw
from utils.NgramAutoComplete import NgramAutocomplete

current_word = ""
full_sentence = []
suggested_word = ""
last_letter_time = 0
last_detected_letter = None
stable_letter_start_time = None
top_suggestions = [("", 0.0)] * 3
suggestion_buttons = []

model = NgramAutocomplete()
try:
    model.load_model(
        ngram_path='models/en_counts.pkl',
        vocab_path='models/vocab.pkl'
    )
except FileNotFoundError:
    raise RuntimeError(
        "N-gram model not found. Please run TrainingNGrams.py first to generate en_counts.pkl and vocab.pkl."
    )

window = tk.Tk()
window.title("Sign Language Sentence Builder")
window.geometry("1180x830")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("❌ Could not access webcam. Please check your camera connection.")

main_frame = ttk.Frame(window, padding=10)
main_frame.pack(fill="both", expand=True)

left_frame = ttk.Frame(main_frame)
left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

right_frame = ttk.Frame(main_frame)
right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

video_label = ttk.Label(left_frame)
video_label.pack()

# ─── Sentence Display Box ──────────────────────────────────
text_box = ttk.LabelFrame(left_frame, text="📝 Live Recognition", padding=10)
text_box.pack(pady=(10, 10), fill="x")

word_var = tk.StringVar()
sentence_var = tk.StringVar()

ttk.Label(text_box, text="🖐️ Signed Letters:").pack(anchor="w")
ttk.Label(text_box, textvariable=word_var, font=("Helvetica", 18)).pack(anchor="w", pady=(0, 10))

ttk.Label(text_box, text="📄 Sentence:").pack(anchor="w")
ttk.Label(text_box, textvariable=sentence_var, font=("Helvetica", 18)).pack(anchor="w")

# ─── Suggestions Box ───────────────────────────────────────
suggestion_frame = ttk.LabelFrame(right_frame, text="💡 Suggested Next Words", padding=10)
suggestion_frame.pack(fill="x", pady=10)

button_font = ("Helvetica", 20)
button_bg = "#F0F0F0"
button_fg = "#222222"
button_border = 3
suggestion_emojis = ["🥇", "🥈", "🥉"]

def on_hover(e, btn):
    btn.config(relief="sunken", bd=4)

def on_leave(e, btn):
    btn.config(relief="raised", bd=button_border)

suggestion_buttons.clear()
for i in range(3):
    btn = tk.Button(suggestion_frame, text="💡 Suggestion", font=button_font,
                    bg=button_bg, fg=button_fg, relief="raised", borderwidth=button_border,
                    command=lambda: None)
    btn.pack(fill="x", pady=5, ipadx=20, ipady=10)
    btn.bind("<Enter>", lambda e, b=btn: on_hover(e, b))
    btn.bind("<Leave>", lambda e, b=btn: on_leave(e, b))
    suggestion_buttons.append(btn)

def update_text():
    global top_suggestions
    word_var.set(current_word if current_word else "Waiting...")
    sentence_var.set(" ".join(full_sentence))

    for i, (word, prob) in enumerate(top_suggestions):
        percent = int(prob * 100)
        label = f"{suggestion_emojis[i]} {word} ({percent}%)"
        suggestion_buttons[i].config(
            text=label,
            command=lambda w=word: accept_suggestion_choice(w)
        )

def accept_suggestion_choice(choice_word):
    global full_sentence, suggested_word, top_suggestions

    if choice_word:
        choice = messagebox.askquestion("Suggestion Options",
            f"Suggestion: '{choice_word}'\n\nConcatenate or insert as new word?\n"
            "Yes = Concatenate\nNo = Insert as new word")

        if choice == "yes":
            if full_sentence:
                full_sentence[-1] += choice_word
            else:
                full_sentence.append(choice_word)
        else:
            full_sentence.append(choice_word)

        # 🆕 Regenerate suggestions based on the new context
        context = full_sentence[-3:]
        top_suggestions = model.suggest_top_k(context, top_k_vocab=1000, num_suggestions=3)
        suggested_word = top_suggestions[0][0] if top_suggestions else ""

        update_text()

def dismiss_suggestions():
    global full_sentence, suggested_word, top_suggestions

    context = full_sentence[-3:]
    old_words = set(word for word, _ in top_suggestions)

    # Get more suggestions than needed
    all_suggestions = model.suggest_top_k(context, top_k_vocab=1000, num_suggestions=10)

    # Filter out current suggestions
    new_suggestions = [(w, p) for w, p in all_suggestions if w not in old_words]

    # Pad if not enough new suggestions
    if len(new_suggestions) < 3:
        new_suggestions += [("", 0.0)] * (3 - len(new_suggestions))

    top_suggestions = new_suggestions[:3]
    suggested_word = top_suggestions[0][0] if top_suggestions else ""
    update_text()

def reset():
    global current_word, full_sentence, suggested_word, last_letter_time
    current_word = ""
    full_sentence = []
    suggested_word = ""
    last_letter_time = 0
    update_text()

# ─── Controls Box ──────────────────────────────────────────
ctrl_buttons_frame = ttk.LabelFrame(right_frame, text="⚙️ Controls", padding=10)
ctrl_buttons_frame.pack(fill="x", pady=20)

for text, command in [
    ("❌ None of these", dismiss_suggestions),
    ("🔄 Reset", reset)
]:
    btn = tk.Button(ctrl_buttons_frame, text=text, font=button_font,
                    bg=button_bg, fg=button_fg, relief="raised", borderwidth=button_border,
                    command=command)
    btn.pack(fill="x", pady=5, ipadx=20, ipady=10)
    btn.bind("<Enter>", lambda e, b=btn: on_hover(e, b))
    btn.bind("<Leave>", lambda e, b=btn: on_leave(e, b))

# ─── Sign Language Guide (Moved to Bottom Right) ───────────
guide_frame = ttk.LabelFrame(right_frame, text="📘 ASL Guide", padding=10)
guide_frame.pack(pady=(10, 0))

asl_img = Image.open(os.path.join("assets", "ASL_image.png"))
asl_img = asl_img.resize((350, 250))
asl_imgtk = ImageTk.PhotoImage(asl_img)
asl_label = ttk.Label(guide_frame, image=asl_imgtk)
asl_label.image = asl_imgtk
asl_label.pack()

def on_key(event):
    global current_word, full_sentence, suggested_word, top_suggestions

    if event.keysym == "space":
        if current_word.strip():
            full_sentence.append(current_word.strip())
            current_word = ""

            # 🆕 Update suggestions right after appending a word
            context = full_sentence[-3:]
            top_suggestions = model.suggest_top_k(context, top_k_vocab=1000, num_suggestions=3)
            suggested_word = top_suggestions[0][0] if top_suggestions else ""
        update_text()

    elif event.keysym == "Return":
        accept_suggestion_choice(suggested_word)

    elif event.keysym == "BackSpace":
        if current_word:
            current_word = current_word[:-1]
        update_text()

def focus_main_window(event=None):
    window.focus_set()

def update_gui():
    global current_word, last_letter_time
    global last_detected_letter, stable_letter_start_time

    ret, frame = cap.read()
    if not ret:
        return

    flipped = cv2.flip(frame, 1)
    annotated_frame, detected_letter = detect_and_draw(flipped)

    now = time.time()
    show_timer = False
    remaining_time = 0

    if detected_letter and detected_letter.isalpha():
        if detected_letter != last_detected_letter:
            last_detected_letter = detected_letter
            stable_letter_start_time = now
        elif stable_letter_start_time:
            elapsed = now - stable_letter_start_time
            remaining_time = round(2 - elapsed, 1)
            if elapsed >= 2:
                current_word += detected_letter.lower()
                stable_letter_start_time = None
                last_detected_letter = None
                last_letter_time = now
            else:
                show_timer = True
    else:
        last_detected_letter = None
        stable_letter_start_time = None

    if detected_letter:
        cv2.putText(annotated_frame, f"Letter: {detected_letter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if show_timer:
            cv2.putText(annotated_frame, f"Accepting in {remaining_time}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)

    resized = cv2.resize(annotated_frame, (640, 480))  # or (480, 360)
    img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    update_text()
    window.after(60, update_gui)

window.bind("<Key>", on_key)
window.bind("<Button-1>", focus_main_window)
window.bind("<FocusIn>", focus_main_window)
window.after(100, lambda: window.focus_set())

update_gui()
window.mainloop()

cap.release()
cv2.destroyAllWindows()
