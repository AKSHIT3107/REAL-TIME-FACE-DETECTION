import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load Models
# -------------------------------
# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Pretrained FaceNet for embeddings
embedder = FaceNet()

# Storage for registered users {name: embedding}
registered_users = {}

# -------------------------------
# Helper Functions
# -------------------------------

def get_face_embedding(face_img):
    """Convert cropped face image into embedding using FaceNet."""
    face_img = cv2.resize(face_img, (160, 160))   # FaceNet expects 160x160
    face_img = np.expand_dims(face_img, axis=0)
    embedding = embedder.embeddings(face_img)
    return embedding[0]   # Return 128D vector

def register_user(frame):
    """Register new user by capturing face and storing embedding."""
    name = simpledialog.askstring("Register User", "Enter user name:")
    if not name:
        return
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        messagebox.showerror("Error", "No face detected. Try again!")
        return
    
    # Take first detected face
    (x, y, w, h) = faces[0]
    face = frame[y:y+h, x:x+w]
    embedding = get_face_embedding(face)
    
    registered_users[name] = embedding
    messagebox.showinfo("Success", f"User {name} registered successfully!")

def recognize_user(frame):
    """Recognize face by comparing embedding with registered users."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return frame
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        embedding = get_face_embedding(face)

        best_match = None
        best_score = 0.0
        
        for name, ref_embedding in registered_users.items():
            score = cosine_similarity([embedding], [ref_embedding])[0][0]
            if score > best_score:
                best_score = score
                best_match = name
        
        if best_match and best_score > 0.7:  # Threshold for recognition
            label = f"{best_match} ({best_score:.2f})"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return frame

# -------------------------------
# GUI and Video Stream
# -------------------------------
def start_app():
    def update_frame():
        ret, frame = cap.read()
        if not ret:
            return
        
        # Recognize user in the frame
        frame_out = recognize_user(frame.copy())
        
        # Convert to Tkinter format
        img = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        video_label.after(10, update_frame)

    def on_register():
        ret, frame = cap.read()
        if ret:
            register_user(frame)

    def on_exit():
        cap.release()
        root.destroy()

    # Tkinter Window
    root = tk.Tk()
    root.title("Face Recognition System")

    video_label = tk.Label(root)
    video_label.pack()

    btn_frame = tk.Frame(root)
    btn_frame.pack(fill="x")

    register_btn = tk.Button(btn_frame, text="Register User", command=on_register)
    register_btn.pack(side="left", expand=True, fill="x")

    exit_btn = tk.Button(btn_frame, text="Exit", command=on_exit)
    exit_btn.pack(side="left", expand=True, fill="x")

    update_frame()
    root.mainloop()

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Start webcam
    start_app()
