import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import time

# Import from our previous files
from environment_data_load import emotion_labels, IMG_WIDTH, IMG_HEIGHT, DEVICE
from model_architecture import EmotionCNN

# Constants
NUM_CLASSES = 7
MODEL_PATH = 'emotion_model_best.pth'

# Load model
model = EmotionCNN(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def realtime_emotion_recognition():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Starting real-time emotion recognition...")
    print("Press 'q' to quit")

    frame_count = 0
    start_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        frame_count += 1
        if frame_count >= 10:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]

            try:
                face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)).convert('L')
                transform = transforms.Compose([
                    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ])
                face_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    output = model(face_tensor)
                    _, predicted = torch.max(output, 1)
                    probabilities = torch.nn.functional.softmax(output, dim=1)[0]

                emotion_idx = predicted.item()
                emotion_text = emotion_labels[emotion_idx]
                probability = probabilities[emotion_idx].item()

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion_text}: {probability:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                bar_width = 100
                bar_height = 15
                for i, prob in enumerate(probabilities.cpu().numpy()):
                    bar_x = frame.shape[1] - bar_width - 10
                    bar_y = 30 + i * 25

                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)
                    filled_width = int(bar_width * prob)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 255, 0), -1)
                    cv2.putText(frame, f"{emotion_labels[i]}: {prob:.2f}", (bar_x - 100, bar_y + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            except Exception as e:
                print(f"Error processing face: {e}")

        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show the frame
        cv2.imshow('Real-time Emotion Recognition', frame)

        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_emotion_recognition()
