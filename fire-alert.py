"""
Fire/Smoke Detection & Alert System with ResNet18

Trains a CNN (ResNet18) to detect Fire vs Normal
Runs real-time detection from webcam/video feed
Sends WhatsApp alert if Fire is detected 
"""

import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch import nn, optim
from twilio.rest import Client
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import time

load_dotenv()  # load .env file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data"
MODEL_PATH = "models/fire_model.pth"
CLASS_NAMES = ['fire', 'normal']
ALERT_THRESHOLD = 0.90

# Twilio credentials
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_WHATSAPP = os.getenv("TWILIO_WHATSAPP")
ALERT_TO = os.getenv("ALERT_TO")

last_alert_time = 0


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def load_dataset():
    train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    test_data = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)
    return train_loader, test_loader


def create_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(DEVICE)


def train_model(model, train_loader, test_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # evaluate on test set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model trained & saved")


def send_whatsapp_alert(label, confidence):
    global last_alert_time
    now = time.time()
    if now - last_alert_time < 60:   # only send once per 60s
        return
    last_alert_time = now
    client = Client(TWILIO_SID, TWILIO_AUTH)
    msg = client.messages.create(
        body=f"Fire ALERT: {label.upper()} detected with {confidence*100:.1f}% confidence!",
        from_=TWILIO_WHATSAPP,
        to=ALERT_TO
    )
    print(f"WhatsApp Alert sent! SID: {msg.sid}")


def detect_realtime(model):
    model.eval()
    cap = cv2.VideoCapture(0)  # 0 = webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = test_transform(img_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probs, dim=0)
            label = CLASS_NAMES[predicted_idx]

        # Draw on frame
        color = (0, 0, 255) if label == 'fire' else (0, 255, 0)
        cv2.putText(frame, f"{label}: {confidence:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if label == 'fire' and confidence > ALERT_THRESHOLD:
            send_whatsapp_alert(label, confidence)

        cv2.imshow("Fire Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    train_loader, test_loader = load_dataset()
    if not os.path.exists(MODEL_PATH):
        model = create_model()
        train_model(model, train_loader, test_loader)
    else:
        print("Model already trained. Loading the saved model...")
        model = create_model()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    detect_realtime(model)
