import cv2
import os
import numpy as np

dataset_path = "dataset/students"

faces = []
labels = []
label_map = {}
current_label = 0

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    label_map[current_label] = folder

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        faces.append(img)
        labels.append(current_label)

    current_label += 1

faces = np.array(faces)
labels = np.array(labels)

# Create LBPH face recognizer
model = cv2.face.LBPHFaceRecognizer_create()

# Train model
model.train(faces, labels)

# Save model
model.save("face_model.yml")

# Save label mapping
np.save("labels.npy", label_map)

print("Training completed successfully")
print("Model saved as face_model.yml")
