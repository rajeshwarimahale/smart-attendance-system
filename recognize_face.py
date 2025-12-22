import cv2
import numpy as np
from attendance.mark_attendance import mark_attendance

# Load trained model and labels
model = cv2.face.LBPHFaceRecognizer_create()
model.read("face_model.yml")

label_map = np.load("labels.npy", allow_pickle=True).item()

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
marked = False

print("Camera started for attendance...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        label, confidence = model.predict(face_img)

        if confidence < 80:
            name = label_map[label]
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if not marked:
                mark_attendance(name)
                marked = True
        else:
            cv2.putText(frame, "Unknown", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("Smart Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
