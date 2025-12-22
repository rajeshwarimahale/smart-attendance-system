
import cv2
import os
import time

name = input("Enter student name: ")
student_id = input("Enter student ID: ")

path = f"dataset/students/{student_id}_{name}"
os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Auto capturing images... Do not press any key.")

while count < 20:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Face Registration", frame)

    cv2.imwrite(f"{path}/{count}.jpg", gray)
    print(f"Image {count} saved")
    count += 1

    time.sleep(1)

cap.release()
cv2.destroyAllWindows()
print("Face registration completed")
