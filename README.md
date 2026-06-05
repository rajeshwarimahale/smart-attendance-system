# Smart Attendance System using Face Recognition

## Overview
This project is a Smart Attendance System that automatically marks attendance
using face recognition. It helps to avoid proxy attendance and reduces manual
work in classrooms or institutions.

The system uses a webcam to detect and recognize registered student faces and
marks attendance with date and time.

---

## Problem Statement
Traditional attendance systems are manual and can be misused by proxy attendance.
This project solves the problem by using computer vision to identify students
automatically.

---

## Features
- Student face registration using webcam
- Face recognition using LBPH algorithm
- Automatic attendance marking
- Supports multiple students
- Easy retraining after deleting students
- Privacy-safe (no face images uploaded to GitHub)

---

## Technologies Used
- Python 3.10
- OpenCV (opencv-contrib-python) — Haar cascades + LBPH algorithm
- NumPy
- Pandas — attendance report generation
- SQLite — attendance record storage
- scikit-learn
---

## Project Structure
smart-attendance/
├── attendance/
│ └── mark_attendance.py
├── dataset/
│ └── students/
│ └── .gitkeep
├── register_face.py
├── train_model.py
├── recognize_face.py
├── requirements.txt
├── README.md
└── .gitignore


---

## How to Run the Project

⚠️ Note:
Make sure Python 3.10 is installed and your system has a working webcam.

1️⃣ Open Terminal in Project Folder
cd D:\Project\smart-attendance

2️⃣ Install Required Dependencies
pip install -r requirements.txt


If pip does not work, use:

py -3.10 -m pip install -r requirements.txt

3️⃣ Register Student Face

This step captures face images using the webcam.

python register_face.py


Process:

Enter student name

Enter student ID

Webcam opens

Face images are captured automatically

Images are stored in:

dataset/students/<ID>_<Name>/

4️⃣ Train the Face Recognition Model

This step trains the LBPH face recognition model.

python train_model.py


Output:

face_model.yml

labels.npy

5️⃣ Start Smart Attendance System

This step recognizes faces in real time and marks attendance.

python recognize_face.py


Result:

Webcam opens

Recognized student name appears on screen

Attendance is saved in:

attendance/attendance_<date>.csv

⏹️ Stop the Program

Press q in the camera window
OR

Close the camera window

🔁 For Multiple Students

Run register_face.py for each student

Run train_model.py again

Run recognize_face.py
