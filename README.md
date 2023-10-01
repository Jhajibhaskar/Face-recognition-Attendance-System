# Simple Face-recognition-Attendance-System
## Purpose:
This project is a Face Recognition Attendance System using Python, OpenCV, and face_recognition library. It captures video from the default camera, detects faces, and recognizes them using pre-trained face encodings. It maintains attendance records in a CSV file for recognized faces.<br><br>
Upon Running a Model a CSV file will be created automatically filename as Today’s Date(YYYY-MM-DD) format & will store Name of Student as well as Reporting Time.
## Project Structure
The project consists of the following main components:
### Imports and Libraries
1. face_recognition for face recognition tasks.
2. cv2 for video capture and image processing.
3. numpy for numerical operations.
4. csv for working with CSV files.
5. os for interacting with the operating system.
6. datetime for handling date and time.
### Initialization and Encoding of Known Faces
➢Images of known faces are loaded and their encodings are computed using face_recognition.face_encodings.
### Video Capture and Face Recognition
➢The video feed is captured using OpenCV.<br>
➢Face locations and encodings are computed for each frame.<br>
➢The computed encodings are compared with the known encodings to recognize the faces.<br>
➢If a recognized face is found, attendance is marked and recorded in a CSV file.
### Attendance Logging
➢Attendance records, including the name and timestamp, are logged in a CSV file for the current date.

## How to Use
➢Install the required libraries<br>
➢Run the provided Python script AttendanceMain.ipynb<br>
➢The system will open a window showing the camera feed with recognized faces and their attendance status.<br>
➢Press 'q' to exit the program and save the attendance records for the day.<br>
## Requirements to Install:
1.Face Recognition by ``pip install face-recognition`` <br>
2.OpenCV by ``pip install opencv-python`` <br>
3.NumPy by ``pip install numpy`` <br>
4.CSV : This is a built-in library in Python, so there is no need to install it separately. <br>
5.OS : OS is a built-in library in Python, so we do not need to install it separately. <br>
6.datetime : Datetime is also a built-in library in Python.

## Deployment on Streamlit
To deploy the project on Streamlit:<br>
➢Run the provided Python script streamlit_app.py using: ``streamlit run streamlit_app.py``<br>
➢Open the provided URL in your web browser to use the face recognition attendance system through the Streamlit interface.<br>
➢The project is accessible through Streamlit Share, offering a web-based interface for face recognition and attendance at: https://jhajibhaskar2.streamlit.app
