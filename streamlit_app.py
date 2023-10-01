import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import streamlit as st
import streamlit_webrtc as webrtc

# Load known face encodings and names (from your original code)
modi_image = face_recognition.load_image_file("./photos/modi.jpg")
modi_encoding = face_recognition.face_encodings(modi_image)[0]

ratan_tata_image = face_recognition.load_image_file("./photos/ratantata.jpg")
ratan_tata_encoding = face_recognition.face_encodings(ratan_tata_image)[0]

ujjwal_image = face_recognition.load_image_file("./photos/ujjwal.jpeg")
ujjwal_encoding = face_recognition.face_encodings(ujjwal_image)[0]

vivek_image = face_recognition.load_image_file("./photos/vivek.jpeg")
vivek_encoding = face_recognition.face_encodings(vivek_image)[0]

sir_image = face_recognition.load_image_file("./photos/sir.jpg")
sir_encoding = face_recognition.face_encodings(sir_image)[0]

known_face_encoding = [
    modi_encoding,
    ratan_tata_encoding,
    ujjwal_encoding,
    vivek_encoding,
    sir_encoding
]

known_faces_names = [
    "Narendra Modi",
    "Ratan Tata",
    "Ujjwal Kumar",
    "Vivek Kumar",
    "Dr. R Chandrasekar"
]

students = known_faces_names.copy()

# Get the current date for the CSV file
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Open the CSV file for writing attendance
f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

# Function to process video frames and perform face recognition
def process_video(frame):
  face_locations = face_recognition.face_locations(frame)
  face_encodings = face_recognition.face_encodings(frame, face_locations)
  face_names = []

  for face_encoding in face_encodings:
    matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
    name = ""
    face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
    best_match_index = np.argmin(face_distance)
    if matches[best_match_index]:
      name = known_faces_names[best_match_index]

    face_names.append(name)

  return frame, face_names

# Streamlit app
st.title("Simple Face Recognition Attendance System")

# Run the video processing function when 'Start Attendance' button is clicked
if st.button("Start Attendance"):
  st.write("Starting attendance...")

  # Create a webrtc video stream
  stream = webrtc.VideoStream()

  # Start the video stream
  stream.start()

  # Process each frame of the video stream
  while True:
    frame = stream.read()
    processed_frame, face_names = process_video(frame)

    # Display the processed frame to the user
    stream.write(processed_frame)

    # Write the attendance to the CSV file
    for name in face_names:
      current_time = now.strftime("%H-%M-%S")
      lnwriter.writerow([name, current_time])

else:
  st.write("Click the 'Start Attendance' button to begin.")
