import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Load known face encodings and names
def load_face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    return face_recognition.face_encodings(image)[0]

known_faces = [
    ("Narendra Modi", "./photos/modi.jpg"),
    ("Ratan Tata", "./photos/ratantata.jpg"),
    ("Ujjwal Kumar", "./photos/ujjwal.jpeg"),
    ("Vivek Kumar", "./photos/vivek.jpeg"),
    ("Dr. R Chandrasekar", "./photos/sir.jpg")
]

known_face_encodings = [load_face_encoding(face[1]) for face in known_faces]
known_face_names = [face[0] for face in known_faces]

students = known_face_names.copy()

# Get the current date for the CSV file
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Open the CSV file for writing attendance
f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

class FaceRecognitionProcessor(VideoProcessorBase):
    def __init__(self):
        self.present_students = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    self.present_students.append((name, now.strftime("%H:%M:%S")))
                    print(name, "is Present")
                    students.remove(name)
                    print("Left Students Name: ", students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit app
st.title("Face Recognition Attendance System")

# Run the video processing function when 'Start Attendance' button is clicked
if st.button("Start Attendance"):
    st.write("Starting attendance...")
    present_students = []
    processor = FaceRecognitionProcessor()
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=FaceRecognitionProcessor,
        async_processing=True,
    )
    if webrtc_ctx.video_processor:
        with st.spinner("Waiting for video..."):
            while not webrtc_ctx.video_processor.present_students:
                pass
            present_students = webrtc_ctx.video_processor.present_students
        st.write("Today's Date:", now.strftime("%d-%m-%Y"))

        if present_students:
            st.write("Students who are present today are:")
            for student, time in present_students:
                st.write(f"- {student} - {time}")
else:
    st.write("Click the 'Start Attendance' button to begin.")
