import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings
import cv2
import numpy as np
import csv
from datetime import datetime
import face_recognition

# Load known face encodings and names
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

# Define a custom video processor
class CustomVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        pass

    def process(self, frame) -> None:
        # Resize the frame to 1/4 of its original size
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2

                cv2.putText(frame, name + ' Present',
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)

                if name in students:
                    present_students.append((name, now.strftime("%H:%M:%S")))
                    print(name, "is Present")
                    students.remove(name)
                    print("Left Students Name: ", students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])

# Streamlit app
st.title("Simple Face Recognition Attendance System")

# Run the video processing function when 'Start Attendance' button is clicked
if st.button("Start Attendance"):
    st.write("Starting attendance...")
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=CustomVideoProcessor,
        client_settings=ClientSettings(),
    )

    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.input_frame()

    st.write("Today's Date:", now.strftime("%d-%m-%Y"))

    present_students = []  # You need to populate this with recognized students
    if present_students:
        st.write("Students who are present today are:")
        for student, time in present_students:
            st.write(f"- {student} - {time}")
else:
    st.write("Click the 'Start Attendance' button to begin.")
