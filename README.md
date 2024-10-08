# face-recognition
import streamlit as st
import cv2
import numpy as np
import face_recognition

# Set title for the app
st.title("Face Recognition App")

# Upload images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an array
    image = face_recognition.load_image_file(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Find all face locations and encodings in the uploaded image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Display face count
    st.write(f"Number of faces detected: {len(face_locations)}")

    # Draw rectangles around the faces
    for face_location in face_locations:
        top, right, bottom, left = face_location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    # Convert the image to a format Streamlit can display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    st.image(image, caption='Processed Image with Face Detection', use_column_width=True)
