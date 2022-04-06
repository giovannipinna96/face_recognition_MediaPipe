import tempfile
import time

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

from face_mesh_for_static_image import detect_faces
from fece_mesh_for_webcam import detect_web_cam

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles

st.title('Face Recognition using MediaPipe')
with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

st.sidebar.title('Face Recognition Sidebar')
st.sidebar.subheader('Parameters')

app_mode = st.sidebar.selectbox('Choose the app mode', ['Run on Image', 'Run on WebCam', 'Run on video'])

if app_mode == 'Run on Image':
    st.markdown('''**Detect Faces**''')
    n_faces_detect = st.markdown('0')
    max_faces = st.sidebar.number_input('Maximum number of faces', value=1, min_value=1, max_value=20)
    st.sidebar.markdown('''---''')
    detection_confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('''---''')

    img_upload = st.sidebar.file_uploader('Upload a image', type=['png', 'jpg', 'jpeg'])
    if img_upload is not None:
        st.sidebar.text('Original image')
        st.sidebar.image(img_upload)
        n_faces = 0
        image_pil = Image.open(img_upload).convert('RGB')
        image_array = np.array(image_pil)
        annotated_img, n_faces_img = detect_faces(image_array, max_faces, detection_confidence, 0)
        if annotated_img is not None and n_faces_img is not None:
            n_faces_detect.write(f"<h1 class='nfaces'>{n_faces_img}</h1>", unsafe_allow_html=True)
            st.image(annotated_img, use_column_width=True)
        else:
            n_faces_detect.write("Not found faces try to change parameters or try to change photo")
    else:
        st.text('Please upload an image in the Sidebar')

elif app_mode == 'Run on WebCam':
    st.markdown('''**Detect Faces**''')
    n_faces_detect = st.markdown('0')
    n_faces = 0
    max_faces = st.sidebar.number_input('Maximum number of faces', value=1, min_value=1, max_value=20)

    st.markdown("### Webcam Live Feed")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    force_stop = 4
    while run:
        frame, n_faces_img = detect_web_cam(max_faces, n_faces)
        if frame is not None and n_faces_img != 0:
            n_faces_detect.write(f"<h1 class='nfaces'>{n_faces_img}</h1>", unsafe_allow_html=True)
            FRAME_WINDOW.image(frame)
        else:
            st.text(f'Change parameter I can not detect nothing you have {force_stop - 1}')
            force_stop -= 1
            if force_stop == 0:
                run = False
                FRAME_WINDOW.image([])
    else:
        st.write('Stopped')

else:
    max_faces = st.sidebar.number_input('Maximum number of faces', value=1, min_value=1, max_value=20)
    st.sidebar.markdown('''---''')
    detection_confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('''---''')
    detection_confidence_tracking = st.sidebar.slider('Confidence tracking', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('''---''')
    try:
        stframe = st.empty()
        video_file_buffer = st.sidebar.file_uploader('Upload a video', type=['mp4', 'mov', 'avi', 'asf'])
        if video_file_buffer is not None:
            tffile = tempfile.NamedTemporaryFile(delete=False)
            tffile.write(video_file_buffer.read())
            video = cv2.VideoCapture(tffile.name)
            st.sidebar.text('Input video')
            st.sidebar.video(tffile.name)

            frame_rate, n_faces_detect, image_wigth = st.columns(3)

            with frame_rate:
                st.markdown('**Frame Rate**')
                frame_rate = st.markdown('0')
            with n_faces_detect:
                st.markdown('**Faces detected**')
                n_faces_detect = st.markdown('0')
            with image_wigth:
                st.markdown('**Image Width**')
                image_wigth = st.markdown('0')
            st.markdown('<hr/>', unsafe_allow_html=True)

            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video.get(cv2.CAP_PROP_FPS))

            codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            out = cv2.VideoWriter('output.mp4', codec, fps, (width, height))
            fps = 0
            i = 0
            drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
            with mp_face_mesh.FaceMesh(
                    max_num_faces=max_faces,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as face_mesh:
                prevTime = 0
                while video.isOpened():
                    i += 1
                    ret, frame = video.read()
                    if not ret:
                        video.release()
                        out.release()
                        break

                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    frame.flags.writeable = False
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(frame)

                    # Draw the face mesh annotations on the image.
                    frame.flags.writeable = True
                    n_faces = 0
                    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            n_faces += 1
                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles
                                    .get_default_face_mesh_tesselation_style())
                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles
                                    .get_default_face_mesh_contours_style())
                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_IRISES,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles
                                    .get_default_face_mesh_iris_connections_style())

                    # FPS counter logic
                    currTime = time.time()
                    fps = 1 / (currTime - prevTime)
                    prevTime = currTime
                    out.write(frame)

                    frame_rate.write(f"<h1 class='nfaces'>{int(fps)}</h1>", unsafe_allow_html=True)
                    n_faces_detect.write(f"<h1 class='nfaces'>{n_faces}</h1>", unsafe_allow_html=True)
                    image_wigth.write(f"<h1 class='nfaces'>{width}</h1>", unsafe_allow_html=True)
                    frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
                    stframe.image(frame, channels='RGB', use_column_width=True)

    except:
        st.text('there something that not work, please check the parameters or change video')
