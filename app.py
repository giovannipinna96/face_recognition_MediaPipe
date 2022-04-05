import numpy as np
import streamlit as st
from PIL import Image

from face_mesh_for_static_image import detect_faces
from fece_mesh_for_webcam import detect_web_cam

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
    pass
