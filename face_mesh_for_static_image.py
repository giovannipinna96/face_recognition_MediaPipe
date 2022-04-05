import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles


def detect_faces(image, max_num_faces, min_detection_confidence, n_faces_detect):
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    try:
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence) as face_mesh:
            results = face_mesh.process(image)
            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                n_faces_detect += 1
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())

        return annotated_image, n_faces_detect
    except:
        return None, 0
