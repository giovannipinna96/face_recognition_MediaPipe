import time

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles


def detect_faces_video(tffile, video_file_buffer, max_num_faces, min_detection_confidence, tracking_confidence):
    try:
        if video_file_buffer:
            tffile.write(video_file_buffer.read())
            video = cv2.VideoCapture(tffile.name)

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))

        codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.Video('output.mp4', codec, fps, (width, height))
        fps = 0
        i = 0
        # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        with mp_face_mesh.FaceMesh(
                max_num_faces=max_num_faces,
                min_tracking_confidence=tracking_confidence,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence) as face_mesh:
            prevTime = 0
            while video.isOpened():
                i += 1
                ret, frame = video.read()
                if not ret:
                    continue

                results = face_mesh.process(frame)
                frame.flags.writeable = True

                face_count = 0
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        face_count += 1
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

        return out, fps, face_count, width
    except:
        return None, None, 0, None
