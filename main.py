import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import math

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

def make_frustum(fov_y_rads: float, aspect_ratio: float, near: float, far: float):
    half_tan = math.tan(fov_y_rads / 2)
    top = near * half_tan
    right = top * aspect_ratio

    matrix = np.array([[near / right, 0,          0,                               0],
                       [0,            near / top, 0,                               0],
                       [0,            0,         -(far + near) / (far - near),    -1],
                       [0,            0,         -(2 * far * near) / (far - near), 0]])
    return matrix

def draw_face_landmarks(image, result):
    face_landmarks_list = result.face_landmarks
    # annotated_image = np.copy(image)

    for face_landmarks in face_landmarks_list:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
    return image

def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    _, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()

def draw_hand_landmarks(image, result):
    hand_landmarks_list = result.hand_landmarks
    handedness_list = result.handedness

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return image

# Takes a series of 3D input points, performs matrix multipy, and performs perspective divide.
def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    p1 = np.hstack((points,np.ones((points.shape[0],1))))
    p2 = np.array([np.matmul(np.array(p), transform) for p in p1])
    p3 = np.array([p / p[3] for p in p2])
    return p3

def ndc_to_screen_points(ndc_points: np.ndarray, screen_size: tuple[int, int]) -> np.ndarray:
    return np.array([[int((p[0] + 1) * 0.5 * screen_size[0]), int((-p[1] + 1) * 0.5 * screen_size[1])] for p in ndc_points])

def draw_gizmo(image: np.ndarray, view_transform: np.ndarray, projection_transform: np.ndarray) -> np.ndarray:
    # y_vals = np.copy(view_transform[1])
    # view_transform[1] = view_transform[2]
    # view_transform[2] = y_vals
    model_points = np.array([[0.1, 0.1, 0.1], [10, 0, 0], [0, 10, 0], [0, 0, 10]])
    ndc_points = transform_points(model_points, np.matmul(view_transform, projection_transform))
    screen_points = ndc_to_screen_points(ndc_points, (image.shape[1], image.shape[0]))

    cv2.arrowedLine(image, (screen_points[0][0], screen_points[0][1]), (screen_points[1][0], screen_points[1][1]), (0, 0, 255), 3)
    cv2.arrowedLine(image, (screen_points[0][0], screen_points[0][1]), (screen_points[2][0], screen_points[2][1]), (0, 255, 0), 3)
    cv2.arrowedLine(image, (screen_points[0][0], screen_points[0][1]), (screen_points[3][0], screen_points[3][1]), (255, 0, 0), 3)
    return image

def show_image(image: np.ndarray, name: str = "video") -> int:
    cv2.imshow(name, image)
    return (cv2.waitKey(1) & 0xFF) # For a video stream, we need to pass a positive timeout value so that OpenCV handles window events.

def init_camera(requested_size:tuple[int,int] = (1280, 720), requested_fps: float = 15.0) -> tuple[cv2.VideoCapture, tuple[int, int], float]:
    cam = cv2.VideoCapture(0) # Passing 0 for index gets us the default camera. We could try to access it by name instead.

    # Request some video properties. We may not be able to set these.
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, requested_size[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_size[1])
    cam.set(cv2.CAP_PROP_FPS, requested_fps)

    # Get the actual video properties.
    size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cam.get(cv2.CAP_PROP_FPS)
    print(f"Video resolution: {size[0]}x{size[1]}, {fps:.2f}fps.")
    return cam, size, fps

def init_hand_detector():
    base_options = BaseOptions(model_asset_path='hand_landmarker.task')
    options = HandLandmarkerOptions(base_options=base_options, running_mode = VisionRunningMode.VIDEO, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    return detector

def init_face_detector():
    base_options = BaseOptions(model_asset_path='face_landmarker.task')
    options = FaceLandmarkerOptions(base_options=base_options, running_mode = VisionRunningMode.VIDEO, output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    return detector

if __name__ == "__main__":

    # Initialize detector objects and video streams.
    hand_detector = init_hand_detector()
    face_detector = init_face_detector()
    cam, resolution, fps = init_camera()
    video_writer = cv2.VideoWriter("video.mp4", cv2.VideoWriter_fourcc(*"WEBM"), fps, resolution)

    # Set up some timing info.
    frame_number: int = 0
    start_time_ns: int = time.monotonic_ns()
    last_time_ns: int = start_time_ns

    # Create a perspective projection matrix. Based vaguely on the known FOV of my webcam.
    resolution = (1280,720)
    projection_transform = make_frustum(math.radians(53.7999992), resolution[0] / resolution[1], 0.1, 100)

    while cam.isOpened():
        # Read the next video frame (or break out of the loop if we didn't get one) and update timing info.
        success, frame = cam.read()
        if not success: break
        current_time_ns = time.monotonic_ns()
        elapsed_ms = int((current_time_ns - last_time_ns) // 1e6)
        timestamp_ms = int(1000 * frame_number / fps)
        last_time_ns = current_time_ns
        # print(f"Detecting frame {frame_number}, last frame took {elapsed_ms}ms.")

        # Convert to a mediapipe image and run through face+hand detection.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        face_detector_result = face_detector.detect_for_video(mp_image, timestamp_ms)
        hand_detector_result = hand_detector.detect_for_video(mp_image, timestamp_ms)

        # Create a black image and annotate it with our results. We could overlay on top of the original frame, if we wanted.
        annotated_image = np.zeros_like(mp_image.numpy_view())
        # annotated_image = np.copy(mp_image.numpy_view())
        annotated_image = draw_face_landmarks(annotated_image, face_detector_result)
        annotated_image = draw_hand_landmarks(annotated_image, hand_detector_result)

        if len(face_detector_result.facial_transformation_matrixes) > 0:
            annotated_image = draw_gizmo(annotated_image, face_detector_result.facial_transformation_matrixes[0].T, projection_transform)
        else:
            view_transform = np.eye(4)
            view_transform[3][1] = -10
            view_transform[3][2] = -20
            # annotated_image = draw_gizmo(annotated_image, view_transform, projection_transform)
        # Write our annotated video frame to the output stream.
        video_writer.write(annotated_image)

        # Try to show the image window. If the user pressed "ESC", exit the loop.
        key = show_image(annotated_image)
        if (key == ord('\x1b')): break # '\x1b' is ESC key

        # If the user pressed "b", show the blendshapes plot.
        if (key == ord('b')) and len(face_detector_result.face_blendshapes) > 0:
            plot_face_blendshapes_bar_graph(face_detector_result.face_blendshapes[0])

        if (key == ord('o')):
            print(face_detector_result.facial_transformation_matrixes)

        frame_number += 1

    cam.release()
    video_writer.release()
    cv2.destroyAllWindows()
