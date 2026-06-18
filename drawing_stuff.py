
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt
import cv2

from math_stuff import transform_points

GIZMO_SIZE_CM = 5.0

HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections
drawing_utils = mp.tasks.vision.drawing_utils
drawing_styles = mp.tasks.vision.drawing_styles

def draw_landmarks(image, landmarks, view_transform, projection_transform, color):
    ndc_points = transform_points(landmarks, projection_transform @ view_transform, True)
    screen_points = ndc_to_screen_points(ndc_points, (image.shape[1], image.shape[0]))
    for point in screen_points:
        cv2.drawMarker(image, (point[0], point[1]), color, cv2.MARKER_CROSS)
    return image

# Draw a little coordinate axis gizmo using the specified object-to-view transform and camera projection transform.
def draw_gizmo(image: np.ndarray, view_transform: np.ndarray, projection_transform: np.ndarray, brightness: float = 1.0) -> np.ndarray:
    model_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) * GIZMO_SIZE_CM
    ndc_points = transform_points(model_points, projection_transform @ view_transform, True)
    screen_points = ndc_to_screen_points(ndc_points, (image.shape[1], image.shape[0]))
    cv2.arrowedLine(image, (screen_points[0][0], screen_points[0][1]), (screen_points[1][0], screen_points[1][1]), (0, 0, int(255 * brightness)), 3)
    cv2.arrowedLine(image, (screen_points[0][0], screen_points[0][1]), (screen_points[2][0], screen_points[2][1]), (0, int(255 * brightness), 0), 3)
    cv2.arrowedLine(image, (screen_points[0][0], screen_points[0][1]), (screen_points[3][0], screen_points[3][1]), (int(255 * brightness), 0, 0), 3)
    return image

# Convert NDC to screen coordinates. Note that the Y-axis is inverted.
def ndc_to_screen_points(ndc_points: np.ndarray, screen_size: tuple[int, int]) -> np.ndarray:
    return np.array([[int((p[0] + 1) * 0.5 * screen_size[0]), int((-p[1] + 1) * 0.5 * screen_size[1])] for p in ndc_points])

# Face landmark drawing from the mediapipe example code.
def draw_face_landmarks(image, result):
    face_landmarks_list = result.face_landmarks
    for face_landmarks in face_landmarks_list:
        drawing_utils.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
        drawing_utils.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
        drawing_utils.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
        drawing_utils.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
    return image

# Blendshape plotting from the mediapipe example code.
def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    _, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values.
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()

# Hand landmark drawing from the mediapipe example code.
def draw_hand_landmarks(image, result):
    hand_landmarks_list = result.hand_landmarks
    handedness_list = result.handedness

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        drawing_utils.draw_landmarks(
            image,
            hand_landmarks,
            HandLandmarksConnections.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the hand's bounding box.
        height, width, _ = image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - 10

        # Draw text to denote which hand is which.
        cv2.putText(image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    1, (88, 205, 54), 1, cv2.LINE_AA)
    return image
