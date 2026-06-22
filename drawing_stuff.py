
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt
import cv2

from math_stuff import transform_points
from geometry_data import *

GIZMO_SIZE_CM = 3.0

drawing_utils = mp.tasks.vision.drawing_utils
drawing_styles = mp.tasks.vision.drawing_styles

def draw_vector(image, p1, p2, view_transform, projection_transform, color):
    ndc_points = transform_points(np.array([p1, p2]), projection_transform @ view_transform, True)
    ndc_points[:, 1] *= -1
    screen_points = ndc_to_screen_points(ndc_points, (image.shape[1], image.shape[0]))
    cv2.arrowedLine(image, (screen_points[0][0], screen_points[0][1]), (screen_points[1][0], screen_points[1][1]), color, 2, cv2.LINE_AA)
    return image

def draw_landmarks(image, landmarks, view_transform, projection_transform, color):
    ndc_points = transform_points(landmarks, projection_transform @ view_transform, True)
    screen_points = ndc_to_screen_points(ndc_points, (image.shape[1], image.shape[0]))
    for point in screen_points: cv2.drawMarker(image, (point[0], point[1]), color, cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
    return image

# Draw a little coordinate axis gizmo using the specified object-to-view transform and camera projection transform.
def draw_gizmo(image: np.ndarray, view_transform: np.ndarray, projection_transform: np.ndarray, brightness: float = 1.0) -> np.ndarray:
    model_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) * GIZMO_SIZE_CM
    ndc_points = transform_points(model_points, projection_transform @ view_transform, True)
    screen_points = ndc_to_screen_points(ndc_points, (image.shape[1], image.shape[0]))
    cv2.arrowedLine(image, (screen_points[0][0], screen_points[0][1]), (screen_points[1][0], screen_points[1][1]), (0, 0, int(255 * brightness)), 2, cv2.LINE_AA)
    cv2.arrowedLine(image, (screen_points[0][0], screen_points[0][1]), (screen_points[2][0], screen_points[2][1]), (0, int(255 * brightness), 0), 2, cv2.LINE_AA)
    cv2.arrowedLine(image, (screen_points[0][0], screen_points[0][1]), (screen_points[3][0], screen_points[3][1]), (int(255 * brightness), 0, 0), 2, cv2.LINE_AA)
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
def draw_hand_landmarks(image, landmarks, hand):
    assert (hand == "Left" or hand == "Right")
    lines = {"Left": CANONICAL_LEFTHAND_INDICES, "Right": CANONICAL_RIGHTHAND_INDICES}
    colors = {"Left": CANONICAL_LEFTHAND_COLORS, "Right": CANONICAL_RIGHTHAND_COLORS}

    text_color = (210, 210, 210)
    linetype = cv2.LINE_AA

    # Draw some circles at each landmarks.
    for i, landmark in enumerate(landmarks):
        color = colors[hand][i]
        p = (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
        cv2.circle(image, p, 3, color, -1, linetype)

    # Connect with lines.
    for line in lines[hand]:
        i1 = line[0]
        i2 = line[1]
        p1 = (int(landmarks[i1].x * image.shape[1]), int(landmarks[i1].y * image.shape[0]))
        p2 = (int(landmarks[i2].x * image.shape[1]), int(landmarks[i2].y * image.shape[0]))
        color = colors[hand][i2]
        cv2.line(image, p1, p2, color, 1, linetype)

    # Offset a little from the top left of the hand's bounding box.
    x = int(min([p.x for p in landmarks]) * image.shape[1])
    y = int(min([p.y for p in landmarks]) * image.shape[0]) - 5

    # Label the hand.
    cv2.putText(image, f"{hand}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 1, linetype)

    return image
