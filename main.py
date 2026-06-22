import time

import mediapipe as mp
import numpy as np
import cv2

from drawing_stuff import *
from math_stuff import *
from geometry_data import *

DO_FACE_TRACKING = True
DO_HAND_TRACKING = True

DRAW_FACE_LANDMARKS = True
DRAW_HAND_LANDMARKS = True
DRAW_HAND_CANONICAL_MODEL = True

DRAW_FACE_TRANFORM = True
DRAW_MEDIAPIPE_FACE_TRANSFORM = True # The face landmarker model can also output a transform. We can draw it for comparison purposes.
DRAW_HAND_TRANSFORM = True

DRAW_ORIGINAL_IMAGE = True
DO_VIDEO_OUTPUT = True

HAND_LANDMARKER_TASK_PATH = "hand_landmarker.task"
FACE_LANDMARKER_TASK_PATH = "face_landmarker.task"

REQUESTED_CAMERA_INDEX = 0 # Note that 0 is the default camera for the system.
REQUESTED_CAMERA_RESOLUTION = (1280, 720)
REQUESTED_CAMERA_FRAMERATE = 30.0

VIDEO_OUTPUT_FILENAME = "video.webm"
VIDEO_OUTPUT_FOURCC = "vp80"
WINDOW_NAME = "video"

# DEFAULT_CAMERA_FOVY_DEG= 53.7999992
DEFAULT_CAMERA_FOVY_DEG= 62.0

def show_image(image: np.ndarray, name: str) -> int:
    cv2.imshow(name, image)
    return (cv2.waitKey(1) & 0xFF) # For a video stream, we need to pass a positive timeout value so that OpenCV handles window events.

def init_camera(requested_idx, requested_size:tuple[int,int], requested_fps: float) -> tuple[cv2.VideoCapture, tuple[int, int], float]:
    cam = cv2.VideoCapture(requested_idx) # We use a device index, but probably there is a way to request by name instead.

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
    base_options = mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task')
    options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options,
                                                    running_mode = mp.tasks.vision.RunningMode.VIDEO,
                                                    num_hands=2)
    detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
    return detector

def init_face_detector():
    base_options = mp.tasks.BaseOptions(model_asset_path='face_landmarker.task')
    options = mp.tasks.vision.FaceLandmarkerOptions(base_options=base_options,
                                                    running_mode = mp.tasks.vision.RunningMode.VIDEO,
                                                    output_face_blendshapes=True,
                                                    output_facial_transformation_matrixes=DRAW_MEDIAPIPE_FACE_TRANSFORM,
                                                    num_faces=1)
    detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    return detector

def is_window_open(name: str) -> bool:
    # We should mostly just be able to check if WIND_PROP_VISIBLE is < 1 (on most operating systems it will be 0 or negative if window is closed).
    # Apparently on some linux GUI backends, checking any window property of a closed window can cause a null pointer exception, so we'll check for that too.
    is_open = True
    try:
        if cv2.getWindowProperty(name, cv2.WND_PROP_ASPECT_RATIO) < 0: is_open = False
    except cv2.error as e:
        if e.code == -27: is_open = False # -27 is OpenCV's code for null pointer exception.
        else: raise e
    return is_open

def main():
    # Initialize detector objects and video streams.
    face_detector = init_face_detector() if DO_FACE_TRACKING else None
    hand_detector = init_hand_detector() if DO_HAND_TRACKING else None
    cam, resolution, fps = init_camera(REQUESTED_CAMERA_INDEX, REQUESTED_CAMERA_RESOLUTION, REQUESTED_CAMERA_FRAMERATE)
    video_writer = cv2.VideoWriter(VIDEO_OUTPUT_FILENAME, cv2.VideoWriter.fourcc(*VIDEO_OUTPUT_FOURCC), fps, resolution) if DO_VIDEO_OUTPUT else None

    # Set up some timing info.
    frame_number: int = 0
    start_time_ns: int = time.monotonic_ns()
    last_time_ns: int = start_time_ns

    # Create a perspective projection matrix. Based vaguely on the known FOV of the camera I'm using (Azure Kinect v2).
    projection_transform = make_frustum(np.deg2rad(DEFAULT_CAMERA_FOVY_DEG), resolution[0] / resolution[1], 0.1, 100)

    # Set up the canonical landmarks and landmark weights.
    canonical_face_indices = CANONICAL_FACE_WEIGHTS.keys()
    canonical_face_landmarks = np.array([CANONICAL_FACE_VERTICES[i] for i in canonical_face_indices])
    canonical_face_weights = np.array(list(CANONICAL_FACE_WEIGHTS.values()))
    canonical_hand_indices = {
        "Left": CANONICAL_LEFTHAND_WEIGHTS.keys(),
        "Right": CANONICAL_RIGHTHAND_WEIGHTS.keys()
    }
    canonical_hand_landmarks = {
        "Left": np.array([CANONICAL_LEFTHAND_VERTICES[i] for i in canonical_hand_indices["Left"]]),
        "Right": np.array([CANONICAL_RIGHTHAND_VERTICES[i] for i in canonical_hand_indices["Right"]])
    }
    canonical_hand_weights = {
        "Left": np.array(list(CANONICAL_LEFTHAND_WEIGHTS.values())),
        "Right": np.array(list(CANONICAL_RIGHTHAND_WEIGHTS.values()))
    }

    print("Press ESC to exit, or \"b\" to show blendshape weights.")
    while cam.isOpened():
        # Read the next video frame (or break out of the loop if we didn't get one).
        success, frame = cam.read()
        if not success: break

        # Update timing info. This is mostly for debug purposes, since the output video will have a fixed framerate.
        current_time_ns = time.monotonic_ns()
        elapsed_ms = int((current_time_ns - last_time_ns) // 1e6)
        timestamp_ms = int(1000 * frame_number / fps)
        last_time_ns = current_time_ns

        # Print some timing info every 10 frames just to avoid spamming the console too much.
        if frame_number % 10 == 0: print(f"Frame {frame_number}, frame time {elapsed_ms}ms, expected {int(1000 / fps)}ms.")

        # Convert to a mediapipe image and run through face+hand detection (if enabled).
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        face_detector_result = face_detector.detect_for_video(mp_image, timestamp_ms) if face_detector is not None else None
        hand_detector_result = hand_detector.detect_for_video(mp_image, timestamp_ms) if hand_detector is not None else None

        # Create an image to annotate with our results (either a black image, or a copy of the original).
        # Using the original image is a much heavier operation for OpenCV.
        annotated_image = np.copy(mp_image.numpy_view()) if DRAW_ORIGINAL_IMAGE else np.zeros_like(mp_image.numpy_view())

        # Process and draw face landmark data, if we got any.
        if face_detector_result is not None and len(face_detector_result.face_landmarks) > 0:
            landmarks = face_detector_result.face_landmarks[0]
            runtime_screen_landmarks = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in canonical_face_indices])
            procrustes_result, runtime_metric_landmarks = convert(runtime_screen_landmarks, canonical_face_landmarks, canonical_face_weights, projection_transform)
            
            # Draw both the full face landmarks/connections, and also just the set of points from the "canonical landmarks".
            if DRAW_FACE_LANDMARKS:
                draw_face_landmarks(annotated_image, face_detector_result)
                draw_landmarks(annotated_image, runtime_metric_landmarks, procrustes_result["transform"], projection_transform, (0, 255, 0))
            if DRAW_FACE_TRANFORM: draw_gizmo(annotated_image, procrustes_result["transform"], projection_transform)

            # If enabled, draw the face transform that mediapipe gives us.
            # The main difference between this result and ours is just the camera FOV (defined at top of file).
            if len(face_detector_result.facial_transformation_matrixes) > 0:
                draw_gizmo(annotated_image, face_detector_result.facial_transformation_matrixes[0], projection_transform, brightness=0.5)

        # Process and draw hand landmark data, if we got any.
        if hand_detector_result is not None and len(hand_detector_result.handedness) > 0:
            for i in range(len(hand_detector_result.handedness)):
                landmarks = hand_detector_result.hand_landmarks[i]
                world = hand_detector_result.hand_world_landmarks[i]
                hand = hand_detector_result.handedness[i][0].category_name
                assert (hand == "Left" or hand == "Right")
                # runtime_screen_landmarks = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in canonical_hand_indices[hand]])
                runtime_screen_landmarks = np.array([[p.x, p.y, p.z] for p in landmarks])
                canonical_metric_landmarks = rotate_hand_world_landmarks(np.array([[p.x, p.y, p.z] for p in world]) * 100, hand)
                canonical_weights = np.ones(len(canonical_metric_landmarks))
                procrustes_result, runtime_metric_landmarks = convert(runtime_screen_landmarks, canonical_metric_landmarks, canonical_weights, projection_transform)

                # Draw both the full hand landmarks/connections, and also just the set of points from the "canonical landmarks".
                if DRAW_HAND_LANDMARKS:
                    draw_hand_landmarks(annotated_image, landmarks, hand)
                    draw_landmarks(annotated_image, runtime_metric_landmarks, procrustes_result["transform"], projection_transform, (0, 255, 0))
                if DRAW_HAND_TRANSFORM: draw_gizmo(annotated_image, procrustes_result["transform"], projection_transform)

            if DRAW_HAND_CANONICAL_MODEL:
                landmarks = hand_detector_result.hand_world_landmarks[0]
                hand = hand_detector_result.handedness[0][0].category_name
                assert (hand == "Left" or hand == "Right")
                view_transform = np.eye(4)
                view_transform[:3, 3] = np.array([0, 0, -100])

                points = np.array([[p.x, p.y, p.z] for p in landmarks]) * 100
                rotated = rotate_hand_world_landmarks(points, hand)
                ndc = transform_points(rotated, projection_transform @ view_transform, True)

                # Technically this is unnecessarily truncating to integer screen coordinates.
                # This loses us resolution when we re-project.
                screen = ndc_to_screen_points(ndc, (annotated_image.shape[1], annotated_image.shape[0]))

                class FakeLandmark:
                    def __init__(self, x, y):
                        self.x = x
                        self.y = y

                unrotated_world = [FakeLandmark(p[0] / annotated_image.shape[1], p[1] / annotated_image.shape[0]) for p in screen]
                draw_hand_landmarks(annotated_image, unrotated_world, hand)

        # Write our annotated video frame to the output stream.
        if video_writer is not None: video_writer.write(annotated_image)

        # Try to show the image window.
        key_pressed = show_image(annotated_image, WINDOW_NAME)

        # If the user closed the window or pressed "ESC", exit the loop.
        if (not is_window_open(WINDOW_NAME)) or (key_pressed == ord('\x1b')): break # '\x1b' is ESC key

        # If the user pressed "b" (and the face model is running), show the blendshapes plot.
        if (key_pressed == ord('b')) and face_detector_result is not None and len(face_detector_result.face_blendshapes) > 0:
            plot_face_blendshapes_bar_graph(face_detector_result.face_blendshapes[0])

        frame_number += 1

    cv2.destroyAllWindows()
    if video_writer is not None: video_writer.release()
    cam.release()

if __name__ == "__main__":
    np.set_printoptions(suppress=True) # This just prevents numpy from printing everything in scientific notation.
    main()
