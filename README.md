To Use:

1. You'll need python (at least 3.11) and [poetry](https://python-poetry.org/docs/). You might just be able to `pip install poetry`.
2. Navigate to the repo and run `poetry install`. Hope that it works.
3. You'll need to separately download the task files for mediapipe's [face landmark](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python) and [hand landmark](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python) models. They should be in the same directory as `main.py` and be called `face_landmarker.task` and `hand_landmarker.task`. Try the following:
    * `wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
    * `wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
4. To run, navigate to the repo and run `poetry run python main.py`. It should use the default webcam. Press ESC to quit or "b" to show a plot of blendshape weights.
