# Real-Time People Counting using YOLO

Detects and tracks people in a video with YOLO and counts how many cross a virtual line in both directions (A → B and B → A). Results are shown live on a Flask dashboard.

## Tech Stack
Python, YOLO (Ultralytics), OpenCV, Flask

## Workflow
1. **Read video**: OpenCV reads the video frame by frame.
2. **Detect**: YOLO finds the people in each frame.
3. **Track**: ByteTrack gives each person a unique ID that stays the same across frames.
4. **Check the line**: the feet position of each person is checked against the virtual line.
5. **Count**: when a person moves from one side to the other, the A → B or B → A counter goes up once.
6. **Display**: Flask streams the video with boxes, IDs and counters to the browser.

## Run
```bash
pip install ultralytics opencv-python lapx flask
python app.py
```
Open http://127.0.0.1:5000 in your browser.
