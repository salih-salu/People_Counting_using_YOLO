import builtins
import os
import re
import runpy
import threading
import cv2
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(BASE_DIR, "people_counter.py")
MODEL_DIR = os.path.join(BASE_DIR, "model")
VIDEO_DIR = os.path.join(BASE_DIR, "video")

COUNT_RE = re.compile(r"A->B[=:]\s*(\d+).*?B->A[=:]\s*(\d+)")

lock = threading.Lock()
stop_event = threading.Event()
frame_event = threading.Event()
latest_jpeg = None


def list_videos():
    if not os.path.isdir(VIDEO_DIR):
        return []
    exts = (".mp4", ".avi", ".mov", ".mkv")
    return sorted(f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(exts))


def list_models():
    if not os.path.isdir(MODEL_DIR):
        return []
    return sorted(f for f in os.listdir(MODEL_DIR) if f.lower().endswith(".pt"))


state = {
    "running": False,
    "count_ab": 0,
    "count_ba": 0,
    "video": (list_videos() or ["peoples.mp4"])[0],
    "model": (list_models() or ["yolo11n.pt"])[0],
}


def encode_frame(frame):
    h, w = frame.shape[:2]
    if w > 960:
        frame = cv2.resize(frame, (960, int(h * 960 / w)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None


def get_preview_jpeg(video_name):
    path = os.path.join(VIDEO_DIR, video_name)
    if not os.path.isfile(path):
        return b""
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    return encode_frame(frame) if ok else b""


def run_counter(selected_video, selected_model):
    global latest_jpeg
    import ultralytics
    orig_named = cv2.namedWindow
    orig_resize = cv2.resizeWindow
    orig_imshow = cv2.imshow
    orig_wait = cv2.waitKey
    orig_destroy = cv2.destroyAllWindows
    orig_cap = cv2.VideoCapture
    orig_puttext = cv2.putText
    orig_yolo = ultralytics.YOLO

    video_path = os.path.join(VIDEO_DIR, selected_video)
    model_path = os.path.join(MODEL_DIR, selected_model)

    def web_imshow(_name, frame):
        global latest_jpeg
        jpg = encode_frame(frame)
        if jpg is not None:
            with lock:
                latest_jpeg = jpg
            frame_event.set()

    def web_wait(_delay=1):
        return ord("q") if stop_event.is_set() else -1

    def web_print(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        m = COUNT_RE.search(text)
        if m:
            with lock:
                state["count_ab"] = int(m.group(1))
                state["count_ba"] = int(m.group(2))
        builtins.print(*args, **kwargs)

    def web_puttext(img, text, org, *args, **kwargs):
        if text == "A":
            org = (org[0], org[1] + 25)
        return orig_puttext(img, text, org, *args, **kwargs)

    cv2.namedWindow = lambda *a, **k: None
    cv2.resizeWindow = lambda *a, **k: None
    cv2.destroyAllWindows = lambda *a, **k: None
    cv2.imshow = web_imshow
    cv2.waitKey = web_wait
    cv2.putText = web_puttext
    cv2.VideoCapture = lambda *a, **k: orig_cap(video_path)
    ultralytics.YOLO = lambda *a, **k: orig_yolo(model_path)

    try:
        os.chdir(BASE_DIR)
        runpy.run_path(SCRIPT_PATH, init_globals={"print": web_print})
    finally:
        cv2.namedWindow = orig_named
        cv2.resizeWindow = orig_resize
        cv2.imshow = orig_imshow
        cv2.waitKey = orig_wait
        cv2.destroyAllWindows = orig_destroy
        cv2.putText = orig_puttext
        cv2.VideoCapture = orig_cap
        ultralytics.YOLO = orig_yolo
        with lock:
            state["running"] = False
        frame_event.set()


@app.route("/")
def index():
    return render_template(
        "index.html",
        videos=list_videos(),
        models=list_models(),
        selected_video=state["video"],
        selected_model=state["model"],
    )


@app.route("/preview.jpg")
def preview():
    videos = list_videos()
    video_name = request.args.get("video", state["video"])
    if video_name not in videos and videos:
        video_name = videos[0]
    return Response(get_preview_jpeg(video_name), mimetype="image/jpeg")


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with lock:
                running = state["running"]
                jpg = latest_jpeg
            if jpg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
            if not running:
                break
            frame_event.wait(timeout=0.5)
            frame_event.clear()

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start", methods=["POST"])
def start():
    global latest_jpeg
    data = request.get_json(silent=True) or {}
    videos = list_videos()
    models = list_models()
    selected_video = data.get("video", state["video"])
    selected_model = data.get("model", state["model"])

    if selected_video not in videos or selected_model not in models:
        return jsonify({"ok": False, "error": "Invalid video or model"}), 400

    with lock:
        if state["running"]:
            return jsonify({"ok": False})
        state["running"] = True
        state["count_ab"] = 0
        state["count_ba"] = 0
        state["video"] = selected_video
        state["model"] = selected_model
        latest_jpeg = None

    stop_event.clear()
    frame_event.clear()
    threading.Thread(
        target=run_counter, args=(selected_video, selected_model), daemon=True
    ).start()
    return jsonify({"ok": True})


@app.route("/stop", methods=["POST"])
def stop():
    stop_event.set()
    return jsonify({"ok": True})


@app.route("/status")
def status():
    with lock:
        return jsonify({
            "running": state["running"],
            "count_ab": state["count_ab"],
            "count_ba": state["count_ba"],
            "total": state["count_ab"] + state["count_ba"],
        })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
