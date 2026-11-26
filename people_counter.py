import time
import cv2
import torch
from ultralytics import YOLO

VIDEO_PATH   = 'peoples.mp4'        
MODEL_PATH   = 'yolo11n.pt'          
IMGSZ        = 960                  # detection size: 640 fast, 960 more accurate for small/close people
PROC_W       = 2000                 # frames wider than this are shrunk before detection
FRAME_SKIP   = 2                    # 1 = process every frame, 2 = every 2nd frame, 3 = every 3rd ...
CONF         = 0.30                 # detection confidence
IOU          = 0.70                 # NMS overlap threshold (higher keeps two close people as 2 boxes)
TRACKER      = 'bytetrack.yaml'     
DEDUP_IOU    = 0.85                 # boxes overlapping more than this is same person, keep best one
LINE_FRAC    = (0.00, 0.70, 1.00, 0.70)   # Fix the line position
SELECT_LINE  = False                # True = click 2 points on the first frame to set the line
MARGIN       = 8                    
COOLDOWN     = 30                   # PROCESSED frames before the same ID can be counted again
SAVE_OUTPUT  = None                 
DISPLAY_W    = 2000                  # on-screen window width (lower it if bigger than your screen)


USE_GPU = torch.cuda.is_available()
DEVICE  = 0 if USE_GPU else 'cpu'
HALF    = USE_GPU                   
print(f'Running on: {"GPU" if USE_GPU else "CPU"} | model={MODEL_PATH} imgsz={IMGSZ} skip={FRAME_SKIP}')

model = YOLO(MODEL_PATH)
video = cv2.VideoCapture(VIDEO_PATH)
succ, first = video.read()
if not succ:
    raise SystemExit('Cannot open video: ' + str(VIDEO_PATH))
if isinstance(VIDEO_PATH, str):
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
src_fps = video.get(cv2.CAP_PROP_FPS) or 25


def shrink(frame):
    h, w = frame.shape[:2]
    if w > PROC_W:
        return cv2.resize(frame, (PROC_W, int(h * PROC_W / w)))
    return frame


first = shrink(first)
H, W = first.shape[:2]                     
DISPLAY_H = int(H * DISPLAY_W / W)


if SELECT_LINE:
    pts = []
    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))
    cv2.namedWindow('Select line', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Select line', DISPLAY_W, DISPLAY_H)
    cv2.setMouseCallback('Select line', on_click)
    while True:
        view = first.copy()
        for p in pts:
            cv2.circle(view, p, 5, (0, 255, 255), -1)
        if len(pts) == 2:
            cv2.line(view, pts[0], pts[1], (0, 255, 255), 2)
            cv2.putText(view, 'Press any key to start', (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Select line', view)
        key = cv2.waitKey(30)
        if key != -1 and len(pts) == 2:
            break
    cv2.destroyWindow('Select line')
    (x1, y1), (x2, y2) = pts
else:
    x1, y1, x2, y2 = LINE_FRAC[0]*W, LINE_FRAC[1]*H, LINE_FRAC[2]*W, LINE_FRAC[3]*H

dx, dy = x2 - x1, y2 - y1
LINE_LEN = (dx*dx + dy*dy) ** 0.5
nx, ny = -dy / LINE_LEN, dx / LINE_LEN          


def signed_distance(px, py):
    return (dx * (py - y1) - dy * (px - x1)) / LINE_LEN


def inside_segment(px, py):
    t = ((px - x1) * dx + (py - y1) * dy) / (LINE_LEN ** 2)
    return 0.0 <= t <= 1.0


def remove_duplicate_boxes(boxes, ids, confs, thr):
    keep = []
    for i in range(len(boxes)):
        dup = False
        ax1, ay1, ax2, ay2 = boxes[i]
        for j in range(len(boxes)):
            if i == j:
                continue
            bx1_, by1_, bx2_, by2_ = boxes[j]
            iw = max(0.0, min(ax2, bx2_) - max(ax1, bx1_))
            ih = max(0.0, min(ay2, by2_) - max(ay1, by1_))
            inter = iw * ih
            union = (ax2 - ax1) * (ay2 - ay1) + (bx2_ - bx1_) * (by2_ - by1_) - inter
            if union > 0 and inter / union > thr:
                if confs[j] > confs[i] or (confs[j] == confs[i] and j < i):
                    dup = True
                    break
        if not dup:
            keep.append(i)
    return boxes[keep], [ids[i] for i in keep], [confs[i] for i in keep]



last_side = {}        
last_counted = {}     
trails = {}          
count_ab = 0
count_ba = 0
flash = 0
frame_no = 0
fps_show, t_prev = 0.0, time.time()

writer = None
if SAVE_OUTPUT:
    writer = cv2.VideoWriter(SAVE_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), src_fps / FRAME_SKIP, (W, H))


def id_color(i):
    return (int(37 * i) % 255, int(91 * i + 80) % 255, int(53 * i + 160) % 255)


cv2.namedWindow('People counter', cv2.WINDOW_NORMAL)
cv2.resizeWindow('People counter', DISPLAY_W, DISPLAY_H)


while True:
    succ, img = video.read()
    if succ == False:
        break

    for _ in range(FRAME_SKIP - 1):
        video.grab()

    frame_no += 1
    img = shrink(img)

    results = model.track(img, persist=True, classes=[0], conf=CONF, iou=IOU,
                          imgsz=IMGSZ, tracker=TRACKER, device=DEVICE, half=HALF,
                          verbose=False)
    result = results[0]

    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids   = result.boxes.id.int().cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()
        boxes, ids, confs = remove_duplicate_boxes(boxes, ids, confs, DEDUP_IOU)

        for (bx1, by1, bx2, by2), tid, cf in zip(boxes, ids, confs):
            foot_x, foot_y = (bx1 + bx2) / 2, by2          
            d = signed_distance(foot_x, foot_y)

            if abs(d) >= MARGIN:
                side = 1 if d > 0 else -1
                prev = last_side.get(tid)
                last_side[tid] = side
                if (prev is not None and prev != side
                        and inside_segment(foot_x, foot_y)
                        and frame_no - last_counted.get(tid, -10**9) >= COOLDOWN):
                    last_counted[tid] = frame_no
                    flash = 4
                    if prev == 1 and side == -1:
                        count_ab += 1
                        print(f'[frame {frame_no}] ID {tid}: A -> B   (A->B={count_ab}, B->A={count_ba})')
                    else:
                        count_ba += 1
                        print(f'[frame {frame_no}] ID {tid}: B -> A   (A->B={count_ab}, B->A={count_ba})')

            # drawing: box, ID 
            col = id_color(tid)
            cv2.rectangle(img, (int(bx1), int(by1)), (int(bx2), int(by2)), col, 2)
            label = f'ID {tid}  {cf:.2f}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(img, (int(bx1), int(by1) - th - 8), (int(bx1) + tw + 6, int(by1)), col, -1)
            cv2.putText(img, label, (int(bx1) + 3, int(by1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.circle(img, (int(foot_x), int(foot_y)), 4, col, -1)
            trail = trails.setdefault(tid, [])
            trail.append((int(foot_x), int(foot_y)))
            trails[tid] = trail[-15:]
            for a, b in zip(trails[tid], trails[tid][1:]):
                cv2.line(img, a, b, col, 2)

    
    line_col = (0, 0, 255) if flash > 0 else (0, 255, 255)
    flash = max(0, flash - 1)
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), line_col, 3)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    cv2.putText(img, 'A', (int(mx + nx * 45) - 10, int(my + ny * 45) + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 2)
    cv2.putText(img, 'B', (int(mx - nx * 45) - 10, int(my - ny * 45) + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 128, 0), 2)
    cv2.rectangle(img, (10, 10), (260, 105), (0, 0, 0), -1)
    cv2.putText(img, f'A -> B : {count_ab}', (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img, f'B -> A : {count_ba}', (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)

    now = time.time()
    fps_show = 0.9 * fps_show + 0.1 * (1.0 / max(now - t_prev, 1e-6))   # smoothed FPS
    t_prev = now
    cv2.putText(img, f'FPS {fps_show:.1f}', (W - 140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if writer:
        writer.write(img)
    cv2.imshow('People counter', cv2.resize(img, (DISPLAY_W, DISPLAY_H)) if DISPLAY_W != W else img)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

video.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
print(f'\nFINAL  A->B: {count_ab}   B->A: {count_ba}')