import os
import sys
import cv2
import glob
import shutil
import subprocess
import threading
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

# ── Constants ──────────────────────────────────────────────────────────────────

CLASS_NAMES     = ["cone", "non-cone"]
DIRS = {
    "train": ("data/images/train", "data/labels/train"),
    "val":   ("data/images/val",   "data/labels/val"),
}
IMG_EXTS        = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
TRAIN_SCRIPT    = "train.py"
TRAIN_OUTPUT_DIR = "runs/train"

COL_CONFIG = [
    (0, 5, 260),  # Image
    (1, 1, 70),   # Set
    (2, 2, 120),  # Status
    (3, 1, 90),   # Redraw
    (4, 1, 90),   # Visualise
    (5, 1, 95),   # Transfer
    (6, 1, 80),   # Delete
]

# ── Palette ────────────────────────────────────────────────────────────────────

PANEL_BG   = "#ffffff"
HDR_BG     = "#f0f4f8"
TEXT_FG    = "#1a1a1a"
GREEN_FG   = "#1a7f5a"
AMBER_FG   = "#b45309"
MUTED      = "#6b7280"
DIVIDER    = "#e5e7eb"
ROW_ALT    = "#f9fafb"
BTN_BG     = "#1a56db"
BTN_FG     = "#ffffff"
BTN_ACTIVE = "#1648c0"

# ── Helpers ────────────────────────────────────────────────────────────────────

def _screen_size():
    try:
        r = tk.Tk()
        w, h = r.winfo_screenwidth(), r.winfo_screenheight()
        r.destroy()
    except Exception:
        w, h = 1920, 1080
    return w, h


def _next_image_number():
    max_n = 0
    for img_dir, _ in DIRS.values():
        for ext in IMG_EXTS:
            for p in (glob.glob(os.path.join(img_dir, f"*{ext}")) +
                      glob.glob(os.path.join(img_dir, f"*{ext.upper()}"))):
                stem = os.path.splitext(os.path.basename(p))[0]
                num_str = stem.replace("image", "").replace(" ", "").replace("_", "").strip()
                try:
                    max_n = max(max_n, int(num_str))
                except ValueError:
                    pass
    return max_n + 1


def get_label_path(img_path, label_dir):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(label_dir, stem + ".txt")


def has_label(img_path, label_dir):
    return os.path.isfile(get_label_path(img_path, label_dir))


def collect_images():
    items = []
    for set_name, (img_dir, lbl_dir) in DIRS.items():
        os.makedirs(lbl_dir, exist_ok=True)
        paths = set()
        for ext in IMG_EXTS:
            for p in glob.glob(os.path.join(img_dir, f"*{ext}")):
                paths.add(p)
            for p in glob.glob(os.path.join(img_dir, f"*{ext.upper()}")):
                paths.add(p)
        for p in sorted(paths):
            items.append((p, lbl_dir, set_name))
    return items


def _save_yolo_labels(label_path, boxes, img_w, img_h):
    with open(label_path, "w") as f:
        for x1, y1, x2, y2, cls in boxes:
            xc = (x1 + x2) / 2 / img_w
            yc = (y1 + y2) / 2 / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def _import_images(paths, target_set):
    target_dir = DIRS[target_set][0]
    os.makedirs(target_dir, exist_ok=True)
    next_n = _next_image_number()
    for src in paths:
        dst = os.path.join(target_dir, f"image{next_n}.jpg")
        img = cv2.imread(src)
        if img is not None:
            cv2.imwrite(dst, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            shutil.copy2(src, dst)
        next_n += 1


# ── OpenCV labeling ────────────────────────────────────────────────────────────

def _draw_frame(st):
    img_disp = st["img"].copy()
    h, w = img_disp.shape[:2]
    if st["scale"] != 1.0:
        img_disp = cv2.resize(img_disp, (int(w * st["scale"]), int(h * st["scale"])),
                              interpolation=cv2.INTER_AREA)
    for box in st["bboxes"]:
        bs = [int(b * st["scale"]) for b in box[:4]]
        color = (0, 165, 255) if box[4] == 0 else (0, 255, 0)
        cv2.rectangle(img_disp, (bs[0], bs[1]), (bs[2], bs[3]), color, 2)
    if st["bbox_start"] and st["bbox_end"]:
        bs = [int(b * st["scale"]) for b in (*st["bbox_start"], *st["bbox_end"])]
        color = (0, 165, 255) if st["class_id"] == 0 else (0, 255, 0)
        cv2.rectangle(img_disp, (bs[0], bs[1]), (bs[2], bs[3]), color, 2)
    cv2.imshow("Label Tool", img_disp)


def _mouse_callback(event, x, y, flags, st):
    def to_orig(px, py):
        return (int(px / st["scale"]), int(py / st["scale"]))

    if event == cv2.EVENT_LBUTTONDOWN:
        st["bbox_start"] = to_orig(x, y)
        st["bbox_end"] = None
    elif event == cv2.EVENT_MOUSEMOVE and st["bbox_start"]:
        st["bbox_end"] = to_orig(x, y)
    elif event == cv2.EVENT_LBUTTONUP and st["bbox_start"]:
        st["bbox_end"] = to_orig(x, y)
        x1, y1 = st["bbox_start"]
        x2, y2 = st["bbox_end"]
        st["bboxes"].append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), st["class_id"]))
        st["bbox_start"] = None
        st["bbox_end"] = None
    _draw_frame(st)


def label_image(img_path, label_dir):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {img_path}")
        return True

    screen_w, screen_h = _screen_size()
    h, w = img.shape[:2]

    st = {
        "img":        img,
        "scale":      min(screen_w / w, screen_h / h, 1.0),
        "bboxes":     [],
        "bbox_start": None,
        "bbox_end":   None,
        "class_id":   0,
    }

    cv2.namedWindow("Label Tool", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Label Tool", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty("Label Tool", cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback("Label Tool", _mouse_callback, st)
    _draw_frame(st)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            _save_yolo_labels(get_label_path(img_path, label_dir), st["bboxes"], w, h)
            print(f"Saved: {get_label_path(img_path, label_dir)}")
            break
        elif key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return False
        elif key == ord("c"):
            st["bboxes"] = []
            _draw_frame(st)
        elif key in (ord("0"), ord("1")):
            st["class_id"] = key - ord("0")
            st["bbox_start"] = None
            st["bbox_end"] = None
            _draw_frame(st)

    cv2.destroyAllWindows()
    return True


def visualise_image(img_path, label_dir):
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]
    label_path = get_label_path(img_path, label_dir)
    if os.path.isfile(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, xc, yc, bw, bh = map(float, parts)
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                color = (0, 165, 255) if int(cls_id) == 0 else (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = (CLASS_NAMES[int(cls_id)]
                         if int(cls_id) < len(CLASS_NAMES) else str(int(cls_id)))
                cv2.putText(img, label, (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    sw, sh = _screen_size()
    ih, iw = img.shape[:2]
    vis_scale = min(sw / iw, sh / ih, 1.0)
    if vis_scale != 1.0:
        img = cv2.resize(img, (int(iw * vis_scale), int(ih * vis_scale)),
                         interpolation=cv2.INTER_AREA)

    cv2.namedWindow("Visualise Labels", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Visualise Labels", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Visualise Labels", img)
    cv2.waitKey(0)
    cv2.destroyWindow("Visualise Labels")


# ── Video annotator ────────────────────────────────────────────────────────────

def _open_video_annotator(video_path, target_set="train"):
    WINDOW_NAME = "Video Frame Inserter"
    image_dir, label_dir = DIRS[target_set]
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0

    screen_w, screen_h = _screen_size()

    st = {
        "frame_idx":    0,
        "frame":        None,
        "next_num":     _next_image_number(),
        "saved_count":  0,
        "frame_cache":  {},
        "boxes":        [],
        "class_id":     0,
        "drag_start":   None,
        "drag_end":     None,
        "scale":        1.0,
        "trackbar_busy": False,
    }

    def load_frame(idx):
        idx = max(0, min(idx, total_frames - 1))
        if idx in st["frame_cache"]:
            st["frame_idx"] = idx
            st["frame"] = st["frame_cache"][idx].copy()
            st["boxes"].clear()
            st["drag_start"] = st["drag_end"] = None
            return True
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            return False
        st["frame_idx"] = idx
        st["frame"] = frame.copy()
        st["frame_cache"][idx] = frame.copy()
        st["boxes"].clear()
        st["drag_start"] = st["drag_end"] = None
        if len(st["frame_cache"]) > 200:
            st["frame_cache"].pop(min(st["frame_cache"].keys()), None)
        return True

    def load_next_frame():
        if st["frame_idx"] >= total_frames - 1:
            return False
        ok, frame = cap.read()
        if not ok:
            return False
        st["frame_idx"] += 1
        st["frame"] = frame.copy()
        st["frame_cache"][st["frame_idx"]] = frame.copy()
        st["boxes"].clear()
        st["drag_start"] = st["drag_end"] = None
        if len(st["frame_cache"]) > 200:
            st["frame_cache"].pop(min(st["frame_cache"].keys()), None)
        return True

    def to_original(x, y):
        return int(x / st["scale"]), int(y / st["scale"])

    def redraw(status_text=""):
        if st["frame"] is None:
            return
        img = st["frame"].copy()
        h, w = img.shape[:2]
        st["scale"] = min(screen_w / w, screen_h / h, 1.0)
        disp = (cv2.resize(img, (int(w * st["scale"]), int(h * st["scale"])),
                           interpolation=cv2.INTER_NEAREST)
                if st["scale"] != 1.0 else img)

        for x1, y1, x2, y2, cls in st["boxes"]:
            color = (0, 165, 255) if cls == 0 else (0, 255, 0)
            cv2.rectangle(disp,
                          (int(x1 * st["scale"]), int(y1 * st["scale"])),
                          (int(x2 * st["scale"]), int(y2 * st["scale"])), color, 2)

        if st["drag_start"] and st["drag_end"]:
            x1, y1 = st["drag_start"]
            x2, y2 = st["drag_end"]
            color = (0, 165, 255) if st["class_id"] == 0 else (0, 255, 0)
            cv2.rectangle(disp,
                          (int(x1 * st["scale"]), int(y1 * st["scale"])),
                          (int(x2 * st["scale"]), int(y2 * st["scale"])), color, 2)

        cv2.putText(disp,
                    f"Frame {st['frame_idx'] + 1}/{total_frames} | "
                    f"Class {st['class_id']} ({CLASS_NAMES[st['class_id']]})",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(disp,
                    "drag=box  0/1=class  c=clear  a/d=prev/next  s=save  Esc/q=quit",
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1)
        if status_text:
            cv2.putText(disp, status_text, (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 255, 140), 2)
        cv2.imshow(WINDOW_NAME, disp)

    def on_trackbar(pos):
        if not st["trackbar_busy"]:
            load_frame(pos)
            redraw()

    def on_mouse(event, x, y, flags, param):
        if st["frame"] is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            st["drag_start"] = to_original(x, y)
            st["drag_end"] = None
        elif event == cv2.EVENT_MOUSEMOVE and st["drag_start"]:
            st["drag_end"] = to_original(x, y)
        elif event == cv2.EVENT_LBUTTONUP and st["drag_start"]:
            st["drag_end"] = to_original(x, y)
            x1, y1 = st["drag_start"]
            x2, y2 = st["drag_end"]
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            if x_max - x_min > 2 and y_max - y_min > 2:
                st["boxes"].append((x_min, y_min, x_max, y_max, st["class_id"]))
            st["drag_start"] = st["drag_end"] = None
        redraw()

    load_frame(0)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    cv2.createTrackbar("Frame", WINDOW_NAME, 0, total_frames - 1, on_trackbar)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    redraw()

    status = ""
    while True:
        st["trackbar_busy"] = True
        cv2.setTrackbarPos("Frame", WINDOW_NAME, st["frame_idx"])
        st["trackbar_busy"] = False
        if status:
            redraw(status)
            status = ""

        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue
        if key in (ord("q"), 27):
            break
        if key in (ord("0"), ord("1")):
            st["class_id"] = key - ord("0"); redraw(); continue
        if key == ord("c"):
            st["boxes"].clear(); redraw(); continue
        if key == ord("a"):
            load_frame(st["frame_idx"] - 1); redraw(); continue
        if key == ord("d"):
            load_next_frame(); redraw(); continue
        if key == ord("s"):
            if st["frame"] is None:
                continue
            out_img = os.path.join(image_dir, f"image{st['next_num']}.bmp")
            out_lbl = os.path.join(label_dir,  f"image{st['next_num']}.txt")
            cv2.imwrite(out_img, st["frame"])
            h, w = st["frame"].shape[:2]
            _save_yolo_labels(out_lbl, st["boxes"], w, h)
            st["saved_count"] += 1
            status = f"Saved image{st['next_num']}.bmp ({w}x{h}, {len(st['boxes'])} boxes)"
            st["next_num"] += 1

    cap.release()
    cv2.destroyWindow(WINDOW_NAME)
    return st["saved_count"]


# ── Tkinter widget helpers ─────────────────────────────────────────────────────

def _clear_root(root):
    for w in root.winfo_children():
        w.destroy()


def _apply_col_config(frame):
    for col, weight, minsize in COL_CONFIG:
        frame.columnconfigure(col, weight=weight, minsize=minsize)


def _btn(parent, text, command, **kwargs):
    defaults = dict(
        bg=BTN_BG, fg=BTN_FG, font=("Segoe UI", 10, "bold"),
        relief="flat", padx=14, pady=6, cursor="hand2",
        activebackground=BTN_ACTIVE, activeforeground=BTN_FG,
    )
    defaults.update(kwargs)
    return tk.Button(parent, text=text, command=command, **defaults)


def _secondary_btn(parent, text, command, **kwargs):
    defaults = dict(
        bg=HDR_BG, fg=TEXT_FG, font=("Segoe UI", 10),
        relief="flat", padx=14, pady=6, cursor="hand2",
        activebackground=DIVIDER, activeforeground=TEXT_FG,
    )
    defaults.update(kwargs)
    return tk.Button(parent, text=text, command=command, **defaults)


def _row_btn(parent, row_bg, text, command):
    cell = tk.Frame(parent, bg=row_bg, pady=3)
    cell.columnconfigure(0, weight=1)
    tk.Button(cell, text=text, command=command,
              bg=BTN_BG, fg=BTN_FG, font=("Segoe UI", 9), relief="flat",
              padx=10, pady=2, cursor="hand2",
              activebackground=BTN_ACTIVE, activeforeground=BTN_FG
              ).grid(row=0, column=0)
    return cell


def _panel_header(root, title, subtitle, on_back):
    top = tk.Frame(root, bg=PANEL_BG, pady=12, padx=18)
    top.pack(fill="x")

    tk.Button(
        top, text="← Back", command=on_back,
        bg=HDR_BG, fg=TEXT_FG, font=("Segoe UI", 9), relief="flat",
        padx=10, pady=4, cursor="hand2",
        activebackground=DIVIDER, activeforeground=TEXT_FG,
    ).pack(side="left")

    title_block = tk.Frame(top, bg=PANEL_BG)
    title_block.pack(side="left", padx=16)
    tk.Label(title_block, text=title, bg=PANEL_BG, fg=TEXT_FG,
             font=("Segoe UI", 14, "bold")).pack(anchor="w")
    tk.Label(title_block, text=subtitle, bg=PANEL_BG, fg=MUTED,
             font=("Segoe UI", 9)).pack(anchor="w")

    tk.Frame(root, bg=DIVIDER, height=1).pack(fill="x")

    body = tk.Frame(root, bg=PANEL_BG, padx=28, pady=20)
    body.pack(fill="both", expand=True)
    return body


# ── Insert Image panel ─────────────────────────────────────────────────────────

def show_insert_image_panel(root, build_home):
    _clear_root(root)
    body = _panel_header(root, "Insert Images",
                         "Choose a destination set then select image files to import.",
                         on_back=build_home)

    tk.Label(body, text="Insert into", bg=PANEL_BG, fg=MUTED,
             font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 6))

    target_set = tk.StringVar(value="train")
    radio_row = tk.Frame(body, bg=PANEL_BG)
    radio_row.grid(row=1, column=0, sticky="w", pady=(0, 20))
    for val in ("train", "val"):
        tk.Radiobutton(
            radio_row, text=val, value=val, variable=target_set,
            bg=PANEL_BG, fg=TEXT_FG, selectcolor=HDR_BG,
            activebackground=PANEL_BG, activeforeground=TEXT_FG,
            highlightthickness=0, font=("Segoe UI", 10),
        ).pack(side="left", padx=(0, 16))

    status_var = tk.StringVar(value="")
    tk.Label(body, textvariable=status_var, bg=PANEL_BG, fg=GREEN_FG,
             font=("Segoe UI", 9)).grid(row=2, column=0, sticky="w", pady=(0, 16))

    btn_row = tk.Frame(body, bg=PANEL_BG)
    btn_row.grid(row=3, column=0, sticky="w")

    def do_insert():
        paths = filedialog.askopenfilenames(
            parent=root,
            title="Select images to insert",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        _import_images(paths, target_set.get())
        status_var.set(f"✓ Inserted {len(paths)} image(s) into '{target_set.get()}'.")

    _btn(btn_row, "Select & Insert Images…", do_insert).pack(side="left", padx=(0, 10))
    _secondary_btn(btn_row, "← Back to List", build_home).pack(side="left")


# ── Sample Video panel ─────────────────────────────────────────────────────────

def show_sample_video_panel(root, build_home):
    _clear_root(root)
    body = _panel_header(root, "Insert From Video",
                         "Choose a video, pick a target set, then launch the annotator.",
                         on_back=build_home)

    tk.Label(body, text="Video file", bg=PANEL_BG, fg=MUTED,
             font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 4))

    file_row = tk.Frame(body, bg=PANEL_BG)
    file_row.grid(row=1, column=0, sticky="ew", pady=(0, 20))
    file_row.columnconfigure(0, weight=1)

    video_var = tk.StringVar()
    tk.Entry(
        file_row, textvariable=video_var,
        bg=HDR_BG, fg=TEXT_FG, insertbackground=TEXT_FG,
        relief="flat", font=("Segoe UI", 9),
    ).grid(row=0, column=0, sticky="ew", ipady=5, padx=(0, 8))

    def browse():
        path = filedialog.askopenfilename(
            parent=root, title="Select a video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"),
                       ("All files", "*.*")],
        )
        if path:
            video_var.set(path)

    _btn(file_row, "Browse…", browse, font=("Segoe UI", 9)).grid(row=0, column=1)

    tk.Label(body, text="Insert into", bg=PANEL_BG, fg=MUTED,
             font=("Segoe UI", 9, "bold")).grid(row=2, column=0, sticky="w", pady=(0, 4))

    target_set = tk.StringVar(value="train")
    radio_row = tk.Frame(body, bg=PANEL_BG)
    radio_row.grid(row=3, column=0, sticky="w", pady=(0, 20))
    for val in ("train", "val"):
        tk.Radiobutton(
            radio_row, text=val, value=val, variable=target_set,
            bg=PANEL_BG, fg=TEXT_FG, selectcolor=HDR_BG,
            activebackground=PANEL_BG, activeforeground=TEXT_FG,
            highlightthickness=0, font=("Segoe UI", 10),
        ).pack(side="left", padx=(0, 16))

    tk.Label(
        body,
        text=("In the video window: drag to draw boxes, scrub with trackbar, "
              "0/1 switch class, C clears boxes, A/D prev/next frame, "
              "S saves frame + labels, Esc/Q exits."),
        bg=PANEL_BG, fg=MUTED, font=("Segoe UI", 9),
        wraplength=700, justify="left",
    ).grid(row=4, column=0, sticky="w", pady=(0, 20))

    status_var = tk.StringVar(value="")
    tk.Label(body, textvariable=status_var, bg=PANEL_BG, fg=GREEN_FG,
             font=("Segoe UI", 9)).grid(row=5, column=0, sticky="w", pady=(0, 16))

    btn_row = tk.Frame(body, bg=PANEL_BG)
    btn_row.grid(row=6, column=0, sticky="w")

    def launch():
        path = video_var.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", "Please select a valid video file.", parent=root)
            return
        saved = _open_video_annotator(path, target_set=target_set.get())
        status_var.set(f"✓ Done — {saved} frame(s) saved.")

    _btn(btn_row, "Open Annotator", launch).pack(side="left", padx=(0, 10))
    _secondary_btn(btn_row, "← Back to List", build_home).pack(side="left")


# ── Train Model panel ─────────────────────────────────────────────────────────

def show_train_panel(root, build_home):
    _clear_root(root)
    body = _panel_header(root, "Train Model",
                         "Runs train.py and streams output live. Log is saved to runs/train/.",
                         on_back=build_home)

    # Log output area
    log_box = scrolledtext.ScrolledText(
        body, bg="#1a1a2e", fg="#e2e8f0", font=("Courier New", 9),
        relief="flat", state="disabled", wrap="word",
    )
    log_box.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 14))
    body.rowconfigure(0, weight=1)
    body.columnconfigure(0, weight=1)

    status_var = tk.StringVar(value="")
    tk.Label(body, textvariable=status_var, bg=PANEL_BG, fg=MUTED,
             font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", pady=(0, 10))

    btn_row = tk.Frame(body, bg=PANEL_BG)
    btn_row.grid(row=2, column=0, sticky="w")

    train_btn = _btn(btn_row, "Start Training", None)
    train_btn.pack(side="left", padx=(0, 10))
    _secondary_btn(btn_row, "← Back to List", build_home).pack(side="left")

    def append_log(text):
        log_box.configure(state="normal")
        log_box.insert("end", text)
        log_box.see("end")
        log_box.configure(state="disabled")

    def run_training():
        if not os.path.isfile(TRAIN_SCRIPT):
            root.after(0, lambda: status_var.set(f"✗ {TRAIN_SCRIPT} not found."))
            root.after(0, lambda: train_btn.configure(state="normal"))
            return

        os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(TRAIN_OUTPUT_DIR, f"train_{timestamp}.log")

        root.after(0, lambda: append_log(f"Starting {TRAIN_SCRIPT}  [{timestamp}]\n"))
        root.after(0, lambda: append_log(f"Log → {log_path}\n{'─' * 60}\n"))

        try:
            proc = subprocess.Popen(
                [sys.executable, TRAIN_SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with open(log_path, "w") as log_file:
                for line in proc.stdout:
                    log_file.write(line)
                    log_file.flush()
                    root.after(0, lambda l=line: append_log(l))
            proc.wait()
            if proc.returncode == 0:
                msg = f"\n{'─' * 60}\n✓ Training complete (exit 0). Log saved to {log_path}\n"
                root.after(0, lambda: status_var.set(f"✓ Done — log saved to {log_path}"))
            else:
                msg = f"\n{'─' * 60}\n✗ Training exited with code {proc.returncode}. Log saved to {log_path}\n"
                root.after(0, lambda: status_var.set(f"✗ Exited with code {proc.returncode}"))
        except Exception as exc:
            msg = f"\n✗ Failed to start training: {exc}\n"
            root.after(0, lambda: status_var.set(f"✗ Error: {exc}"))

        root.after(0, lambda: append_log(msg))
        root.after(0, lambda: train_btn.configure(state="normal"))

    def start():
        train_btn.configure(state="disabled")
        status_var.set("Running…")
        log_box.configure(state="normal")
        log_box.delete("1.0", "end")
        log_box.configure(state="disabled")
        threading.Thread(target=run_training, daemon=True).start()

    train_btn.configure(command=start)


# ── Home screen ────────────────────────────────────────────────────────────────

def home_screen():
    items = collect_images()
    result = {"action": None, "images": []}

    root = tk.Tk()
    root.title("YOLO Label GUI")
    root.geometry("1200x700")
    root.minsize(1200, 600)
    root.configure(bg=PANEL_BG)

    def on_close():
        result["action"] = None
        root.quit()

    root.protocol("WM_DELETE_WINDOW", on_close)

    def build_home():
        _clear_root(root)

        nonlocal items
        items = collect_images()

        # Title row
        title_row = tk.Frame(root, bg=PANEL_BG, pady=10, padx=18)
        title_row.pack(fill="x")

        tk.Label(title_row, text="YOLO Labeling Tool", bg=PANEL_BG, fg=TEXT_FG,
                 font=("Segoe UI", 14, "bold")).pack(side="left")

        stats_label = tk.Label(title_row, bg=PANEL_BG, fg=MUTED, font=("Segoe UI", 9))
        stats_label.pack(side="left", padx=(12, 0))

        def update_stats():
            n = sum(1 for p, l, s in items if has_label(p, l))
            stats_label.config(text=f"{n} of {len(items)} images labeled")

        update_stats()

        def start_label(image_list):
            if not image_list:
                return
            result["action"] = "label"
            result["images"] = image_list
            root.quit()

        # Button row — canvas-backed so buttons never clip regardless of window width
        btn_canvas = tk.Canvas(root, bg=HDR_BG, height=42, highlightthickness=0)
        btn_canvas.pack(fill="x")
        btn_inner = tk.Frame(btn_canvas, bg=HDR_BG)
        btn_canvas.create_window((8, 4), window=btn_inner, anchor="nw")

        for text, cmd, color in [
            ("Insert Image",    lambda: show_insert_image_panel(root, build_home), BTN_BG),
            ("Sample Video",    lambda: show_sample_video_panel(root, build_home), BTN_BG),
            ("Label Unlabeled", lambda: start_label([(p, l, s) for p, l, s in items
                                                     if not has_label(p, l)]),     BTN_BG),
            ("Redraw All",      lambda: start_label(list(items)),                  BTN_BG),
            ("Train Model",     lambda: show_train_panel(root, build_home),        "#1a7f5a"),
        ]:
            tk.Button(btn_inner, text=text, command=cmd,
                      bg=color, fg=BTN_FG, font=("Segoe UI", 9, "bold"),
                      relief="flat", padx=10, pady=5, cursor="hand2",
                      activebackground=BTN_ACTIVE, activeforeground=BTN_FG
                      ).pack(side="left", padx=(0, 6))

        def _update_btn_scroll(e):
            btn_canvas.configure(scrollregion=btn_canvas.bbox("all"))
        btn_inner.bind("<Configure>", _update_btn_scroll)


        tk.Frame(root, bg=DIVIDER, height=1).pack(fill="x")

        outer = tk.Frame(root, bg=PANEL_BG)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, bg=PANEL_BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=PANEL_BG)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        _apply_col_config(inner)

        for col, text in enumerate(["Image", "Set", "Status", "Redraw",
                                     "Visualise", "Transfer", "Delete"]):
            tk.Label(inner, text=text, bg=HDR_BG, fg=MUTED,
                     font=("Segoe UI", 9, "bold"), anchor="center", pady=8
                     ).grid(row=0, column=col, sticky="ew", padx=1)

        for col in range(7):
            tk.Frame(inner, bg=DIVIDER, height=1).grid(row=1, column=col, sticky="ew")

        for idx, (img_path, label_dir, set_name) in enumerate(items):
            labeled = has_label(img_path, label_dir)
            row_bg = PANEL_BG if idx % 2 == 0 else ROW_ALT
            gr = idx + 2

            row_frame = tk.Frame(inner, bg=row_bg)
            row_frame.grid(row=gr, column=0, columnspan=7, sticky="ew")
            _apply_col_config(row_frame)

            tk.Label(row_frame, text=os.path.basename(img_path), bg=row_bg,
                     fg=TEXT_FG, font=("Segoe UI", 9), anchor="center", pady=7
                     ).grid(row=0, column=0, sticky="ew", padx=1)

            set_fg = GREEN_FG if set_name == "train" else AMBER_FG
            set_label = tk.Label(row_frame, text=set_name, bg=row_bg, fg=set_fg,
                                 font=("Segoe UI", 9, "bold"), anchor="center")
            set_label.grid(row=0, column=1, sticky="ew", padx=1)

            tk.Label(row_frame,
                     text="labeled" if labeled else "unlabeled",
                     bg=row_bg, fg=GREEN_FG if labeled else AMBER_FG,
                     font=("Segoe UI", 9), anchor="center"
                     ).grid(row=0, column=2, sticky="ew", padx=1)

            state = {"img_path": img_path, "label_dir": label_dir, "set_name": set_name}

            def make_redraw(s=state):
                def _redraw():
                    start_label([(s["img_path"], s["label_dir"], s["set_name"])])
                return _redraw

            _row_btn(row_frame, row_bg, "Redraw", make_redraw()
                     ).grid(row=0, column=3, sticky="ew", padx=1)

            vis_cell = tk.Frame(row_frame, bg=row_bg, pady=3)
            vis_cell.columnconfigure(0, weight=1)
            vis_cell.grid(row=0, column=4, sticky="ew", padx=1)
            if labeled:
                def make_visualise(s=state):
                    def _vis():
                        visualise_image(s["img_path"], s["label_dir"])
                    return _vis
                tk.Button(vis_cell, text="Visualise", command=make_visualise(),
                          bg=BTN_BG, fg=BTN_FG, font=("Segoe UI", 9), relief="flat",
                          padx=10, pady=2, cursor="hand2",
                          activebackground=BTN_ACTIVE, activeforeground=BTN_FG
                          ).grid(row=0, column=0)
            else:
                tk.Label(vis_cell, text="—", bg=row_bg, fg=MUTED,
                         font=("Segoe UI", 9), anchor="center"
                         ).grid(row=0, column=0, sticky="ew")

            def make_transfer(s=state, sl=set_label):
                def _transfer():
                    old_img_path  = s["img_path"]
                    old_label_dir = s["label_dir"]
                    old_set_name  = s["set_name"]
                    target_set = "val" if old_set_name == "train" else "train"
                    target_img_dir, target_label_dir = DIRS[target_set]
                    os.makedirs(target_img_dir, exist_ok=True)
                    os.makedirs(target_label_dir, exist_ok=True)

                    base = os.path.basename(old_img_path)
                    _, ext = os.path.splitext(base)
                    dst_img_path = os.path.join(target_img_dir, base)
                    if os.path.exists(dst_img_path):
                        dst_img_path = os.path.join(
                            target_img_dir,
                            f"image{_next_image_number()}{ext.lower() or '.png'}",
                        )

                    try:
                        shutil.move(old_img_path, dst_img_path)
                    except FileNotFoundError:
                        messagebox.showerror("Transfer failed",
                                             "Image file was not found. Refresh and try again.",
                                             parent=root)
                        return

                    old_label_path = get_label_path(old_img_path, old_label_dir)
                    new_label_path = os.path.join(
                        target_label_dir,
                        os.path.splitext(os.path.basename(dst_img_path))[0] + ".txt",
                    )
                    if os.path.isfile(old_label_path):
                        shutil.move(old_label_path, new_label_path)

                    old_tuple = (old_img_path, old_label_dir, old_set_name)
                    s["img_path"]  = dst_img_path
                    s["label_dir"] = target_label_dir
                    s["set_name"]  = target_set
                    if old_tuple in items:
                        items[items.index(old_tuple)] = (s["img_path"], s["label_dir"], s["set_name"])

                    sl.config(text=target_set,
                              fg=GREEN_FG if target_set == "train" else AMBER_FG)
                return _transfer

            _row_btn(row_frame, row_bg, "Transfer", make_transfer()
                     ).grid(row=0, column=5, sticky="ew", padx=1)

            def make_delete(s=state, rf=row_frame):
                def _delete():
                    p = s["img_path"]
                    l = s["label_dir"]
                    if not messagebox.askyesno(
                        "Delete image",
                        f"Delete {os.path.basename(p)} and its label (if any)?",
                        parent=root,
                    ):
                        return
                    try:
                        os.remove(p)
                    except FileNotFoundError:
                        pass
                    lp = get_label_path(p, l)
                    if os.path.isfile(lp):
                        os.remove(lp)
                    current_tuple = (s["img_path"], s["label_dir"], s["set_name"])
                    if current_tuple in items:
                        items.remove(current_tuple)
                    rf.destroy()
                    update_stats()
                return _delete

            _row_btn(row_frame, row_bg, "Delete", make_delete()
                     ).grid(row=0, column=6, sticky="ew", padx=1)

    build_home()
    root.mainloop()
    root.destroy()
    return result


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    while True:
        result = home_screen()
        if not result["action"]:
            break
        if result["images"]:
            for img_path, label_dir, _ in result["images"]:
                if not label_image(img_path, label_dir):
                    break


if __name__ == "__main__":
    main()