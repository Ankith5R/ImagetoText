import tkinter as tk
from tkinter import filedialog, font
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageOps
import os

BG       = "#ececec"
PANEL    = "#f5f5f5"
BORDER   = "#b0b0b0"
ACCENT   = "#000000"   #button below
ACCENT2  = "#000000"
TEXT     = "#000000"
TEXT_DIM = "#000000"
ERROR    = "#e03131"

MODEL_ID = "Salesforce/blip-image-captioning-large"


def load_model():
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BlipForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    return processor, model, device


def describe_image(image_path, processor, model, device):
    import torch

    pil_img = Image.open(image_path).convert("RGB")
    pil_img = ImageOps.exif_transpose(pil_img)

    #BLIP
    inputs = processor(pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, num_beams=5, early_stopping=True)
    caption = processor.decode(out[0], skip_special_tokens=True).strip()
    if caption:
        caption = caption[0].upper() + caption[1:]

    #OpenCV
    cv_img = cv2.imread(image_path)
    h, w = cv_img.shape[:2]

    #k-means for color dominance
    pixels = cv_img.reshape(-1, 3).astype(np.float32)
    _, labels, centers = cv2.kmeans(
        pixels, 3, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        3, cv2.KMEANS_RANDOM_CENTERS
    )
    dominant = centers[np.argmax(np.bincount(labels.flatten()))].astype(int)
    dom_name = bgr_to_name(dominant[0], dominant[1], dominant[2])

    #brightness
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    brightness = int(np.mean(gray))
    light_desc = "dark" if brightness < 64 else "moderately lit" if brightness < 140 else "bright"

    #Edges for determining complexity
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.count_nonzero(edges) / edges.size
    complexity = "simple" if edge_ratio < 0.05 else "detailed" if edge_ratio < 0.15 else "complex"

    #orientation
    aspect = "landscape" if w > h * 1.1 else "portrait" if h > w * 1.1 else "square"

    return (
        f"{caption}.\n\n"
        f"The image is {aspect} ({w}×{h} px), {light_desc}, "
        f"with a predominantly {dom_name} colour palette and a {complexity} composition."
    )


def bgr_to_name(b, g, r):
    colours = {
        "red": (255,0,0), "green": (0,128,0), "blue": (0,0,255),
        "yellow": (255,255,0), "orange": (255,165,0), "purple": (128,0,128),
        "pink": (255,105,180), "cyan": (0,255,255), "white": (255,255,255),
        "black": (0,0,0), "gray": (128,128,128), "brown": (139,69,19),
        "beige": (245,245,220), "navy": (0,0,128), "teal": (0,128,128),
    }
    best, best_d = "unknown", float("inf")
    for name, (nr, ng, nb) in colours.items():
        d = (r-nr)**2 + (g-ng)**2 + (b-nb)**2
        if d < best_d:
            best, best_d = name, d
    return best


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Describer · Local CV")
        self.geometry("1120x680")
        self.minsize(820, 540)
        self.configure(bg=BG)
        self._path = None
        self._photo = None
        self._processor = self._model = self._device = None
        self._build_fonts()
        self._build_ui()
        threading.Thread(target=self._load_model_worker, daemon=True).start()

    def _build_fonts(self):
        self.f_title  = font.Font(family="Georgia",     size=14, weight="bold")
        self.f_label  = font.Font(family="Georgia", size=14)
        self.f_body   = font.Font(family="Georgia",     size=12)
        self.f_hint   = font.Font(family="Georgia", size=12, weight="bold")
        self.f_btn    = font.Font(family="Georgia", size=12, weight="bold")
        self.f_status = font.Font(family="Courier New", size=9)

    def _build_ui(self):
        bar = tk.Frame(self, bg=PANEL, height=50)
        bar.pack(fill="x"); bar.pack_propagate(False)
        tk.Label(bar, text=" IMAGE to TEXT", font=self.f_title,
                 fg=ACCENT, bg=PANEL).pack(expand=True, pady=10)
        tk.Frame(self, bg=BORDER, height=10).pack(fill="x")

        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=18, pady=18)
        main.columnconfigure(0, weight=1, uniform="h")
        main.columnconfigure(1, weight=1, uniform="h")
        main.rowconfigure(0, weight=1)
        self._build_left(main)
        self._build_right(main)

        self._sv = tk.StringVar(value="Loading model…")
        sb = tk.Frame(self, bg=PANEL, height=28); sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)
        tk.Label(sb, textvariable=self._sv, font=self.f_status,
                 fg=TEXT_DIM, bg=PANEL, anchor="w", padx=14).pack(fill="both", expand=True)

    def _build_left(self, p):
        col = tk.Frame(p, bg=BG)
        col.grid(row=0, column=0, sticky="nsew", padx=(0,10))
        col.rowconfigure(1, weight=1)
        tk.Label(col, text="IMAGE", font=self.f_label, fg=TEXT_DIM,
                 bg=BG, anchor="w").grid(row=0, column=0, sticky="ew", pady=(0,6))

        self._img_frame = tk.Frame(col, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        self._img_frame.grid(row=1, column=0, sticky="nsew")
        self._img_frame.rowconfigure(0, weight=1); self._img_frame.columnconfigure(0, weight=1)

        self._ph = tk.Frame(self._img_frame, bg=PANEL)
        self._ph.grid(row=0, column=0, sticky="nsew")
        self._ph.rowconfigure(0, weight=1); self._ph.columnconfigure(0, weight=1)

        tk.Label(self._ph, text="No image selected", font=self.f_hint,
                 fg=TEXT_DIM, bg=PANEL).grid(row=1, column=0, pady=(0,60))
        self._img_lbl = tk.Label(self._img_frame, bg=PANEL)

        btn_row = tk.Frame(col, bg=BG)
        btn_row.grid(row=2, column=0, sticky="ew", pady=(10,0))
        btn_row.columnconfigure(0, weight=1); btn_row.columnconfigure(1, weight=1)
        self._btn_open = self._btn(btn_row, "  OPEN IMAGE  ", ACCENT, self._pick)
        self._btn_open.grid(row=0, column=0, sticky="ew", padx=(0,5))
        self._btn_run = self._btn(btn_row, "  DESCRIBE", ACCENT2, self._run, state="disabled")
        self._btn_run.grid(row=0, column=1, sticky="ew", padx=(5,0))

    def _build_right(self, p):
        col = tk.Frame(p, bg=BG)
        col.grid(row=0, column=1, sticky="nsew", padx=(10,0))
        col.rowconfigure(1, weight=1)
        tk.Label(col, text="DESCRIPTION", font=self.f_label, fg=TEXT_DIM,
                 bg=BG, anchor="w").grid(row=0, column=0, sticky="ew", pady=(0,6))
        tf = tk.Frame(col, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        tf.grid(row=1, column=0, sticky="nsew")
        tf.rowconfigure(0, weight=1); tf.columnconfigure(0, weight=1)
        self._txt = tk.Text(tf, font=self.f_body, fg=TEXT, bg=PANEL,
                            relief="flat", bd=0, wrap="word", width=40, height=20,
                            padx=20, pady=20, state="disabled", spacing1=4, spacing3=4)
        self._txt.grid(row=0, column=0, sticky="nsew")
        sc = tk.Scrollbar(tf, orient="vertical", command=self._txt.yview,
                          bg=PANEL, troughcolor=PANEL, highlightthickness=0)
        sc.grid(row=0, column=1, sticky="ns")
        self._txt.configure(yscrollcommand=sc.set)
        self._txt.tag_configure("dim",   foreground=TEXT_DIM, font=self.f_hint)
        self._txt.tag_configure("body",  foreground=TEXT,     font=self.f_body)
        self._txt.tag_configure("error", foreground=ERROR,    font=self.f_body)
        self._set_text("Loading local model — please wait…", "dim")

    def _btn(self, parent, label, color, cmd, state="normal"):
        b = tk.Button(parent, text=label, font=self.f_btn, fg="#000000", bg=color,
                      activeforeground="#000000", activebackground=color,
                      relief="flat", bd=0, padx=10, pady=10,
                      cursor="circle", command=cmd, state=state)
        lc = self._lighten(color)
        b.bind("<Enter>", lambda e: b.configure(bg=lc))
        b.bind("<Leave>", lambda e: b.configure(bg=color))
        return b

    @staticmethod
    def _lighten(h, n=28):
        r,g,b = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
        return f"#{min(255,r+n):02x}{min(255,g+n):02x}{min(255,b+n):02x}"

    def _set_text(self, s, tag="body"):
        self._txt.configure(state="normal")
        self._txt.delete("1.0", "end")
        self._txt.insert("end", s, tag)
        self._txt.configure(state="disabled")

    def _load_model_worker(self):
        try:
            self.after(0, lambda: self._sv.set("Downloading / loading BLIP model — first run only…"))
            self._processor, self._model, self._device = load_model()
            dev = "GPU" if self._device == "cuda" else "CPU"
            self.after(0, lambda: self._sv.set(f"Model ready on {dev} · open an image to begin"))
            self.after(0, lambda: self._set_text("Model loaded!  Open an image and press  DESCRIBE", "dim"))
            self.after(0, lambda: self._btn_open.configure(state="normal"))
        except Exception as e:
            self.after(0, lambda: self._set_text(
                f"Failed to load model:\n\n{e}\n\n"
                "Run:  pip install torch torchvision transformers pillow opencv-python", "error"))
            self.after(0, lambda: self._sv.set("Model load failed."))

    def _pick(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.webp *.gif"),("All","*.*")])
        if not path: return
        self._path = path
        img = Image.open(path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        self.update_idletasks()
        fw = self._img_frame.winfo_width()  or 500
        fh = self._img_frame.winfo_height() or 560
        img.thumbnail((fw-4, fh-4), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self._ph.grid_remove()
        self._img_lbl.configure(image=self._photo)
        self._img_lbl.grid(row=0, column=0)
        self._btn_run.configure(state="normal")
        self._sv.set(f"Loaded: {os.path.basename(path)}")
        self._set_text("Press  DESCRIBE to analyse.", "dim")

    def _run(self):
        if not self._path or self._model is None: return
        self._btn_run.configure(state="disabled", text="  ANALYSING…  ")
        self._btn_open.configure(state="disabled")
        self._set_text("Running computer vision…", "dim")
        self._sv.set("Analysing image locally…")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            result = describe_image(self._path, self._processor, self._model, self._device)
            self.after(0, lambda: self._set_text(result, "body"))
            self.after(0, lambda: self._sv.set(f"Done — {os.path.basename(self._path)}"))
        except Exception as e:
            self.after(0, lambda: self._set_text(f"Error:\n\n{e}", "error"))
            self.after(0, lambda: self._sv.set("Error during analysis."))
        finally:
            self.after(0, lambda: self._btn_run.configure(state="normal", text="  DESCRIBE  ▶"))
            self.after(0, lambda: self._btn_open.configure(state="normal"))


if __name__ == "__main__":
    App().mainloop()
