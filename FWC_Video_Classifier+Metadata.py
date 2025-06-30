#!/usr/bin/env python3
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.simplefilter("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass

import os
os.environ["TK_SILENCE_DEPRECATION"] = "1"

import cv2
import shutil
from datetime import datetime
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import easyocr
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# ───── Color Palette ────────────────────────────────────────
BG = "#1B3B5A"
TEXT = "#F0F0F0"
ACCENT = "#2ECC71"
BUTTON_BG = "#76C7C5"
CARD = "#F0F0F0"

# thresholds
CONFIRM_THRESH = 0.7
POSSIBLE_THRESH = 0.3

class YoloTkApp:
    def __init__(self, model_path):
        self.root = tk.Tk()
        self.root.title("FWC Panther Detector")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        self.root.configure(bg=BG)

        for pat, col in [
            ("*Background", BG), ("*Frame.background", BG),
            ("*Canvas.background", CARD), ("*Canvas.highlightBackground", ACCENT),
            ("*Text.background", CARD), ("*Text.foreground", TEXT),
            ("*Text.highlightBackground", ACCENT)
        ]:
            self.root.option_add(pat, col)

        # Image canvas
        self.canvas = tk.Canvas(
            self.root, width=1000, height=600,
            highlightthickness=4, highlightbackground=ACCENT
        )
        self.canvas.pack(padx=20, pady=(20,10))

        # Control buttons
        ctrl = tk.Frame(self.root)
        ctrl.pack(fill=tk.X, padx=20, pady=(0,10))
        opts = {
            "bg": BUTTON_BG, "fg": "#000000",
            "activebackground": ACCENT, "activeforeground": "#000000",
            "font": ("Poppins",12,"bold"), "bd":0, "relief":"flat",
            "highlightthickness":2, "highlightbackground":ACCENT,
            "width":18, "height":2
        }
        tk.Button(ctrl, text="Detect Image", command=self.open_image, **opts).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl, text="Batch Videos", command=self.process_folder, **opts).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl, text="Extract Metadata", command=self.extract_metadata, **opts).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl, text="Exit", command=self.exit_app, **opts).pack(side=tk.RIGHT, padx=5)

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(
            self.root, textvariable=self.status_var,
            font=("Poppins",10,"italic"), bg=BG, fg=TEXT
        ).pack(pady=(0,10))

        # Log area
        self.log = tk.Text(
            self.root, height=10, font=("Roboto",10), bd=0, relief="flat",
            highlightthickness=4, highlightbackground=ACCENT
        )
        self.log.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0,20))

        # Load model
        self.model = YOLO(model_path)
        msg = f"Loaded model: {model_path}\nClasses: {self.model.names}\n\n"
        print(msg, end="")
        self.log.insert(tk.END, msg)

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files","*.jpg *.png *.jpeg")])
        if not path:
            return
        img = cv2.imread(path)
        self._display_frame(img)

    def _display_frame(self, frame):
        results = self.model(frame)[0]
        for box in results.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            cls,conf = int(box.cls[0]), float(box.conf[0])
            name = self.model.names[cls]
            color = (0,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(
                frame, f"{name} {conf:.2f}",
                (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.canvas.delete("all")
        self.current_image = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(0,0,anchor="nw",image=self.current_image)

    def process_folder(self):
        src = filedialog.askdirectory(title="Select folder of videos")
        if not src:
            return
        vids = [f for f in os.listdir(src) if f.lower().endswith((".mp4",".avi",".mov",".mkv"))]
        total = len(vids)
        if total == 0:
            messagebox.showinfo("No videos","No video files found.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.dirname(__file__)
        def_dir = os.path.join(base, f"panther_definite_{ts}")
        pos_dir = os.path.join(base, f"panther_possible_{ts}")
        os.makedirs(def_dir, exist_ok=True)
        os.makedirs(pos_dir, exist_ok=True)

        definite, possible = [], []
        self.log.delete("1.0", tk.END)
        for i,f in enumerate(vids,1):
            self.status_var.set(f"{i}/{total}: {f}")
            self.log.insert(tk.END, f"{f}... ")
            self.log.update_idletasks()
            conf = self._analyze(f"{src}/{f}")
            if conf >= CONFIRM_THRESH:
                shutil.copy2(f"{src}/{f}", f"{def_dir}/{f}")
                definite.append((f,conf))
            elif conf >= POSSIBLE_THRESH:
                shutil.copy2(f"{src}/{f}", f"{pos_dir}/{f}")
                possible.append((f,conf))
            self.log.insert(tk.END, f"(conf={conf:.2f})\n")

        # Summary
        self.log.insert(tk.END, "\nSummary:\n")
        self.log.insert(tk.END, f"Definite({len(definite)}):\n")
        for fn,c in definite:
            self.log.insert(tk.END, f"  {fn}({c:.2f})\n")
        self.log.insert(tk.END, f"Possible({len(possible)}):\n")
        for fn,c in possible:
            self.log.insert(tk.END, f"  {fn}({c:.2f})\n")
        saved = len(definite) + len(possible)
        self.status_var.set(f"Done: {saved} videos saved.")
        messagebox.showinfo(
            "Finished",
            f"Processed {total} videos.\nDefinite:{len(definite)} Possible:{len(possible)}"
        )

    def extract_metadata(self):
        src = filedialog.askdirectory(title="Select videos for metadata")
        if not src:
            return
        out = os.path.dirname(os.path.abspath(__file__))

        vids = [f for f in os.listdir(src) if f.lower().endswith((".mp4",".avi",".mov",".mkv"))]
        total = len(vids)
        if total == 0:
            messagebox.showinfo("No videos","No video files found.")
            return

        self.log.delete("1.0", tk.END)
        self.status_var.set("Starting metadata extraction…")
        self.ocr_reader = easyocr.Reader(['en'], gpu=False)
        rows = []

        def ocr_task(filename):
            cap = cv2.VideoCapture(f"{src}/{filename}")
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return filename, ""
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            # ROI bottom 20%
            h,w = gray.shape
            gray = gray[int(0.8*h):h, :w]
            # Otsu threshold
            _, gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            res = self.ocr_reader.readtext(gray)
            text = " | ".join([t[1] for t in res])
            return filename, text

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(ocr_task,f): f for f in vids}
            for i, fut in enumerate(as_completed(futures),1):
                fn,text = fut.result()
                self.status_var.set(f"OCR {i}/{total}: {fn}")
                self.log.insert(tk.END, f"{fn}... ok\n")
                rows.append({"filename":fn, "text":text})
                self.log.update_idletasks()

        df = pd.DataFrame(rows)
        csv_path = os.path.join(out, "Video_Metadata_Extraction.csv")
        df.to_csv(csv_path, index=False)

        self.status_var.set("Metadata extraction done")
        self.log.insert(tk.END, f"\nSaved CSV to: {csv_path}\n")
        messagebox.showinfo("Done", f"Extracted metadata for {len(rows)} videos.\nCSV at:\n{csv_path}")

    def _analyze(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return 0.0
        best = 0.0
        for t in (1000,3000,5000):
            cap.set(cv2.CAP_PROP_POS_MSEC, t)
            ret, frame = cap.read()
            if not ret:
                continue
            res = self.model(frame)[0]
            for b in res.boxes:
                if self.model.names[int(b.cls[0])] == "panther":
                    best = max(best, float(b.conf[0]))
        cap.release()
        return best

    def exit_app(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    splash = tk.Tk()
    splash.title("FWC Panther Detector")
    splash.configure(bg=BG)
    splash.geometry("600x400")
    splash.resizable(False,False)
    splash.update()
    w,h = splash.winfo_width(), splash.winfo_height()
    ws,hs = splash.winfo_screenwidth(), splash.winfo_screenheight()
    splash.geometry(f"{w}x{h}+{(ws-w)//2}+{(hs-h)//2}")
    try:
        logo = ImageTk.PhotoImage(
            Image.open("fwc_logo.png").resize((120,120))
        )
        tk.Label(splash,image=logo,bg=BG).pack(pady=30)
    except:
        pass
    tk.Label(
        splash, text="Panther Detector",
        font=("Poppins",24,"bold"), bg=BG, fg=TEXT
    ).pack()
    tk.Button(
        splash, text="Launch",
        command=lambda:[splash.destroy(), YoloTkApp(os.path.join(os.path.dirname(__file__),"best.pt")).run()],
        font=("Poppins",14,"bold"),
        bg=BUTTON_BG, fg="#000000",
        bd=0, relief="flat",
        highlightthickness=2, highlightbackground=ACCENT,
        width=20, height=2
    ).pack(pady=20)
    splash.mainloop()
