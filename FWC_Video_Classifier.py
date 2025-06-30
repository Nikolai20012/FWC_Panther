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

# ───── Color Palette ────────────────────────────────────────
BG = "#1B3B5A"        # deep navy-blue background
TEXT = "#F0F0F0"      # crisp off-white text
ACCENT = "#2ECC71"     # vibrant green highlights
BUTTON_BG = "#76C7C5"  # bold aqua button backgrounds
CARD = "#F0F0F0"       # off-white cards/log

# classification thresholds
CONFIRM_THRESH = 0.7    # definite panther
POSSIBLE_THRESH = 0.3   # possible panther

class YoloTkApp:
    def __init__(self, model_path):
        # Main window
        self.root = tk.Tk()
        self.root.title("FWC Panther Detector")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        self.root.configure(bg=BG)

        # Theme
        for pat, col in [("*Background", BG), ("*Frame.background", BG),
                         ("*Canvas.background", CARD), ("*Canvas.highlightBackground", ACCENT),
                         ("*Text.background", CARD), ("*Text.foreground", TEXT),
                         ("*Text.highlightBackground", ACCENT)]:
            self.root.option_add(pat, col)

        # Canvas
        self.canvas = tk.Canvas(self.root, width=1000, height=600,
                                highlightthickness=4, highlightbackground=ACCENT)
        self.canvas.pack(padx=20, pady=(20,10))

        # Controls
        ctrl = tk.Frame(self.root)
        ctrl.pack(fill=tk.X, padx=20, pady=(0,10))
        opts = {"bg": BUTTON_BG, "fg":"#000000",
                "activebackground": ACCENT, "activeforeground":"#000000",
                "font": ("Poppins",10,"bold"), "bd":0, "relief":"flat",
                "highlightthickness":2, "highlightbackground":ACCENT}
        tk.Button(ctrl, text="Detect Image", command=self.open_image, **opts).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl, text="Batch Detect Videos", command=self.process_folder, **opts).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl, text="Exit", command=self.exit_app, **opts).pack(side=tk.RIGHT, padx=5)

        # Status
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var,
                 font=("Poppins",10,"italic"), bg=BG, fg=TEXT).pack(pady=(0,10))

        # Log
        self.log = tk.Text(self.root, height=10, font=("Roboto",10), bd=0, relief="flat",
                           highlightthickness=4, highlightbackground=ACCENT)
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
            color=(0,255,0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"{name} {conf:.2f}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
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
        if total==0:
            messagebox.showinfo("No videos","No video files found.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.dirname(__file__)
        def_dir = os.path.join(base,f"panther_definite_{ts}")
        pos_dir = os.path.join(base,f"panther_possible_{ts}")
        os.makedirs(def_dir,exist_ok=True)
        os.makedirs(pos_dir,exist_ok=True)

        definite, possible = [], []
        self.log.delete("1.0",tk.END)
        for i,f in enumerate(vids,1):
            self.status_var.set(f"{i}/{total}: {f}")
            path = os.path.join(src,f)
            self.log.insert(tk.END,f"{f}... ")
            self.log.update_idletasks()
            conf,det = self._analyze(path)
            if conf>=CONFIRM_THRESH:
                shutil.copy2(path,os.path.join(def_dir,f))
                definite.append((f,conf))
            elif conf>=POSSIBLE_THRESH:
                shutil.copy2(path,os.path.join(pos_dir,f))
                possible.append((f,conf))
            self.log.insert(tk.END,f"(conf={conf:.2f})\n")
        # Summary
        self.log.insert(tk.END,"\nSummary:\n")
        self.log.insert(tk.END,f"Definite({len(definite)}):\n")
        for fn,c in definite: self.log.insert(tk.END,f"  {fn}({c:.2f})\n")
        self.log.insert(tk.END,f"Possible({len(possible)}):\n")
        for fn,c in possible: self.log.insert(tk.END,f"  {fn}({c:.2f})\n")
        saved=len(definite)+len(possible)
        self.status_var.set(f"Done: {saved} saved.")
        messagebox.showinfo("Finished",
            f"Processed {total}\nDefinite:{len(definite)} Possible:{len(possible)}")

    def _analyze(self,path):
        cap=cv2.VideoCapture(path)
        if not cap.isOpened(): return 0.0,{}
        best=0.0;det={}
        for t in (1000,3000,5000):
            cap.set(cv2.CAP_PROP_POS_MSEC,t)
            r,fr=cap.read()
            if not r: continue
            res=self.model(fr)[0]
            for b in res.boxes:
                idx=int(b.cls[0]); name=self.model.names[idx]
                c=float(b.conf[0]); det[name]=max(det.get(name,0),c)
                if name=="panther": best=max(best,c)
        cap.release(); return best,det

    def exit_app(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()

# Splash and launch
if __name__=="__main__":
    splash=tk.Tk()
    splash.title("FWC Panther Detector")
    splash.configure(bg=BG)
    splash.geometry("600x400")
    splash.resizable(False,False)
    splash.update()
    w,h=splash.winfo_width(),splash.winfo_height()
    ws,hs=splash.winfo_screenwidth(),splash.winfo_screenheight()
    splash.geometry(f"{w}x{h}+{(ws-w)//2}+{(hs-h)//2}")
    try:
        logo=ImageTk.PhotoImage(Image.open("fwc_logo.png").resize((120,120)))
        tk.Label(splash,image=logo,bg=BG).pack(pady=30)
    except:
        pass
    tk.Label(splash,text="Panther Detector",font=("Poppins",24,"bold"),bg=BG,fg=TEXT).pack()
    def start():
        splash.destroy()
        app=YoloTkApp(os.path.join(os.path.dirname(__file__),"best.pt"))
        app.run()
    tk.Button(splash,text="Launch",command=start,
              font=("Poppins",14,"bold"),bg=BUTTON_BG,fg="#000000",
              bd=0,relief="flat",highlightthickness=2,highlightbackground=ACCENT,
              width=20,height=2).pack(pady=20)
    splash.mainloop()
