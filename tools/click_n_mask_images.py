# -*- coding: utf-8 -*-
"""
SAM GUI – brukervennlig klikk & mask for binære verktøy-masker.
- Åpne bilde eller mappe
- Venstreklikk: positivt punkt (på verktøy)
- Høyreklikk: negativt punkt (på bakgrunn)
- Ctrl+Z: angre siste punkt
- r: reset alle punkter
- s: lagre binær maske (0/255) som .png ved siden av bilde
- o: lagre overlay (mask over original)
- Piltaster venstre/høyre: bytt bilde når mappe er åpnet

Forhåndskrav:
- sam_vit_b_01ec64.pth (eller annen .pth) i samme mappe som dette skriptet
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageTk

# HEIC-støtte (valgfritt)
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import torch
from segment_anything import sam_model_registry, SamPredictor

# ---------- KONFIG ----------
BASE_DIR = Path(r"C:\Anvendt KI - Projects\project-samlingsuke1\tools")
SAM_CHECKPOINT = BASE_DIR / "sam_vit_b_01ec64.pth"  # legg .pth her
SAM_MODEL_TYPE = "vit_b"  # "vit_b" | "vit_l" | "vit_h"

CANVAS_BG = "#111111"
POINT_POS_COLOR = (0, 255, 0)   # grønn
POINT_NEG_COLOR = (255, 64, 64) # rød
MASK_COLOR      = (0, 200, 0)   # for overlay
MASK_ALPHA      = 0.45

# ---------- HJELP ----------
def load_image_any(path: Path) -> Image.Image:
    """Åpne bilde (inkl. HEIC) og korriger EXIF-orientering."""
    im = Image.open(path).convert("RGB")
    im = ImageOps.exif_transpose(im)
    return im

def np_rgb_to_bgr(a: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)

def np_bgr_to_rgb(a: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

def rgba_overlay(base_bgr: np.ndarray, mask_bool: np.ndarray, color=(0,200,0), alpha=0.45) -> np.ndarray:
    base = base_bgr.copy()
    overlay = base_bgr.copy()
    overlay[mask_bool] = color
    return cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)

def ensure_mask_suffix(out_path: Path) -> Path:
    # Hvis original også er .png, legg _mask
    if out_path.suffix.lower() == ".png":
        return out_path.with_name(out_path.stem + "_mask.png")
    return out_path

# ---------- GUI-APP ----------
class SamGuiApp:
    def __init__(self, root):
        self.root = root
        root.title("Segment Anything – Verktøy-maskering")
        root.geometry("1280x800")

        # Statusvariabler
        self.current_image_path: Path | None = None
        self.image_list: list[Path] = []
        self.image_index: int = -1

        # Bilde/visning
        self.img_pil: Image.Image | None = None
        self.img_np_rgb: np.ndarray | None = None
        self.img_np_bgr: np.ndarray | None = None
        self.preview_img: Image.Image | None = None
        self.preview_scale = 1.0
        self.preview_offset = (0, 0)  # (x0,y0) i canvas

        # SAM
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = None
        self.predictor: SamPredictor | None = None

        # Klikk-punkter (original-coords)
        self.pos_points: list[tuple[int,int]] = []
        self.neg_points: list[tuple[int,int]] = []

        # Gjeldende maske (bool i original-størrelse)
        self.current_mask_bool: np.ndarray | None = None

        self._build_ui()
        self._load_sam()

    # ---------- UI ----------
    def _build_ui(self):
        # Top toolbar
        toolbar = ttk.Frame(self.root, padding=(8,6))
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn_open = ttk.Button(toolbar, text="Åpne bilde", command=self.open_image_dialog)
        self.btn_open.pack(side=tk.LEFT, padx=4)

        self.btn_open_dir = ttk.Button(toolbar, text="Åpne mappe", command=self.open_dir_dialog)
        self.btn_open_dir.pack(side=tk.LEFT, padx=4)

        self.sep1 = ttk.Separator(toolbar, orient=tk.VERTICAL)
        self.sep1.pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.btn_prev = ttk.Button(toolbar, text="◀ Forrige", command=lambda: self.change_image(-1))
        self.btn_prev.pack(side=tk.LEFT, padx=2)

        self.btn_next = ttk.Button(toolbar, text="Neste ▶", command=lambda: self.change_image(+1))
        self.btn_next.pack(side=tk.LEFT, padx=2)

        self.sep2 = ttk.Separator(toolbar, orient=tk.VERTICAL)
        self.sep2.pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.btn_reset = ttk.Button(toolbar, text="Reset (r)", command=self.reset_points)
        self.btn_reset.pack(side=tk.LEFT, padx=2)

        self.btn_undo = ttk.Button(toolbar, text="Angre (Ctrl+Z)", command=self.undo_point)
        self.btn_undo.pack(side=tk.LEFT, padx=2)

        self.btn_recompute = ttk.Button(toolbar, text="Oppdater maske", command=self.recompute_mask)
        self.btn_recompute.pack(side=tk.LEFT, padx=2)

        self.sep3 = ttk.Separator(toolbar, orient=tk.VERTICAL)
        self.sep3.pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.btn_save_mask = ttk.Button(toolbar, text="Lagre maske (s)", command=self.save_mask_png)
        self.btn_save_mask.pack(side=tk.LEFT, padx=2)

        self.btn_save_overlay = ttk.Button(toolbar, text="Lagre overlay (o)", command=self.save_overlay_png)
        self.btn_save_overlay.pack(side=tk.LEFT, padx=2)

        # Statuslabel
        self.status_var = tk.StringVar(value="Klar")
        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=4)

        # Canvas for bilde
        self.canvas = tk.Canvas(self.root, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Bind events
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_left_click)   # positivt punkt
        self.canvas.bind("<Button-3>", self._on_right_click)  # negativt punkt

        self.root.bind("<Key>", self._on_key)
        self.root.bind("<Control-z>", lambda e: self.undo_point())

    def set_status(self, text: str):
        self.status_var.set(text)
        self.root.update_idletasks()

    # ---------- SAM ----------
    def _load_sam(self):
        if not SAM_CHECKPOINT.exists():
            messagebox.showerror("Feil", f"Fant ikke SAM checkpoint:\n{SAM_CHECKPOINT}\n\n"
                                         f"Last ned f.eks. sam_vit_b_01ec64.pth og plasser her.")
            return
        self.set_status(f"Laster SAM ({SAM_MODEL_TYPE}) …")
        self.sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        self.set_status(f"SAM klar ({self.device})")

    # ---------- FILHÅNDTERING ----------
    def open_image_dialog(self):
        f = filedialog.askopenfilename(
            title="Velg et bilde",
            filetypes=[("Bilder", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.heic;*.heif"),
                       ("Alle filer", "*.*")]
        )
        if not f:
            return
        self.image_list = [Path(f)]
        self.image_index = 0
        self._load_current_image()

    def open_dir_dialog(self):
        d = filedialog.askdirectory(title="Velg mappe med bilder")
        if not d:
            return
        root = Path(d)
        exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".heic",".heif"}
        files = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in exts and p.is_file()]
        if not files:
            messagebox.showinfo("Ingen bilder", "Fant ingen bildefiler i mappen.")
            return
        self.image_list = files
        self.image_index = 0
        self._load_current_image()

    def change_image(self, delta: int):
        if not self.image_list:
            return
        self.image_index = (self.image_index + delta) % len(self.image_list)
        self._load_current_image()

    def _load_current_image(self):
        self.current_mask_bool = None
        self.pos_points.clear()
        self.neg_points.clear()

        p = self.image_list[self.image_index]
        self.current_image_path = p
        try:
            self.img_pil = load_image_any(p)
        except Exception as e:
            messagebox.showerror("Feil", f"Kunne ikke åpne bildet:\n{p}\n\n{e}")
            return

        self.img_np_rgb = np.array(self.img_pil)              # HxWx3 (RGB)
        self.img_np_bgr = np_rgb_to_bgr(self.img_np_rgb)      # for overlay
        if self.predictor is not None:
            self.predictor.set_image(self.img_np_rgb)

        self.set_status(f"{p.name}  ({self.image_index+1}/{len(self.image_list)})")
        self._refresh_preview()

    # ---------- VISNING ----------
    def _on_resize(self, event):
        self._refresh_preview()

    def _compute_fit(self, w, h, frame_w, frame_h):
        # skaler til å passe inn i (frame_w, frame_h)
        scale = min(frame_w / w, frame_h / h) if w and h else 1.0
        new_w = int(w * scale)
        new_h = int(h * scale)
        x0 = (frame_w - new_w) // 2
        y0 = (frame_h - new_h) // 2
        return scale, x0, y0, new_w, new_h

    def _refresh_preview(self):
        if self.img_pil is None:
            self.canvas.delete("all")
            return
        c_w = self.canvas.winfo_width() or 1280
        c_h = self.canvas.winfo_height() or 720

        h, w = self.img_np_rgb.shape[:2]
        scale, x0, y0, new_w, new_h = self._compute_fit(w, h, c_w, c_h)
        self.preview_scale = scale
        self.preview_offset = (x0, y0)

        # base preview
        vis = self.img_np_bgr.copy()

        # mask overlay
        if self.current_mask_bool is not None:
            vis = rgba_overlay(vis, self.current_mask_bool, color=MASK_COLOR, alpha=MASK_ALPHA)

        vis_resized = cv2.resize(vis, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # tegn punkter (skalert til canvas)
        for (x, y) in self.pos_points:
            sx = int(x * scale) + x0
            sy = int(y * scale) + y0
            cv2.circle(vis_resized, (sx - x0, sy - y0), 5, POINT_POS_COLOR, -1)
        for (x, y) in self.neg_points:
            sx = int(x * scale) + x0
            sy = int(y * scale) + y0
            cv2.circle(vis_resized, (sx - x0, sy - y0), 5, POINT_NEG_COLOR, -1)

        # til Tk
        vis_rgb = np_bgr_to_rgb(vis_resized)
        im = Image.fromarray(vis_rgb)
        self.preview_img = ImageTk.PhotoImage(im)

        self.canvas.delete("all")
        self.canvas.create_image(x0, y0, anchor="nw", image=self.preview_img)

    def _canvas_to_image_xy(self, cx, cy):
        """Konverter canvas-klikk til original bildepiksel (x,y). Returnerer None om utenfor."""
        if self.img_np_rgb is None:
            return None
        x0, y0 = self.preview_offset
        scale = self.preview_scale
        x = (cx - x0) / scale
        y = (cy - y0) / scale
        h, w = self.img_np_rgb.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return int(round(x)), int(round(y))

    # ---------- INTERAKSJON ----------
    def _on_left_click(self, event):
        pt = self._canvas_to_image_xy(event.x, event.y)
        if pt is None:
            return
        self.pos_points.append(pt)
        self.recompute_mask()

    def _on_right_click(self, event):
        pt = self._canvas_to_image_xy(event.x, event.y)
        if pt is None:
            return
        self.neg_points.append(pt)
        self.recompute_mask()

    def _on_key(self, event):
        k = event.keysym.lower()
        if k == "r":
            self.reset_points()
        elif k == "s":
            self.save_mask_png()
        elif k == "o":
            self.save_overlay_png()
        elif k in ("left", "prior"):  # PgUp
            self.change_image(-1)
        elif k in ("right", "next"):  # PgDn
            self.change_image(+1)

    def undo_point(self):
        if self.neg_points:
            self.neg_points.pop()
        elif self.pos_points:
            self.pos_points.pop()
        self.recompute_mask()

    def reset_points(self):
        self.pos_points.clear()
        self.neg_points.clear()
        self.current_mask_bool = None
        self._refresh_preview()

    # ---------- MASKERING ----------
    def recompute_mask(self):
        if self.predictor is None or self.img_np_rgb is None:
            return
        if not self.pos_points and not self.neg_points:
            self.current_mask_bool = None
            self._refresh_preview()
            return

        pts = np.array(self.pos_points + self.neg_points, dtype=np.float32)
        labels = np.array([1]*len(self.pos_points) + [0]*len(self.neg_points), dtype=np.int32)

        try:
            masks, scores, _ = self.predictor.predict(
                point_coords=pts,
                point_labels=labels,
                multimask_output=True
            )
            i = int(np.argmax(scores))
            mask = masks[i].astype(np.uint8)  # HxW {0,1}
            self.current_mask_bool = mask.astype(bool)
        except Exception as e:
            messagebox.showerror("SAM-feil", str(e))
            return

        self._refresh_preview()

    # ---------- LAGRING ----------
    def save_mask_png(self):
        if self.current_mask_bool is None or self.current_image_path is None:
            messagebox.showinfo("Ingen maske", "Lag maske først (klikk på verktøyet).")
            return
        out = self.current_image_path.with_suffix(".png")
        if out == self.current_image_path:
            out = ensure_mask_suffix(out)
        try:
            bin_mask = (self.current_mask_bool.astype(np.uint8) * 255)
            ok = cv2.imwrite(str(out), bin_mask)
            if ok:
                self.set_status(f"Lagret maske: {out}")
            else:
                raise RuntimeError("cv2.imwrite returnerte False")
        except Exception as e:
            messagebox.showerror("Lagringsfeil", str(e))

    def save_overlay_png(self):
        if self.current_mask_bool is None or self.current_image_path is None:
            messagebox.showinfo("Ingen maske", "Lag maske først (klikk på verktøyet).")
            return
        try:
            over = rgba_overlay(self.img_np_bgr, self.current_mask_bool, color=MASK_COLOR, alpha=MASK_ALPHA)
            out = self.current_image_path.with_name(self.current_image_path.stem + "_overlay.png")
            ok = cv2.imwrite(str(out), over)
            if ok:
                self.set_status(f"Lagret overlay: {out}")
            else:
                raise RuntimeError("cv2.imwrite returnerte False")
        except Exception as e:
            messagebox.showerror("Lagringsfeil", str(e))

# ---------- MAIN ----------
def main():
    # Sjekk modell
    if not SAM_CHECKPOINT.exists():
        tk.Tk().withdraw()
        messagebox.showerror("Mangler modell",
                             f"Fant ikke SAM checkpoint:\n{SAM_CHECKPOINT}\n\n"
                             f"Last ned f.eks. sam_vit_b_01ec64.pth og plasser der.")
        sys.exit(1)

    root = tk.Tk()
    # Litt finere ttk-stil
    try:
        from tkinter import font
        default_style = ttk.Style()
        if "clam" in default_style.theme_names():
            default_style.theme_use("clam")
    except Exception:
        pass

    app = SamGuiApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
