"""Real-time explorer for EVERY tunable parameter of the multilayer cloud shader.

Left  : live render (continuous loop, so the GPU stays warmed and the perf read-out is meaningful).
Right : scrollable sliders for all cloud / atmosphere / geometry parameters, grouped.
Top   : live GPU cost (GL_TIME_ELAPSED, median+min over a rolling window) -> ms and fps, for the
        currently selected render resolution.

  Mouse on the render:  LEFT-drag  = camera pitch (vertical) + yaw (horizontal)
                        RIGHT-drag = sun elevation (vertical) + azimuth (horizontal)
                        WHEEL      = focal length (zoom / FoV)
  Keys:  1 / 2 / 3 = render at HD 1280x720 / FHD 1920x1080 / 4K 3840x2160  (perf is for THIS res)
         b = load the best run's shared+coverage params   r = reset all   s = save preset
         l = load preset   q / Esc = quit

Perf note: the read-out needs a second or two of continuous rendering to settle onto the sustained
boost clock (idle clock is ~210 MHz on this GPU). The 'min' column is the best-clock cost; the 'med'
column is the steady sustained cost. Both are pure GPU time, independent of the Tk display overhead.

Run with the `dino` env (glfw + PyOpenGL + Pillow):  python param_explorer.py
"""
from __future__ import annotations
import os, json, math, time, collections
import numpy as np, glfw
from OpenGL.GL import *
from PIL import Image, ImageTk
import tkinter as tk

HERE   = os.path.dirname(os.path.abspath(__file__))
SHADER = os.path.join(HERE, "cumulus_multilayer.glsl")
BEST   = os.path.join(HERE, "..", "results", "multi_best_dual.json")
PRESET = os.path.join(HERE, "param_preset.json")
VS = ("#version 330 core\nvoid main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));"
      " gl_Position=vec4(p*2.0-1.0,0.0,1.0);}\n")
RES = {"HD": (1280, 720), "FHD": (1920, 1080), "4K": (3840, 2160)}
DISP_W = 880                                     # on-screen width of the render (16:9 -> 495 tall)

# ---- parameter groups: (label, [(key, lo, hi, kind)])  kind: 'f' float, 'deg' angle-in-degrees
GROUPS = [
 ("Camera / view", [
    ("height", 0.0, 9.0, 'f'), ("pitch", -90, 90, 'deg'), ("yaw", -180, 180, 'deg'),
    ("focal", 0.2, 3.0, 'f')]),
 ("Sun", [("sunEl", -18, 90, 'deg'), ("sunAz", -180, 180, 'deg')]),
 ("Exposure", [("expo", 0.20, 1.60, 'f')]),
 ("Atmosphere (scattering)", [
    ("uAtmDens", 0.4, 1.8, 'f'), ("uSunI", 1.0, 7.0, 'f'), ("uHaze", 0.0, 0.5, 'f'),
    ("uAmbStr", 0.0, 1.2, 'f')]),
 ("Cloud lighting / scatter", [
    ("uDens", 2.0, 24.0, 'f'), ("uSigma", 2.0, 16.0, 'f'), ("uSilver", 0.0, 6.0, 'f'),
    ("uMsBoost", 0.0, 2.0, 'f'), ("uDiffuse", 0.0, 0.5, 'f')]),
 ("Cloud detail / shape", [
    ("uDetAmp", 0.0, 0.8, 'f'), ("uDet2Amp", 0.0, 0.6, 'f'), ("uDet3Amp", 0.0, 0.4, 'f'),
    ("uCovFreq", 0.05, 0.5, 'f'), ("uWarp", 0.0, 2.0, 'f'), ("uShear", 0.0, 4.0, 'f')]),
 ("Deck altitudes (km)", [
    ("uLowBase", 0.2, 2.0, 'f'), ("uLowTop", 1.5, 5.0, 'f'), ("uMidBase", 3.5, 6.0, 'f'),
    ("uMidTop", 4.5, 7.0, 'f'), ("uHiBase", 5.5, 8.5, 'f'), ("uHiTop", 6.5, 9.5, 'f')]),
 ("Coverage / type", [
    ("uCovLow", 0.0, 1.0, 'f'), ("uCovMid", 0.0, 1.0, 'f'), ("uCovHigh", 0.0, 1.0, 'f'),
    ("uTypeAmp", 0.0, 0.30, 'f'), ("uTypeSeed", 0.0, 8.0, 'f')]),
]
SHADER_KEYS = [k for _, items in GROUPS for (k, *_ ) in items
               if k not in ("height", "pitch", "yaw", "focal", "sunEl", "sunAz", "expo")]

DEFAULTS = dict(height=0.06, pitch=math.radians(20), yaw=0.0, focal=1.05,
                sunEl=math.radians(45), sunAz=math.radians(35), expo=0.85,
                uAtmDens=1.1, uSunI=4.5, uHaze=0.12, uAmbStr=0.55,
                uDens=8.0, uSigma=8.0, uSilver=1.0, uMsBoost=0.5, uDiffuse=0.10,
                uDetAmp=0.55, uDet2Amp=0.30, uDet3Amp=0.16, uCovFreq=0.22, uWarp=0.7, uShear=0.8,
                uLowBase=0.9, uLowTop=3.0, uMidBase=4.4, uMidTop=5.4, uHiBase=6.6, uHiTop=7.8,
                uCovLow=0.6, uCovMid=0.3, uCovHigh=0.3, uTypeAmp=0.10, uTypeSeed=2.0)

def _sh(src, st):
    s = glCreateShader(st); glShaderSource(s, src); glCompileShader(s)
    if glGetShaderiv(s, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s

class Renderer:
    def __init__(self):
        glfw.init()
        for k, v in [(glfw.VISIBLE, glfw.FALSE), (glfw.CONTEXT_VERSION_MAJOR, 3),
                     (glfw.CONTEXT_VERSION_MINOR, 3), (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)]:
            glfw.window_hint(k, v)
        self.win = glfw.create_window(64, 64, "gl", None, None); glfw.make_context_current(self.win)
        p = glCreateProgram()
        glAttachShader(p, _sh(VS, GL_VERTEX_SHADER))
        glAttachShader(p, _sh(open(SHADER, encoding="utf-8").read(), GL_FRAGMENT_SHADER))
        glLinkProgram(p)
        if glGetProgramiv(p, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError(glGetProgramInfoLog(p).decode())
        self.prog = p; self.vao = glGenVertexArrays(1); self.q = int(glGenQueries(1)[0])
        self.W = self.H = 0; self.tex = self.fbo = None
    def _alloc(self, W, H):
        if (W, H) == (self.W, self.H): return
        if self.tex: glDeleteTextures(1, [self.tex]); glDeleteFramebuffers(1, [self.fbo])
        self.tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, None)
        self.fbo = glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.tex, 0)
        self.W, self.H = W, H
    def render(self, p, W, H):
        """Render at W x H. Returns (rgb float array HxWx3, gpu_ms). Does the ambient 1x1 pre-pass
        so the perf number reflects the production (hoisted-ambient) path."""
        self._alloc(W, H); glViewport(0, 0, W, H)
        glUseProgram(self.prog); U = lambda n: glGetUniformLocation(self.prog, n)
        eye = np.array([0.0, p["height"], 0.0], np.float32)
        cy, sy, cp, sp = math.cos(p["yaw"]), math.sin(p["yaw"]), math.cos(p["pitch"]), math.sin(p["pitch"])
        d = np.array([sy*cp, sp, cy*cp], np.float32); f = d/np.linalg.norm(d)
        up0 = np.array([0.0, 1.0, 0.0]); r = np.cross(up0, f)
        r = r/np.linalg.norm(r) if np.linalg.norm(r) > 1e-6 else np.array([1.0, 0.0, 0.0])
        u = np.cross(f, r); cam = np.stack([r, u, f], 1).astype(np.float32)
        sd = np.array([math.cos(p["sunEl"])*math.sin(p["sunAz"]), math.sin(p["sunEl"]),
                       math.cos(p["sunEl"])*math.cos(p["sunAz"])], np.float32)
        glUniform2f(U("uRes"), float(W), float(H)); glUniform3f(U("uEye"), *[float(z) for z in eye])
        glUniformMatrix3fv(U("uCam"), 1, GL_FALSE, cam.T.copy()); glUniform1f(U("uFocal"), float(p["focal"]))
        glUniform3f(U("uSunDir"), *[float(z) for z in sd]); glUniform1f(U("uExposure"), float(p["expo"]))
        for k in SHADER_KEYS: glUniform1f(U(k), float(p[k]))
        glBindVertexArray(self.vao)
        # ambient pre-pass (1x1): fill uAmbZen/uAmbHor so main() takes the hoisted path
        glUniform1f(U("uAmbMode"), 0.0)
        glUniform1f(U("uProbe"), 1.0); glDrawArrays(GL_TRIANGLES, 0, 3)
        zen = np.frombuffer(glReadPixels(0, 0, 1, 1, GL_RGB, GL_FLOAT), np.float32)[:3]
        glUniform1f(U("uProbe"), 2.0); glDrawArrays(GL_TRIANGLES, 0, 3)
        hor = np.frombuffer(glReadPixels(0, 0, 1, 1, GL_RGB, GL_FLOAT), np.float32)[:3]
        glUniform1f(U("uProbe"), 0.0); glUniform1f(U("uAmbMode"), 1.0)
        glUniform3f(U("uAmbZen"), *[float(z) for z in zen]); glUniform3f(U("uAmbHor"), *[float(z) for z in hor])
        glBeginQuery(GL_TIME_ELAPSED, self.q); glDrawArrays(GL_TRIANGLES, 0, 3); glEndQuery(GL_TIME_ELAPSED)
        a = GL_FALSE
        while a == GL_FALSE: a = glGetQueryObjectiv(self.q, GL_QUERY_RESULT_AVAILABLE)
        ms = glGetQueryObjectuiv(self.q, GL_QUERY_RESULT)/1e6
        rgb = np.frombuffer(glReadPixels(0, 0, W, H, GL_RGB, GL_FLOAT), np.float32).reshape(H, W, 3)[::-1]
        return np.clip(rgb, 0, 1), ms

class App:
    def __init__(self, root):
        self.root = root
        self.p = dict(DEFAULTS)
        self.rend = Renderer()
        self.res_name = "HD"
        self.hist = collections.deque(maxlen=60)      # rolling GPU ms
        self.tk_img = None
        # --- layout: left render, right scrollable sliders
        top = tk.Frame(root, bg="#111"); top.pack(fill="both", expand=True)
        left = tk.Frame(top, bg="#111"); left.pack(side="left")
        self.status = tk.Label(left, anchor="w", font=("Consolas", 12), bg="#000", fg="#6f6",
                               width=1, height=2, justify="left")
        self.status.pack(fill="x")
        self.canvas = tk.Canvas(left, width=DISP_W, height=int(DISP_W*9/16),
                                highlightthickness=0, bg="#111", cursor="tcross")
        self.canvas.pack()
        tk.Label(left, anchor="w", font=("Consolas", 9), bg="#181818", fg="#999", justify="left",
                 text=" L-drag pitch/yaw | R-drag sun el/az | wheel focal | 1/2/3 HD/FHD/4K | "
                      "b best  r reset  s save  l load  q quit").pack(fill="x")
        # scrollable slider panel
        right = tk.Frame(top, bg="#222"); right.pack(side="left", fill="both", expand=True)
        pcan = tk.Canvas(right, bg="#222", highlightthickness=0, width=430)
        sb = tk.Scrollbar(right, orient="vertical", command=pcan.yview)
        self.pframe = tk.Frame(pcan, bg="#222")
        self.pframe.bind("<Configure>", lambda e: pcan.configure(scrollregion=pcan.bbox("all")))
        pcan.create_window((0, 0), window=self.pframe, anchor="nw")
        pcan.configure(yscrollcommand=sb.set)
        pcan.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")
        pcan.bind_all("<MouseWheel>", lambda e: pcan.yview_scroll(int(-e.delta/120), "units"))
        self.vars, self.vlabels, self.meta = {}, {}, {}
        self._build_sliders()
        # mouse on render
        self.canvas.bind("<ButtonPress-1>", self._press); self.canvas.bind("<B1-Motion>", self._drag_cam)
        self.canvas.bind("<ButtonPress-3>", self._press); self.canvas.bind("<B3-Motion>", self._drag_sun)
        self.canvas.bind("<MouseWheel>", self._wheel)
        for seq, fn in (("1", lambda e: self._setres("HD")), ("2", lambda e: self._setres("FHD")),
                        ("3", lambda e: self._setres("4K")), ("b", self._load_best), ("r", self._reset),
                        ("s", self._save), ("l", self._load_preset), ("q", self.quit), ("<Escape>", self.quit)):
            root.bind(seq, fn)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.running = True
        self._tick()

    def _build_sliders(self):
        for gi, (label, items) in enumerate(GROUPS):
            tk.Label(self.pframe, text=label, font=("Consolas", 10, "bold"), bg="#333", fg="#fd6",
                     anchor="w").grid(row=self._grow(), column=0, columnspan=3, sticky="we", pady=(6, 1))
            for (k, lo, hi, kind) in items:
                row = self._grow()
                self.meta[k] = (lo, hi, kind)
                disp = math.degrees(self.p[k]) if kind == 'deg' else self.p[k]
                var = tk.DoubleVar(value=disp); self.vars[k] = var
                tk.Label(self.pframe, text=k, font=("Consolas", 9), bg="#222", fg="#ddd",
                         width=10, anchor="w").grid(row=row, column=0, sticky="w")
                sc = tk.Scale(self.pframe, from_=lo, to=hi, orient="horizontal", variable=var,
                              resolution=(hi-lo)/400.0, showvalue=False, length=250,
                              bg="#222", fg="#ddd", troughcolor="#444", highlightthickness=0,
                              command=lambda v, kk=k: self._on_slider(kk))
                sc.grid(row=row, column=1, sticky="we")
                vl = tk.Label(self.pframe, text=f"{disp:.3f}", font=("Consolas", 9), bg="#222",
                              fg="#8cf", width=8, anchor="e"); vl.grid(row=row, column=2, sticky="e")
                self.vlabels[k] = vl
    _row = 0
    def _grow(self):
        r = App._row; App._row += 1; return r

    def _on_slider(self, k):
        lo, hi, kind = self.meta[k]; v = self.vars[k].get()
        self.p[k] = math.radians(v) if kind == 'deg' else v
        self.vlabels[k].config(text=f"{v:.3f}")

    def _sync_sliders(self):
        for k, (lo, hi, kind) in self.meta.items():
            disp = math.degrees(self.p[k]) if kind == 'deg' else self.p[k]
            self.vars[k].set(disp); self.vlabels[k].config(text=f"{disp:.3f}")

    # ---- mouse
    def _press(self, e): self.lx, self.ly = e.x, e.y
    def _drag_cam(self, e):
        dx, dy = e.x-self.lx, e.y-self.ly; self.lx, self.ly = e.x, e.y
        self.p["pitch"] = max(-math.pi/2, min(math.pi/2, self.p["pitch"] - dy/300.0))
        self.p["yaw"] = (self.p["yaw"] + dx/300.0 + math.pi) % (2*math.pi) - math.pi
        self._sync_sliders()
    def _drag_sun(self, e):
        dx, dy = e.x-self.lx, e.y-self.ly; self.lx, self.ly = e.x, e.y
        self.p["sunEl"] = max(math.radians(-18), min(math.pi/2, self.p["sunEl"] - dy/300.0))
        self.p["sunAz"] = (self.p["sunAz"] + dx/300.0 + math.pi) % (2*math.pi) - math.pi
        self._sync_sliders()
    def _wheel(self, e):
        self.p["focal"] = max(0.2, min(3.0, self.p["focal"] * (1.1 if e.delta > 0 else 1/1.1)))
        self._sync_sliders()

    # ---- keys
    def _setres(self, name): self.res_name = name; self.hist.clear()
    def _load_best(self, e=None):
        try:
            j = json.load(open(BEST, encoding="utf-8")); self.p.update(j["shared"])
            pp = next(iter(j["per_photo"].values()))
            for k in ("uCovLow", "uCovMid", "uCovHigh", "uTypeAmp", "uTypeSeed", "expo"):
                if k in pp: self.p[k] = pp[k]
            self._sync_sliders()
        except Exception as ex: print("load best failed:", ex)
    def _reset(self, e=None): self.p = dict(DEFAULTS); self._sync_sliders()
    def _save(self, e=None):
        json.dump(self.p, open(PRESET, "w", encoding="utf-8"), indent=2); print("saved", PRESET)
    def _load_preset(self, e=None):
        try: self.p.update(json.load(open(PRESET, encoding="utf-8"))); self._sync_sliders()
        except Exception as ex: print("load preset failed:", ex)

    # ---- main loop
    def _tick(self):
        if not self.running: return
        W, H = RES[self.res_name]
        rgb, ms = self.rend.render(self.p, W, H)
        self.hist.append(ms)
        arr = sorted(self.hist); med = arr[len(arr)//2]; mn = arr[0]
        disp = Image.fromarray((rgb*255).astype(np.uint8)).resize((DISP_W, int(DISP_W*H/W)), Image.BILINEAR)
        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        warm = "  [warming up...]" if len(self.hist) < self.hist.maxlen else ""
        self.status.config(text=(
            f" {self.res_name} {W}x{H}   med {med:6.2f} ms = {1000/med:5.1f} fps   "
            f"(best {mn:5.2f} ms = {1000/mn:5.1f} fps){warm}\n"
            f" cam h={self.p['height']:.2f}km pitch={math.degrees(self.p['pitch']):+.0f} "
            f"yaw={math.degrees(self.p['yaw']):+.0f} FoV={2*math.degrees(math.atan(0.5/self.p['focal'])):.0f}   "
            f"sun el={math.degrees(self.p['sunEl']):+.0f} az={math.degrees(self.p['sunAz']):+.0f}"))
        self.root.after(1, self._tick)

    def quit(self, e=None):
        self.running = False
        try: glfw.terminate()
        except Exception: pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk(); root.title("Paper 6 - real-time parameter explorer"); root.configure(bg="#111")
    App(root); root.mainloop()
