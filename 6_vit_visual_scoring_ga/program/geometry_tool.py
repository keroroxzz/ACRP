"""Interactive per-photo geometry tool (camera pitch/FoV + sun az/el).

Left panel = reference photo, right panel = render with that photo's fixed geometry.

  SPACE        toggle: right panel shows the pure render  <->  render with the photo
               overlaid at 50% (for checking horizon / sun alignment)
  LEFT-drag    sun azimuth (full canvas width = 360 deg) + elevation (full height = 180 deg)
  RIGHT-drag   vertical = camera pitch (-60..+90 deg), horizontal = FoV / focal (0.2..3.0)
  <- / ->      previous / next photo (changes are saved automatically)
  c            toggle sparse "geometry" clouds <-> this photo's optimized cloud params
  s            force save,   r  reset this photo to the built-in default,   q/Esc  quit

Edits are written to `fixed_geo.json` next to this file; optimize_multiphoto.py picks that
file up automatically (it overrides the built-in FIXED_GEO table).

Run with the `dino` env (or any env with glfw/PyOpenGL/Pillow):  python geometry_tool.py
"""
from __future__ import annotations
import os, glob, json, math
import numpy as np, glfw
from OpenGL.GL import *
from PIL import Image, ImageTk
import tkinter as tk

HERE   = os.path.dirname(os.path.abspath(__file__))
PDIR   = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reference_photos"))
SHADER = os.path.join(HERE, "cumulus_multilayer.glsl")
CFG    = os.path.join(HERE, "fixed_geo.json")
BEST   = os.path.join(HERE, "..", "results", "multi_best_dual.json")
VS = ("#version 330 core\nvoid main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));"
      " gl_Position=vec4(p*2.0-1.0,0.0,1.0);}\n")
W, H = 640, 480                      # per-panel render size (4:3)
HEIGHT = 0.06                        # camera height (fixed, matches the optimizer)
PITCH_MIN, PITCH_MAX = math.radians(-60.0), math.radians(90.0)
FOCAL_MIN, FOCAL_MAX = 0.2, 3.0

# ---------------------------------------------------------------- defaults / config
def builtin_geo():
    """Read the FIXED_GEO table out of optimize_multiphoto.py without importing it (no model load)."""
    src = open(os.path.join(HERE, "optimize_multiphoto.py"), encoding="utf-8").read()
    s = src.index("FIXED_GEO={"); e = src.index("}", s) + 1
    ns = {}; exec(src[s:e], ns)
    return ns["FIXED_GEO"]

def load_shared():
    """Shared cloud/atmosphere params from the latest best run (falls back to sane defaults)."""
    d = dict(uAtmDens=2.0, uSunI=5.5, uHaze=0.10, uAmbStr=0.5, uDens=6.0, uSigma=4.2, uSilver=2.5,
             uMsBoost=0.5, uDetAmp=0.55, uDet2Amp=0.30, uDet3Amp=0.18, uCovFreq=0.22, uWarp=0.8,
             uLowBase=0.9, uLowTop=3.0, uMidBase=4.4, uMidTop=5.4, uHiBase=6.6, uHiTop=7.8)
    best = {}
    try:
        j = json.load(open(BEST, encoding="utf-8")); d.update(j["shared"]); best = j.get("per_photo", {})
    except Exception:
        pass
    return d, best

SPARSE = dict(uCovLow=0.24, uCovMid=0.0, uCovHigh=0.0, uTypeAmp=0.10, uTypeSeed=2.0, expo=0.85)

# ---------------------------------------------------------------- GL
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
        self.win = glfw.create_window(W, H, "gl", None, None)
        glfw.make_context_current(self.win)
        tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, None)
        fbo = glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
        glViewport(0, 0, W, H)
        p = glCreateProgram()
        glAttachShader(p, _sh(VS, GL_VERTEX_SHADER))
        glAttachShader(p, _sh(open(SHADER, encoding="utf-8").read(), GL_FRAGMENT_SHADER))
        glLinkProgram(p)
        if glGetProgramiv(p, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError(glGetProgramInfoLog(p).decode())
        self.prog = p; self.vao = glGenVertexArrays(1)

    def render(self, shared, geo, cloud):
        glUseProgram(self.prog); U = lambda n: glGetUniformLocation(self.prog, n)
        eye = np.array([0, HEIGHT, 0], np.float32); yaw = 0.0
        d = np.array([math.sin(yaw)*math.cos(geo['pitch']), math.sin(geo['pitch']),
                      math.cos(yaw)*math.cos(geo['pitch'])], np.float32)
        f = d/np.linalg.norm(d); r = np.cross([0, 1, 0], f); r /= np.linalg.norm(r); u = np.cross(f, r)
        cam = np.stack([r, u, f], 1).astype(np.float32)
        sd = np.array([math.cos(geo['sunEl'])*math.sin(geo['sunAz']), math.sin(geo['sunEl']),
                       math.cos(geo['sunEl'])*math.cos(geo['sunAz'])], np.float32)
        glUniform2f(U("uRes"), float(W), float(H)); glUniform3f(U("uEye"), *[float(z) for z in eye])
        glUniformMatrix3fv(U("uCam"), 1, GL_FALSE, cam.T.copy()); glUniform1f(U("uFocal"), float(geo['focal']))
        glUniform3f(U("uSunDir"), *[float(z) for z in sd])
        glUniform1f(U("uExposure"), float(cloud.get('expo', 0.85)))
        for k, v in shared.items(): glUniform1f(U(k), float(v))
        for k in ("uCovLow", "uCovMid", "uCovHigh", "uTypeAmp", "uTypeSeed"):
            glUniform1f(U(k), float(cloud[k]))
        glBindVertexArray(self.vao); glDrawArrays(GL_TRIANGLES, 0, 3); glFinish()
        buf = glReadPixels(0, 0, W, H, GL_RGB, GL_FLOAT)
        return np.frombuffer(buf, np.float32).reshape(H, W, 3)[::-1]

# ---------------------------------------------------------------- app
class App:
    def __init__(self, root):
        self.root = root
        self.files = sorted(glob.glob(os.path.join(PDIR, "*.jpg")))
        if not self.files: raise SystemExit(f"no photos in {PDIR}")
        self.names = [os.path.basename(f)[:-4] for f in self.files]
        self.defaults = builtin_geo()
        self.shared, self.best_pp = load_shared()
        self.geo = {n: dict(self.defaults.get(n, dict(pitch=0.25, sunAz=0.0, sunEl=0.5, focal=1.05)))
                    for n in self.names}
        if os.path.exists(CFG):                                   # saved edits win
            try:
                for n, g in json.load(open(CFG, encoding="utf-8")).items():
                    if n in self.geo: self.geo[n].update({k: float(v) for k, v in g.items()})
            except Exception: pass
        self.idx = 0; self.overlay = False; self.use_best_clouds = False
        self.photos = {}                                          # lazy cache of resized photos
        self.rend = Renderer()
        self.canvas = tk.Canvas(root, width=2*W, height=H, highlightthickness=0, bg="#111")
        self.canvas.pack()
        self.status = tk.Label(root, anchor="w", font=("Consolas", 11), bg="#222", fg="#eee")
        self.status.pack(fill="x")
        self.help = tk.Label(root, anchor="w", font=("Consolas", 9), bg="#181818", fg="#999",
                             text=" SPACE overlay | L-drag sun az/el | R-drag pitch(V)/FoV(H) | <- -> photo | c clouds | r reset | s save | q quit")
        self.help.pack(fill="x")
        for seq, fn in (("<space>", self.on_space), ("<Left>", self.prev), ("<Right>", self.next),
                        ("<Key-c>", self.on_c), ("<Key-r>", self.on_r), ("<Key-s>", self.on_s),
                        ("<Key-q>", self.quit), ("<Escape>", self.quit)):
            root.bind(seq, fn)
        self.canvas.bind("<ButtonPress-1>", self.press); self.canvas.bind("<B1-Motion>", self.drag_sun)
        self.canvas.bind("<ButtonPress-3>", self.press); self.canvas.bind("<B3-Motion>", self.drag_cam)
        self.canvas.bind("<ButtonRelease-1>", self.release); self.canvas.bind("<ButtonRelease-3>", self.release)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.redraw()

    # ---- data helpers
    def name(self): return self.names[self.idx]
    def cur(self):  return self.geo[self.name()]
    def photo(self):
        n = self.name()
        if n not in self.photos:
            self.photos[n] = (np.asarray(Image.open(self.files[self.idx]).convert("RGB")
                              .resize((W, H), Image.LANCZOS), np.float32)/255.0)
        return self.photos[n]
    def clouds(self):
        if self.use_best_clouds:
            pp = self.best_pp.get(self.name())
            if pp: return {k: pp.get(k, SPARSE[k]) for k in
                           ("uCovLow", "uCovMid", "uCovHigh", "uTypeAmp", "uTypeSeed", "expo")}
        return SPARSE

    def save(self):
        json.dump(self.geo, open(CFG, "w", encoding="utf-8"), indent=2)

    # ---- rendering
    def redraw(self):
        ph = self.photo()
        rnd = self.rend.render(self.shared, self.cur(), self.clouds())
        right = 0.5*rnd + 0.5*ph if self.overlay else rnd
        img = np.clip(np.concatenate([ph, right], 1), 0, 1)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray((img*255).astype(np.uint8)))
        self.canvas.delete("all"); self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        g = self.cur(); deg = math.degrees
        fov = 2*math.degrees(math.atan(0.5/g['focal']))
        self.status.config(text=(
            f" [{self.idx+1}/{len(self.files)}] {self.name()}    "
            f"FoV {fov:5.1f}° (focal {g['focal']:.3f})   cam pitch {deg(g['pitch']):6.1f}°   "
            f"sun el {deg(g['sunEl']):6.1f}°   sun az {deg(g['sunAz']):7.1f}°   "
            f"[{'overlay' if self.overlay else 'render'}|{'best' if self.use_best_clouds else 'sparse'} clouds]"))

    # ---- events
    def press(self, e):  self.lx, self.ly = e.x, e.y
    def release(self, e): self.save()
    def drag_sun(self, e):
        dx, dy = e.x - self.lx, e.y - self.ly; self.lx, self.ly = e.x, e.y
        g = self.cur()
        g['sunAz'] += dx/(2.0*W) * 2*math.pi                       # full canvas width = 360 deg
        g['sunEl'] -= dy/H * math.pi                               # full height = 180 deg
        g['sunAz'] = (g['sunAz'] + math.pi) % (2*math.pi) - math.pi
        g['sunEl'] = max(-math.pi/2, min(math.pi/2, g['sunEl']))
        self.redraw()
    def drag_cam(self, e):
        dx, dy = e.x - self.lx, e.y - self.ly; self.lx, self.ly = e.x, e.y
        g = self.cur()
        g['pitch'] = max(PITCH_MIN, min(PITCH_MAX, g['pitch'] - dy/H*(PITCH_MAX-PITCH_MIN)))
        g['focal'] = max(FOCAL_MIN, min(FOCAL_MAX, g['focal'] + dx/(2.0*W)*(FOCAL_MAX-FOCAL_MIN)))
        self.redraw()
    def on_space(self, e): self.overlay = not self.overlay; self.redraw()
    def on_c(self, e): self.use_best_clouds = not self.use_best_clouds; self.redraw()
    def on_r(self, e):
        self.geo[self.name()] = dict(self.defaults.get(self.name(), self.cur())); self.save(); self.redraw()
    def on_s(self, e): self.save(); self.redraw()
    def prev(self, e): self.save(); self.idx = (self.idx-1) % len(self.files); self.redraw()
    def next(self, e): self.save(); self.idx = (self.idx+1) % len(self.files); self.redraw()
    def quit(self, e=None):
        self.save()
        try: glfw.terminate()
        except Exception: pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk(); root.title("Paper 6 — per-photo geometry (camera / sun)"); root.resizable(False, False)
    App(root); root.mainloop()
