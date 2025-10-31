import numpy as np
from math import acos, degrees
import mediapipe as mp

mp_pose = mp.solutions.pose

def angle_between(u, v, eps=1e-9):
    u = np.array(u, dtype=float)
    v = np.array(v, dtype=float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < eps or nv < eps: return None
    c = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return degrees(acos(c))

def angle_3pts(a, b, c):
    return angle_between(np.array(a) - np.array(b), np.array(c) - np.array(b))

def project_to_plane(v, normal, eps=1e-9):
    n, v = np.array(normal, float), np.array(v, float)
    nn = np.linalg.norm(n)
    if nn < eps: return v
    n = n / nn
    return v - np.dot(v, n) * n

def gl(pt):
    return np.array([pt.x, pt.y, getattr(pt, "z", 0.0)], dtype=float)

def visible(lm, thr=0.5):
    return (getattr(lm, "visibility", 1.0) or 1.0) >= thr

class EMA:
    def __init__(self, alpha=0.25): self.alpha, self.v = alpha, None
    def update(self, x):
        if x is None: return self.v
        self.v = x if self.v is None else (self.alpha*x + (1-self.alpha)*self.v)
        return self.v

def compute_angles(plm, pwm, side="left"):
    idx = mp_pose.PoseLandmark
    if side.lower().startswith("l"):
        SH, EL, WR, HIP, LHIP, RHIP = idx.LEFT_SHOULDER, idx.LEFT_ELBOW, idx.LEFT_WRIST, idx.LEFT_HIP, idx.LEFT_HIP, idx.RIGHT_HIP
    else:
        SH, EL, WR, HIP, LHIP, RHIP = idx.RIGHT_SHOULDER, idx.RIGHT_ELBOW, idx.RIGHT_WRIST, idx.RIGHT_HIP, idx.LEFT_HIP, idx.RIGHT_HIP

    if plm is None: return None, None, None, 0.0
    vsrc = plm.landmark
    need = [SH, EL, WR, HIP, LHIP, RHIP]
    if not all(visible(vsrc[i]) for i in need):
        vis = np.mean([getattr(vsrc[i], "visibility", 0.0) for i in need])
        return None, None, None, vis

    src = pwm or plm
    pts = src.landmark
    S, E, W = gl(pts[SH]), gl(pts[EL]), gl(pts[WR])
    H, LH, RH = gl(pts[HIP]), gl(pts[LHIP]), gl(pts[RHIP])

    elbow = angle_3pts(S, E, W)
    upper = E - S
    torso_down = H - S
    sh_abd = angle_between(upper, torso_down)

    pelvis_lr = RH - LH
    ua_proj = project_to_plane(upper, pelvis_lr)
    td_proj = project_to_plane(torso_down, pelvis_lr)
    sh_flex = angle_between(ua_proj, td_proj)

    vis = np.mean([getattr(vsrc[i], "visibility", 0.0) for i in need])
    return elbow, sh_abd, sh_flex, vis
