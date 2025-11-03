import numpy as np
from math import acos, degrees
import mediapipe as mp

mp_pose = mp.solutions.pose

def _angle_between(u, v, eps=1e-9):
    u = np.array(u, dtype=float); v = np.array(v, dtype=float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < eps or nv < eps: return None
    c = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return degrees(acos(c))

def _angle_3pts(a, b, c):
    # ∠ABC (BA·BC)
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    return _angle_between(ba, bc)

def _g(pt):
    return np.array([pt.x, pt.y, getattr(pt, "z", 0.0)], dtype=float)

def _visible(lm, thr=0.5):
    return (getattr(lm, "visibility", 1.0) or 1.0) >= thr

def compute_angles(plm, pwm, side="left"):
    """
    Returns (elbow, shoulder, wrist, visibility)
    elbow:   ∠(Shoulder–Elbow–Wrist)
    shoulder: angle between UpperArm(S→E) and TorsoDown(S→Hip)
    wrist:   angle between Forearm(E→W) and Hand direction(W→Index)
    """
    idx = mp_pose.PoseLandmark
    if side.lower().startswith("l"):
        SH, EL, WR, HIP, INDEX = idx.LEFT_SHOULDER, idx.LEFT_ELBOW, idx.LEFT_WRIST, idx.LEFT_HIP, idx.LEFT_INDEX
        LHIP, RHIP = idx.LEFT_HIP, idx.RIGHT_HIP
    else:
        SH, EL, WR, HIP, INDEX = idx.RIGHT_SHOULDER, idx.RIGHT_ELBOW, idx.RIGHT_WRIST, idx.RIGHT_HIP, idx.RIGHT_INDEX
        LHIP, RHIP = idx.LEFT_HIP, idx.RIGHT_HIP

    if plm is None: return None, None, None, 0.0
    vsrc = plm.landmark
    needed = [SH, EL, WR, HIP, INDEX, LHIP, RHIP]
    vis_vals = []
    for i in needed:
        vis_vals.append(getattr(vsrc[i], "visibility", 0.0))
    if not all(_visible(vsrc[i]) for i in [SH, EL, WR]):
        return None, None, None, float(np.mean(vis_vals))

    # world 있으면 그걸 우선 사용(스케일 감도↓)
    src = pwm or plm
    pts = src.landmark

    S, E, W = _g(pts[SH]), _g(pts[EL]), _g(pts[WR])
    H, I  = _g(pts[HIP]), _g(pts[INDEX])

    # Elbow: ∠SEW
    elbow = _angle_3pts(S, E, W)

    # Shoulder: 상완(S→E) vs 몸통 아래(S→H)
    upper = E - S
    torso_down = H - S
    shoulder = _angle_between(upper, torso_down)

    # Wrist: 전완(E→W) vs 손 방향(W→Index)
    forearm = W - E
    hand_dir = I - W
    wrist = _angle_between(forearm, hand_dir)

    visibility = float(np.mean(vis_vals))
    return elbow, shoulder, wrist, visibility

class EMA:
    """지수 이동 평균 필터 (각도 안정화용)"""
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.v = None

    def update(self, x):
        if x is None:
            return self.v
        if self.v is None:
            self.v = x
        else:
            self.v = self.alpha * x + (1 - self.alpha) * self.v
        return self.v
