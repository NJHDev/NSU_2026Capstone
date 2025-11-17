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
    Returns (elbow, shoulder, visibility)
    elbow:   ∠(Shoulder–Elbow–Wrist)
    shoulder: angle between UpperArm(S→E) and TorsoDown(S→Hip)
    """
    idx = mp_pose.PoseLandmark
    if side.lower().startswith("l"):
        SH, EL, WR, HIP = idx.LEFT_SHOULDER, idx.LEFT_ELBOW, idx.LEFT_WRIST, idx.LEFT_HIP
        LHIP, RHIP = idx.LEFT_HIP, idx.RIGHT_HIP
    else:
        SH, EL, WR, HIP = idx.RIGHT_SHOULDER, idx.RIGHT_ELBOW, idx.RIGHT_WRIST, idx.RIGHT_HIP
        LHIP, RHIP = idx.LEFT_HIP, idx.RIGHT_HIP

    if plm is None: return None, None, 0.0
    vsrc = plm.landmark
    needed = [SH, EL, WR, HIP, LHIP, RHIP]
    vis_vals = []
    for i in needed:
        vis_vals.append(getattr(vsrc[i], "visibility", 0.0))
        
    if not all(_visible(vsrc[i]) for i in [SH, EL, WR]):
        return None, None, float(np.mean(vis_vals))

    src = pwm or plm
    pts = src.landmark

    S, E, W = _g(pts[SH]), _g(pts[EL]), _g(pts[WR])
    H  = _g(pts[HIP])

    # Elbow: ∠SEW
    elbow = _angle_3pts(S, E, W)

    # Shoulder: UpperArm(S→E) vs TorsoDown(S→H)
    upper = E - S
    torso_down = H - S
    shoulder = _angle_between(upper, torso_down)

    visibility = float(np.mean(vis_vals))
    return elbow, shoulder, visibility


def compute_torso_angle(plm, pwm, eps=1e-9):
    """
    상체 회전 각도 (어깨선과 수평선 사이의 각도)를 계산합니다.
    """
    idx = mp_pose.PoseLandmark
    RS, LS = idx.RIGHT_SHOULDER, idx.LEFT_SHOULDER

    if plm is None: return None, 0.0
    vsrc = plm.landmark
    
    if not (_visible(vsrc[RS]) and _visible(vsrc[LS])):
        visibility = float((getattr(vsrc[RS], "visibility", 0.0) + getattr(vsrc[LS], "visibility", 0.0)) / 2.0)
        return None, visibility

    src = pwm or plm
    pts = src.landmark
    
    R = _g(pts[RS])
    L = _g(pts[LS])
    
    shoulder_vector = L - R
    
    x_component = shoulder_vector[0]
    y_component = shoulder_vector[1]
    
    angle_rad = np.arctan2(y_component, x_component)
    angle_deg = np.degrees(angle_rad)
    
    visibility = float((getattr(vsrc[RS], "visibility", 0.0) + getattr(vsrc[LS], "visibility", 0.0)) / 2.0)

    return angle_deg, visibility


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