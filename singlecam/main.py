import os, sys, time, csv, platform
from datetime import datetime
import cv2
import mediapipe as mp
import serial
import time
import numpy as np

from src.camera_utils import select_camera_index, open_camera
from src.angles import compute_angles, EMA, compute_torso_angle # compute_torso_angle 추가
from src.draw import draw_pose

# Arduino serial ("COM0" or "/dev/ttyUSB0")
arduino = serial.Serial('/dev/cu.usbserial-110', 9600)
time.sleep(2)  #시리얼 연결 대기

mp_pose = mp.solutions.pose

def main():
    cam_index = select_camera_index()
    cap = open_camera(cam_index)
    if not cap.isOpened():
        print(f"카메라 {cam_index} 열기 실패.")
        sys.exit(1)

    # 기본 캡처 설정(원하면 조정)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    mirror = False
    log_on = False
    os.makedirs("logs", exist_ok=True)
    csv_writer = None
    csv_file = None

    # ✅ EMA(지수이동평균) 스무딩 버퍼 (Wr 제거, T_angle 추가)
    ema = {
        "L_el": EMA(0.25), "L_sh": EMA(0.25),
        "R_el": EMA(0.25), "R_sh": EMA(0.25),
        "T_angle": EMA(0.25),
    }

    # ✅ ROM(최소/최대) 기록 (Wr 제거, T_angle 추가)
    rom = {k: [999.0, -999.0] for k in ema.keys()}

    fps_t = time.time()
    fps = 0.0

    with mp_pose.Pose(model_complexity=1, enable_segmentation=False, smooth_landmarks=True) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = pose.process(rgb)
            rgb.flags.writeable = True

            if res.pose_landmarks:
                # --- 각도 계산: (elbow, shoulder, visibility) - Wr 제거 ---
                L_el_raw, L_sh_raw, L_vis = compute_angles(res.pose_landmarks, res.pose_world_landmarks, "left")
                R_el_raw, R_sh_raw, R_vis = compute_angles(res.pose_landmarks, res.pose_world_landmarks, "right")

                # --- 상체 회전 각도 계산 ---
                T_angle_raw, T_vis = compute_torso_angle(res.pose_landmarks, res.pose_world_landmarks)

                # --- EMA 스무딩 ---
                L_el = ema["L_el"].update(L_el_raw)
                L_sh = ema["L_sh"].update(L_sh_raw)
                # L_wr 제거
                R_el = ema["R_el"].update(R_el_raw)
                R_sh = ema["R_sh"].update(R_sh_raw)
                # R_wr 제거
                T_angle = ema["T_angle"].update(T_angle_raw) # T_angle 스무딩

                # --- 팔 라인 + 각도 라벨(El/Sh) 그리기 ---
                if None not in (L_el, L_sh, R_el, R_sh):
                    draw_pose(
                        frame,
                        res.pose_landmarks,
                        (L_el, L_sh, L_vis), # Wr 제거
                        (R_el, R_sh, R_vis), # Wr 제거
                    )

                # --- ROM 갱신 ---
                for k, v in {
                    "L_el": L_el, "L_sh": L_sh,
                    "R_el": R_el, "R_sh": R_sh,
                    "T_angle": T_angle, # T_angle 추가
                }.items():
                    if v is None:
                        continue
                    rom[k][0] = min(rom[k][0], v)
                    rom[k][1] = max(rom[k][1], v)
                
                # --- 아두이노로 각도 전송 ---
                # (R_el 각도를 시리얼로 보냄. 원하는 각도로 변경 가능)
                if R_el is not None:
                    arduino.write(f"{R_el:0.0f}\n".encode())

                # --- 화면 텍스트 오버레이 ---
                def put(y, text):
                    cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2, cv2.LINE_AA)

                put(28, f"OS:{platform.system()}  Cam:{cam_index}  FPS:{fps:0.1f}   Mirror:[{'ON' if mirror else 'OFF'}]   Log:[{'ON' if log_on else 'OFF'}]")
                # Wr 제거
                put(56, f"L: El:{'-' if L_el is None else f'{L_el:5.1f}'}  Sh:{'-' if L_sh is None else f'{L_sh:5.1f}'}")
                put(84, f"R: El:{'-' if R_el is None else f'{R_el:5.1f}'}  Sh:{'-' if R_sh is None else f'{R_sh:5.1f}'}")
                
                # T_angle 추가 및 ROM 통합
                put(112, f"Torso Angle: {'-' if T_angle is None else f'{T_angle:5.1f}'} deg  ROM:{rom['T_angle'][0]:.0f}/{rom['T_angle'][1]:.0f}")

                # Wr 제거 및 Y좌표 조정
                put(140, f"L-ROM El:{rom['L_el'][0]:.0f}/{rom['L_el'][1]:.0f}  Sh:{rom['L_sh'][0]:.0f}/{rom['L_sh'][1]:.0f}")
                put(168, f"R-ROM El:{rom['R_el'][0]:.0f}/{rom['R_el'][1]:.0f}  Sh:{rom['R_sh'][0]:.0f}/{rom['R_sh'][1]:.0f}")

                # --- CSV 로깅 ---
                if log_on:
                    if csv_writer is None:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        path = f"logs/angles_{ts}.csv"
                        csv_file = open(path, "w", newline="", encoding="utf-8")
                        csv_writer = csv.writer(csv_file)
                        # T_angle 추가, Wr 제거
                        csv_writer.writerow(
                            ["time", "L_el", "L_sh", "R_el", "R_sh", "T_angle", "L_vis", "R_vis", "T_vis"]
                        )
                    # T_angle 추가, Wr 제거
                    csv_writer.writerow([time.time(), L_el, L_sh, R_el, R_sh, T_angle, L_vis, R_vis, T_vis])
            else:
                # 사람 미인식 안내
                cv2.putText(frame, "No person detected", (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2, cv2.LINE_AA)

            # --- FPS 계산 ---
            now = time.time()
            dt = now - fps_t
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            fps_t = now

            # --- 키 입력 ---
            cv2.imshow("Pose Angles (Arms: El / Sh / Torso)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("s"):
                mirror = not mirror
            elif k == ord("l"):
                log_on = not log_on
                if not log_on and csv_file:
                    csv_file.close()
                    csv_file = None
                    csv_writer = None
            elif k == ord("r"):
                # T_angle 포함 ROM 초기화
                rom = {k: [999.0, -999.0] for k in rom.keys()}

    if csv_file:
        csv_file.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()