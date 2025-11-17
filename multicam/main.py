import os, sys, time, csv, platform
from datetime import datetime
import cv2
import mediapipe as mp
import time
import numpy as np

from src.camera_utils import select_camera_index, open_camera
from src.angles import compute_angles, EMA, compute_torso_angle
from src.draw import draw_pose # draw_pose만 가져와서 사용
from src.arduino import connect_arduino

# connect_arduino를 호출하여 serial 객체 생성
arduino = connect_arduino()

mp_pose = mp.solutions.pose

# 상수 정의
OUTPUT_W, OUTPUT_H = 1920, 1080
CAM_W, CAM_H = OUTPUT_W // 2, OUTPUT_H // 2
N_CAMERAS = 3

def draw_status_text(frame, all_data, log_on, mirror, cam_indices, fps):
    """우측 하단 사분면에 모든 카메라의 측정값을 텍스트로 표시합니다."""
    
    # 우측 하단 (960, 540) 좌표로 시작
    x_offset = OUTPUT_W // 2 + 10
    y_offset = OUTPUT_H // 2 + 30
    line_height = 28
    
    # 시스템 정보
    cv2.putText(frame, f"OS:{platform.system()}  FPS:{fps:0.1f}   Mirror:[{'ON' if mirror else 'OFF'}]   Log:[{'ON' if log_on else 'OFF'}]", 
                (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    y_offset += line_height * 2

    # 각 카메라의 데이터 출력
    for i in range(N_CAMERAS):
        data = all_data[i]
        cam_index = cam_indices[i]
        
        # 데이터 정리
        L_el, L_sh, R_el, R_sh, T_angle = data["angles"]
        rom = data["rom"]
        
        cam_title = f"--- CAM {i+1} ({cam_index}번) ---"
        # 카메라별 제목 색상 지정
        title_color = (255, 100, 0) if i == 0 else (100, 255, 0) if i == 1 else (0, 100, 255)
        cv2.putText(frame, cam_title, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_color, 2, cv2.LINE_AA)
        y_offset += line_height

        # L-Arm
        text_l = f"L: El:{'-' if L_el is None else f'{L_el:5.1f}'}  Sh:{'-' if L_sh is None else f'{L_sh:5.1f}'} | ROM El:{rom['L_el'][0]:.0f}/{rom['L_el'][1]:.0f}  Sh:{rom['L_sh'][0]:.0f}/{rom['L_sh'][1]:.0f}"
        cv2.putText(frame, text_l, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        y_offset += line_height

        # R-Arm
        text_r = f"R: El:{'-' if R_el is None else f'{R_el:5.1f}'}  Sh:{'-' if R_sh is None else f'{R_sh:5.1f}'} | ROM El:{rom['R_el'][0]:.0f}/{rom['R_el'][1]:.0f}  Sh:{rom['R_sh'][0]:.0f}/{rom['R_sh'][1]:.0f}"
        cv2.putText(frame, text_r, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        y_offset += line_height
        
        # Torso
        text_t = f"Torso Angle: {'-' if T_angle is None else f'{T_angle:5.1f}'} deg | ROM T:{rom['T_angle'][0]:.0f}/{rom['T_angle'][1]:.0f}"
        cv2.putText(frame, text_t, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
        y_offset += line_height * 2 # 다음 카메라와 간격

def init_cam_data():
    """각 카메라 데이터 셋의 초기화"""
    return {
        # EMA 스무딩 버퍼
        "ema": {
            "L_el": EMA(0.25), "L_sh": EMA(0.25),
            "R_el": EMA(0.25), "R_sh": EMA(0.25),
            "T_angle": EMA(0.25),
        },
        # ROM(최소/최대) 기록
        "rom": {k: [999.0, -999.0] for k in ["L_el", "L_sh", "R_el", "R_sh", "T_angle"]},
        # 현재 각도와 가시성 저장 (for display/logging)
        "angles": [None] * 5, # L_el, L_sh, R_el, R_sh, T_angle
        "visibility": [0.0] * 3, # L_vis, R_vis, T_vis
        "frame": None,
        "pose_results": None,
    }


def main():
    # 1. 카메라 선택 및 초기화
    cam_indices = select_camera_index() # [cam1_idx, cam2_idx, cam3_idx]
    
    caps = []
    for i, idx in enumerate(cam_indices):
        cap = open_camera(idx)
        if not cap.isOpened():
            print(f"카메라 {idx} (cam{i+1}) 열기 실패.")
            sys.exit(1)
        # 4분할 화면에 맞춰 해상도 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        caps.append(cap)

    # 2. 전역 변수 초기화
    mirror = False
    log_on = False
    os.makedirs("logs", exist_ok=True)
    csv_writer = None
    csv_file = None
    
    # 3개 카메라 데이터 리스트
    all_cam_data = [init_cam_data() for _ in range(N_CAMERAS)]

    fps_t = time.time()
    fps = 0.0
    
    # 3. MediaPipe Pose 초기화
    with mp_pose.Pose(model_complexity=0, enable_segmentation=False, smooth_landmarks=True) as pose_processor:
        while True:
            all_frames = [] # 3개 카메라의 처리된 프레임을 저장
            
            # --- 4. 3개 카메라 프레임 처리 루프 ---
            for i in range(N_CAMERAS):
                cap = caps[i]
                data = all_cam_data[i]

                ok, frame = cap.read()
                if not ok:
                    # 카메라 캡처 실패 시 블랙 화면으로 대체
                    data["frame"] = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8) 
                    all_frames.append(data["frame"])
                    continue
                
                if mirror:
                    frame = cv2.flip(frame, 1)

                # 프레임 크기 조정 (혹시 모를 캡처 오류 대비)
                if frame.shape[0] != CAM_H or frame.shape[1] != CAM_W:
                    frame = cv2.resize(frame, (CAM_W, CAM_H))
                
                # Pose 인식
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = pose_processor.process(rgb)
                rgb.flags.writeable = True
                
                data["pose_results"] = res
                
                L_el, L_sh, R_el, R_sh, T_angle = [None] * 5
                L_vis, R_vis, T_vis = 0.0, 0.0, 0.0

                if res.pose_landmarks:
                    # --- 각도 계산 및 스무딩 ---
                    L_el_raw, L_sh_raw, L_vis = compute_angles(res.pose_landmarks, res.pose_world_landmarks, "left")
                    R_el_raw, R_sh_raw, R_vis = compute_angles(res.pose_landmarks, res.pose_world_landmarks, "right")
                    T_angle_raw, T_vis = compute_torso_angle(res.pose_landmarks, res.pose_world_landmarks)

                    L_el = data["ema"]["L_el"].update(L_el_raw)
                    L_sh = data["ema"]["L_sh"].update(L_sh_raw)
                    R_el = data["ema"]["R_el"].update(R_el_raw)
                    R_sh = data["ema"]["R_sh"].update(R_sh_raw)
                    T_angle = data["ema"]["T_angle"].update(T_angle_raw)
                    
                    # --- ROM 갱신 ---
                    angles_to_update = {
                        "L_el": L_el, "L_sh": L_sh,
                        "R_el": R_el, "R_sh": R_sh,
                        "T_angle": T_angle,
                    }
                    for k, v in angles_to_update.items():
                        if v is None:
                            continue
                        data["rom"][k][0] = min(data["rom"][k][0], v)
                        data["rom"][k][1] = max(data["rom"][k][1], v)
                        
                    # --- 팔 라인 + 각도 라벨 그리기 ---
                    if None not in (L_el, L_sh, R_el, R_sh):
                        draw_pose(
                            frame,
                            res.pose_landmarks,
                            (L_el, L_sh, L_vis),
                            (R_el, R_sh, R_vis),
                        )
                else:
                    # 사람 미인식 안내
                    cv2.putText(frame, f"Cam {i+1}: No person detected", (40, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2, cv2.LINE_AA)

                # 데이터 저장
                data["angles"] = [L_el, L_sh, R_el, R_sh, T_angle]
                data["visibility"] = [L_vis, R_vis, T_vis]
                data["frame"] = frame
                all_frames.append(frame)

            # --- 5. 아두이노 전송 (cam1의 값만 사용) ---
            data1 = all_cam_data[0]["angles"]
            R_el1, R_sh1, T_angle1 = data1[2], data1[3], data1[4]
            
            if arduino is not None and None not in (R_el1, R_sh1, T_angle1):
                send_data = f"{R_el1:0.0f},{R_sh1:0.0f},{T_angle1:0.0f}\n"
                arduino.write(send_data.encode())

            # --- 6. CSV 로깅 (모든 카메라 데이터 기록) ---
            if log_on:
                if csv_writer is None:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = f"logs/angles_{ts}.csv"
                    csv_file = open(path, "w", newline="", encoding="utf-8")
                    csv_writer = csv.writer(csv_file)
                    header = ["time"]
                    for i in range(N_CAMERAS):
                        header.extend([f"C{i+1}_L_el", f"C{i+1}_L_sh", f"C{i+1}_R_el", f"C{i+1}_R_sh", f"C{i+1}_T_angle", f"C{i+1}_L_vis", f"C{i+1}_R_vis", f"C{i+1}_T_vis"])
                    csv_writer.writerow(header)
                
                row = [time.time()]
                for data in all_cam_data:
                    row.extend(data["angles"])
                    row.extend(data["visibility"])
                csv_writer.writerow(row)
            
            # --- 7. 4분할 화면 합성 (1920x1080) ---
            if len(all_frames) == N_CAMERAS:
                
                # 4분할 최종 프레임 생성 (우측 하단은 텍스트 전용)
                final_frame = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8) 
                
                # 영역 1: 좌측 상단 (cam1)
                final_frame[:CAM_H, :CAM_W] = all_frames[0]
                
                # 영역 2: 우측 상단 (cam2)
                final_frame[:CAM_H, CAM_W:] = all_frames[1]
                
                # 영역 3: 좌측 하단 (cam3)
                final_frame[CAM_H:, :CAM_W] = all_frames[2]
                
                # 영역 4: 우측 하단 (텍스트 오버레이) - 검은색 바탕에 텍스트 표시
                draw_status_text(final_frame, all_cam_data, log_on, mirror, cam_indices, fps)
                
                cv2.imshow("Multi-Cam Pose Analysis (1920x1080)", final_frame)
            
            # --- 8. FPS 계산 ---
            now = time.time()
            dt = now - fps_t
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            fps_t = now

            # --- 9. 키 입력 ---
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
                # 모든 ROM 초기화
                for data in all_cam_data:
                    data["rom"] = {k: [999.0, -999.0] for k in data["rom"].keys()}

    # --- 10. 종료 처리 ---
    if csv_file:
        csv_file.close()
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    
    if arduino is not None:
        arduino.close()


if __name__ == "__main__":
    main()