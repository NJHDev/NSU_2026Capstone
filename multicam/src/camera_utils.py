import platform, subprocess, json, sys, cv2

def _windows_cam_names():
    try:
        ps = r"Get-CimInstance Win32_PnPEntity | Where-Object {$_.PNPClass -eq 'Image'} | Select-Object -ExpandProperty Name | ConvertTo-Json"
        out = subprocess.check_output(["powershell","-NoProfile","-Command", ps], stderr=subprocess.DEVNULL)
        data = json.loads(out.decode("utf-8"))
        if isinstance(data, list): return data
        if isinstance(data, str):  return [data]
    except Exception:
        pass
    return []

def _mac_cam_names():
    try:
        out = subprocess.check_output(["system_profiler","SPCameraDataType","-json"], stderr=subprocess.DEVNULL)
        data = json.loads(out.decode("utf-8")).get("SPCameraDataType", [])
        names = []
        for cam in data:
            name = cam.get("_name")
            if name: names.append(name)
            models = cam.get("spcamera_model")
            if isinstance(models, list): names.extend(models)
            elif isinstance(models, str): names.append(models)
        dedup = []
        for n in names:
            if n and n not in dedup: dedup.append(n)
        return dedup
    except Exception:
        return []

def list_connected_cameras(max_index=10):
    system = platform.system()
    names = []
    if system == "Windows":
        names = _windows_cam_names()
        backend = cv2.CAP_DSHOW
    elif system == "Darwin":
        names = _mac_cam_names()
        backend = cv2.CAP_AVFOUNDATION
    else:
        print("이 스크립트는 Windows/macOS 전용입니다.")
        backend = 0

    available = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, backend)
        if not cap.isOpened():
            if system == "Windows":
                cap.release()
                cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if not cap.isOpened():
            continue
        ok, _ = cap.read()
        cap.release()
        if not ok:
            continue
        name = names[i] if i < len(names) else "Unknown camera"
        available.append((i, name, backend))
    return available

def select_camera_index():
    system = platform.system()
    print(f"[INFO] Detected OS: {system}")

    cams = list_connected_cameras(max_index=10)
    if not cams:
        print("연결된 카메라를 찾지 못했습니다.")
        sys.exit(1)

    print("=== 연결된 카메라 목록 ===")
    for idx, name, _ in cams:
        print(f"{idx}: {name}")
    print("========================")
    
    # 3개의 카메라가 필수
    N_CAMERAS = 3
    if len(cams) < N_CAMERAS:
        print(f"오류: 최소 {N_CAMERAS}개의 카메라가 필요합니다. 현재 {len(cams)}개 연결됨.")
        sys.exit(1)

    valid = {idx for idx, _, _ in cams}
    selected_indices = []
    cam_names = ["cam1", "cam2", "cam3"]

    for i, name in enumerate(cam_names):
        while True:
            s = input(f"사용할 {name} ({i+1}/{N_CAMERAS})의 카메라 번호를 입력하세요 : ").strip()
            
            if s.isdigit():
                v = int(s)
                if v in valid:
                    if v in selected_indices:
                        print(f"{v}번 카메라는 이미 선택되었습니다. 다른 번호를 입력해 주세요.")
                        continue
                    selected_indices.append(v)
                    print(f"선택: {name}에 {v}번 카메라 연결.")
                    break
            print(f"잘못된 입력입니다. 가능한 번호: {sorted(list(valid - set(selected_indices)))}")

    return selected_indices # [cam1_index, cam2_index, cam3_index]

def open_camera(index):
    system = platform.system()
    backends = []
    if system == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    elif system == "Darwin":
        backends = [cv2.CAP_AVFOUNDATION]
    else:
        backends = [0]

    cap = None
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if cap.isOpened():
            return cap
        if cap: cap.release()
    return cv2.VideoCapture(index)