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

    valid = {idx for idx, _, _ in cams}
    while True:
        s = input("사용할 카메라 번호를 입력하세요 : ").strip()
        if s == "":
            choice = min(valid)
            print(f"(기본값) {choice}번 카메라를 사용합니다.")
            return choice
        if s.isdigit():
            v = int(s)
            if v in valid:
                return v
        print(f"잘못된 입력입니다. 가능한 번호: {sorted(valid)}")

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
