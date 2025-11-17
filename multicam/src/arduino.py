import serial
import serial.tools.list_ports
import time
import sys

def select_arduino_port():
    # 시스템에서 사용 가능한 시리얼 포트 중 아두이노 포트를 선택합니다.
    # 일반적인 아두이노 Vendor ID 목록
    ARDUINO_VID = [
        0x2341,  # Arduino Vendor ID
        0x1A86,  # CH340/CH341
        0x0403,  # FTDI
        0x10C4,  # CP210x
    ]

    print("사용 가능한 시리얼 포트 검색 중...")
    ports = serial.tools.list_ports.comports()
    
    # 아두이노 패턴과 일치하는 포트 식별
    arduino_ports = []
    for port in ports:
        # VID/PID 또는 설명(Description)에 'arduino'가 포함된 경우
        if port.vid in ARDUINO_VID or "arduino" in port.description.lower():
            arduino_ports.append(port)
        
    if arduino_ports:
        # 아두이노 추정 포트가 발견된 경우
        
        if len(arduino_ports) == 1:
            # 포트가 하나일 경우, 자동으로 선택
            selected_port = arduino_ports[0].device
            print(f"아두이노 포트 : {selected_port} ({arduino_ports[0].description})")
            return selected_port
        else:
            # 포트가 여러 개일 경우, 사용자에게 선택하도록 함
            print("여러 개의 아두이노 포트가 발견되었습니다. 연결할 포트를 선택하세요:")
            
            for i, port in enumerate(arduino_ports):
                print(f"  [{i+1}] {port.device} ({port.description})")
            
            while True:
                try:
                    choice = input("연결할 포트 번호를 입력하거나 ('q'를 눌러 취소): ")
                    if not choice or choice.lower() == 'q':
                        print("포트 선택을 취소합니다.")
                        return None 
                    
                    index = int(choice) - 1
                    if 0 <= index < len(arduino_ports):
                        selected_port = arduino_ports[index].device
                        print(f"선택한 포트: {selected_port}")
                        return selected_port
                    else:
                        print("잘못된 번호입니다. 다시 입력해 주세요.")
                except ValueError:
                    print("잘못된 입력입니다. 숫자를 입력해 주세요.")
    
    # 아두이노 포트를 찾지 못했을 경우, 전체 포트 목록 제시
    elif ports:
        print("경고: 아두이노 포트를 자동으로 식별할 수 없습니다. 전체 포트 목록:")
        for i, port in enumerate(ports):
            print(f"  [{i+1}] {port.device} ({port.description})")
        
        while True:
            try:
                choice = input("연결할 포트 번호를 입력하거나 ('q'를 눌러 취소): ")
                if not choice or choice.lower() == 'q':
                    print("포트 선택을 취소합니다.")
                    return None 
                
                index = int(choice) - 1
                if 0 <= index < len(ports):
                    selected_port = ports[index].device
                    print(f"선택한 포트: {selected_port}")
                    return selected_port
                else:
                    print("잘못된 번호입니다. 다시 입력해 주세요.")
            except ValueError:
                print("잘못된 입력입니다. 숫자를 입력해 주세요.")
                
    # 사용 가능한 포트가 전혀 없는 경우
    else:
        print("오류: 시스템에서 사용 가능한 시리얼 포트를 찾을 수 없습니다.")
        return None

def connect_arduino():
    """아두이노 포트를 찾아 연결하고 serial.Serial 객체를 반환합니다."""
    arduino_port = select_arduino_port()
    
    if arduino_port is None:
        print("아두이노 연결이 취소되거나 포트를 찾지 못했습니다.")
        return None

    try:
        # serial 객체 생성 (Baud Rate: 9600, timeout: 1초)
        arduino = serial.Serial(arduino_port, 9600, timeout=1) 
        time.sleep(2) # 아두이노 리셋 시간 대기
        print(f"아두이노 연결 성공: {arduino_port}")
        return arduino
    except serial.SerialException as e:
        print(f"오류: 아두이노 포트 연결 실패. ({e})")
        print("시리얼 포트가 이미 사용 중이거나 연결 권한이 없습니다.")
        return None