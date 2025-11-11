#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// I2C default address is 0x40
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// --- Servo Configuration ---
// Adjust these values to match your specific servo's range
#define SERVOMIN 150  // This is the 'minimum' pulse length count (approx. 0 degrees)
#define SERVOMAX 600  // This is the 'maximum' pulse length count (approx. 180 degrees)
#define SERVO_FREQ 50 // Standard analog servo frequency (50 Hz)

// Helper function to map an angle (0-180) to the correct PWM pulse length
uint16_t setServoAngle(uint8_t channel, int angle) {
  // 1. 각도 범위를 0에서 180 사이로 제한합니다.
  angle = constrain(angle, 0, 180);

  // 2. map() 함수를 사용하여 각도(0-180)를 펄스 값(SERVOMIN-SERVOMAX)으로 변환합니다.
  uint16_t pulselen = map(angle, 0, 180, SERVOMIN, SERVOMAX);

  // 3. 서보 채널에 펄스 값을 설정하여 모터를 움직입니다.
  pwm.setPWM(channel, 0, pulselen);
  
  return pulselen;
}

void setup() {
  Serial.begin(9600);
  Serial.println("PCA9685 3 Channel Servo Controller Ready.");
  Serial.println("Enter angles in format: Angle1,Angle2,Angle3 (e.g., 90,45,180)");

  pwm.begin();
  
  // Set the internal oscillator frequency for accurate timing
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);  // Set the PWM frequency
  
  delay(10);
}

void loop() {
  // 시리얼 데이터가 들어올 때까지 기다립니다.
  if (Serial.available() > 0) {
    // 줄바꿈 문자('\n')를 만날 때까지 문자열을 읽어옵니다.
    String input = Serial.readStringUntil('\n');
    input.trim(); // 앞뒤 공백 제거

    // 입력 문자열에서 쉼표(,) 위치를 찾습니다.
    int firstComma = input.indexOf(',');
    // 마지막 쉼표의 위치를 찾습니다. (세 번째 숫자를 분리하기 위함)
    int secondComma = input.lastIndexOf(',');

    // 쉼표 두 개가 모두 있고, 같은 위치가 아니어야 합니다. (올바른 3개 값 형식 확인)
    if (firstComma != -1 && firstComma < secondComma) {
      // 1. 첫 번째 각도 문자열 추출 및 변환 (채널 0)
      String angle1Str = input.substring(0, firstComma);
      int angle1 = angle1Str.toInt();

      // 2. 두 번째 각도 문자열 추출 및 변환 (채널 1)
      String angle2Str = input.substring(firstComma + 1, secondComma);
      int angle2 = angle2Str.toInt();

      // 3. 세 번째 각도 문자열 추출 및 변환 (채널 2)
      String angle3Str = input.substring(secondComma + 1);
      int angle3 = angle3Str.toInt();

      // 변환된 각도를 시리얼 모니터에 출력합니다.
      Serial.print("Received: ");
      Serial.print(angle1); Serial.print(",");
      Serial.print(angle2); Serial.print(",");
      Serial.print(angle3); Serial.println("");
      
      // 서보 제어 함수 호출
      setServoAngle(0, angle1);
      setServoAngle(1, angle2);
      setServoAngle(2, angle3);

      Serial.println("Servos updated successfully.");
    } else {
      // 잘못된 형식으로 입력된 경우 안내 메시지를 출력합니다.
      Serial.print("Invalid format: '");
      Serial.print(input);
      Serial.println("'. Use: Angle1,Angle2,Angle3 (e.g., 90,45,180)");
    }
  }
  
  delay(10); // CPU 과부하 방지를 위한 짧은 지연
}