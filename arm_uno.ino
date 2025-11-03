#include <Servo.h>

Servo myServo;
int angle = 0;

void setup() {
  Serial.begin(9600);
  myServo.attach(10); // Servo signal pin to D9
  myServo.write(0);
}

void loop() {
  if (Serial.available() > 0) {
    angle = Serial.parseInt();  // Read angle from Python
    Serial.print("Serial.parseInt : ");
    Serial.println(angle);
    if (angle >= 0 && angle <= 180) {
      myServo.write(angle);
    }
  }
}