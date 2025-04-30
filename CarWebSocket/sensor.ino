#include <WiFi.h>
#include <WebSocketsServer.h>
#include <Wire.h>
#include <Adafruit_AMG88xx.h>
#include <PubSubClient.h>
#include <ESP32Servo.h>
#include <TinyGPS++.h>
#include <HardwareSerial.h>

// Wi-Fi & MQTT Config
const char* ssid = "ESP32-Thermal";
const char* password = "12345678";
const char* temperature_topic = "car/temperature";

// WebSockets Server
WebSocketsServer webSocket = WebSocketsServer(81);

// Motor Pins (L298N)
#define MOTOR1_IN1 19
#define MOTOR1_IN2 18
#define MOTOR2_IN1 23
#define MOTOR2_IN2 4

// Serbo Pins
#define H_SERVO_PIN 15
#define V_SERVO_PIN 2

// GPS Pins
#define RXD2 16
#define TXD2 17

#define GPS_BAUD 9600

Servo h_servo;
Servo v_servo;

WiFiClient espClient;
PubSubClient client(espClient);
Adafruit_AMG88xx amg;

HardwareSerial gpsSerial(2); // Use UART2
TinyGPSPlus gps;

unsigned long lastPingTime = 0;
const unsigned long pingInterval = 30000; // 30 seconds
unsigned long lastGpsCheckTime = 0;
const unsigned long gpsCheckInterval = 200;
int lastStationCount = 0; 
String camIP = "";

void setup() {
  Serial.begin(115200);
  
  pinMode(MOTOR1_IN1, OUTPUT);
  pinMode(MOTOR1_IN2, OUTPUT);
  pinMode(MOTOR2_IN1, OUTPUT);
  pinMode(MOTOR2_IN2, OUTPUT);

  // Allocate only two timers for the servos
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);

  h_servo.setPeriodHertz(50);
  h_servo.attach(H_SERVO_PIN, 500, 2500);
  v_servo.setPeriodHertz(50);
  v_servo.attach(V_SERVO_PIN, 500, 2500);

  h_servo.write(90);
  v_servo.write(90);
  
  stopCar();

  WiFi.softAP(ssid, password);
  Serial.print("Creating Wi-Fi network");
  while (WiFi.softAPgetStationNum() == 0) {
    Serial.print(".");
    delay(500);
  }
  Serial.println("\nNetwork Created!");

  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
  Serial.println("WebSocket server started on port 81");

  // Initialize AMG8831 sensor
  while (!amg.begin()) {
    Serial.println("Could not find a valid AMG8831 sensor, check wiring!");
    delay(1000); // Retry every 1 second
  }
  Serial.println("AMG8831 sensor initialized.");
  
  gpsSerial.begin(GPS_BAUD, SERIAL_8N1, RXD2, TXD2);
}

void loop() {
  webSocket.loop();

  unsigned long currentTime = millis();
  if (currentTime - lastPingTime > pingInterval) {
    webSocket.broadcastTXT("ping");
    lastPingTime = currentTime;
  }

  checkNewConnections();

  // Read and print temperature data periodically
  // static unsigned long lastTempReadTime = 0;
  // const unsigned long tempReadInterval = 5000; // 5 seconds
  // if (millis() - lastTempReadTime > tempReadInterval) {
  //   readTemperatureData();
  //   lastTempReadTime = millis();
  // }

   // GPS non-blocking handler
  while (gpsSerial.available()) {
    gps.encode(gpsSerial.read());
  }

  if (currentTime - lastGpsCheckTime > gpsCheckInterval) {
    if (gps.location.isUpdated()) {
      Serial.print("LAT: ");
      Serial.println(gps.location.lat(), 6);
      Serial.print("LONG: "); 
      Serial.println(gps.location.lng(), 6);
      Serial.print("SPEED (km/h) = "); 
      Serial.println(gps.speed.kmph()); 
      Serial.print("ALT (min)= "); 
      Serial.println(gps.altitude.meters());
      Serial.print("HDOP = "); 
      Serial.println(gps.hdop.value() / 100.0); 
      Serial.print("Satellites = "); 
      Serial.println(gps.satellites.value()); 
      Serial.print("Time in UTC: ");
      Serial.println(String(gps.date.year()) + "/" + String(gps.date.month()) + "/" + String(gps.date.day()) + "," + String(gps.time.hour()) + ":" + String(gps.time.minute()) + ":" + String(gps.time.second()));
      Serial.println("");

      String gpsData = "GPS_DATA:{";
      gpsData += "\"lat\":" + String(gps.location.lat(), 6) + ",";
      gpsData += "\"lng\":" + String(gps.location.lng(), 6) + ",";
      gpsData += "\"speed\":" + String(gps.speed.kmph()) + ",";
      gpsData += "\"alt\":" + String(gps.altitude.meters()) + ",";
      gpsData += "\"hdop\":" + String(gps.hdop.value() / 100.0) + ",";
      gpsData += "\"satellites\":" + String(gps.satellites.value()) + ",";
      gpsData += "\"time\":\"" + String(gps.date.year()) + "/" + String(gps.date.month()) + "/" + String(gps.date.day()) + " " + String(gps.time.hour()) + ":" + String(gps.time.minute()) + ":" + String(gps.time.second()) + "\"}";
      webSocket.broadcastTXT(gpsData);
    }

    lastGpsCheckTime = currentTime;
  }
}

void checkNewConnections() {
  int currentStationCount = WiFi.softAPgetStationNum(); // Get the current number of connected devices

  if (currentStationCount > lastStationCount) {
    // A new device has connected
    Serial.println("New device connected!");
    webSocket.broadcastTXT("New device connected!");
    lastStationCount = currentStationCount; // Update the last station count
  } else if (currentStationCount < lastStationCount) {
    // A device has disconnected
    Serial.println("A device disconnected.");
    webSocket.broadcastTXT("A device disconnected.");
    lastStationCount = currentStationCount;
  }
}

void webSocketEvent(uint8_t num, WStype_t type, uint8_t* payload, size_t length) {
  if (type == WStype_TEXT) {
    String command = "";
    for (size_t i = 0; i < length; i++) {
      command += (char)payload[i];
    }
    Serial.println("Received Command: " + command);

    if (command == "forward") moveForward();
    else if (command == "backward") moveBackward();
    else if (command == "left") turnLeft();
    else if (command == "right") turnRight();
    else if (command == "stop") stopCar();
    else if (command == "forward_right") moveForwardRight();
    else if (command == "forward_left") moveForwardLeft();
    else if (command == "backward_right") moveBackwardRight();
    else if (command == "backward_left") moveBackwardLeft();
    else if (command == "get_temp") readTemperatureData();
    else if (command.startsWith("H_TURN_CAM:")) {
      String pos = command.substring(11);
      int position = pos.toInt();
      if (position >= 0 && position <= 180) {
        h_servo.write(position);
      } else {
        Serial.println("Invalid horizontal servo position: " + String(position));
      }
    }
    else if (command.startsWith("V_TURN_CAM:")) {
      String pos = command.substring(11);
      int position = pos.toInt();
      if (position >= 0 && position <= 180) {
        v_servo.write(position);
      } else {
        Serial.println("Invalid vertical servo position: " + String(position));
      }
    }
    else if (command.startsWith("CAM_IP:")) {
      camIP = command.substring(7);
      webSocket.broadcastTXT("CAM_IP:" + camIP);
      Serial.println("ESP32-CAM IP stored: " + camIP);
    } else if (command == "GET_CAM_IP") {
      webSocket.sendTXT(num, "CAM_IP:" + camIP);
    }
  }
}

void moveForward() {
  digitalWrite(MOTOR1_IN1, HIGH);
  digitalWrite(MOTOR1_IN2, LOW);
  digitalWrite(MOTOR2_IN1, HIGH);
  digitalWrite(MOTOR2_IN2, LOW);
  Serial.println("Moving Forward");
}

void moveBackward() {
  digitalWrite(MOTOR1_IN1, LOW);
  digitalWrite(MOTOR1_IN2, HIGH);
  digitalWrite(MOTOR2_IN1, LOW);
  digitalWrite(MOTOR2_IN2, HIGH);
  Serial.println("Moving Backward");
}

void turnLeft() {
  digitalWrite(MOTOR1_IN1, LOW);
  digitalWrite(MOTOR1_IN2, HIGH);
  digitalWrite(MOTOR2_IN1, HIGH);
  digitalWrite(MOTOR2_IN2, LOW);
  Serial.println("Turning Left");
}

void turnRight() {
  digitalWrite(MOTOR1_IN1, HIGH);
  digitalWrite(MOTOR1_IN2, LOW);
  digitalWrite(MOTOR2_IN1, LOW);
  digitalWrite(MOTOR2_IN2, HIGH);
  Serial.println("Turning Right");
}

void moveForwardRight() {
  digitalWrite(MOTOR1_IN1, HIGH);
  digitalWrite(MOTOR1_IN2, LOW);
  digitalWrite(MOTOR2_IN1, LOW);
  digitalWrite(MOTOR2_IN2, LOW);
  Serial.println("Moving Forward Right");
}

void moveForwardLeft() {
  digitalWrite(MOTOR1_IN1, LOW);
  digitalWrite(MOTOR1_IN2, LOW);
  digitalWrite(MOTOR2_IN1, HIGH);
  digitalWrite(MOTOR2_IN2, LOW);
  Serial.println("Moving Forward Left");
}

void moveBackwardRight() {
  digitalWrite(MOTOR1_IN1, LOW);
  digitalWrite(MOTOR1_IN2, HIGH);
  digitalWrite(MOTOR2_IN1, LOW);
  digitalWrite(MOTOR2_IN2, LOW);
  Serial.println("Moving Backward Right");
}

void moveBackwardLeft() {
  digitalWrite(MOTOR1_IN1, LOW);
  digitalWrite(MOTOR1_IN2, LOW);
  digitalWrite(MOTOR2_IN1, LOW);
  digitalWrite(MOTOR2_IN2, HIGH);
  Serial.println("Moving Backward Left");
}

void stopCar() {
  digitalWrite(MOTOR1_IN1, LOW);
  digitalWrite(MOTOR1_IN2, LOW);
  digitalWrite(MOTOR2_IN1, LOW);
  digitalWrite(MOTOR2_IN2, LOW);
  Serial.println("Stopping");
}

// Function to read temperature data from AMG8831
void readTemperatureData() {
  float pixels[64]; // AMG8831 has an 8x8 grid of temperature readings
  amg.readPixels(pixels);

  Serial.println("Temperature Data:");
  String temperatureData = "["; // Start JSON array
  for (int i = 0; i < 64; i++) {
    Serial.print(pixels[i]);
    Serial.print((i % 8 == 7) ? "\n" : ", "); // Print in an 8x8 grid format

    // Append temperature data to JSON array
    temperatureData += String(pixels[i]);
    if (i < 63) temperatureData += ",";
  }
  temperatureData += "]"; // End JSON array

  // Broadcast temperature data via WebSocket
  webSocket.broadcastTXT("TEMP_DATA:" + temperatureData);
  Serial.println("Temperature data broadcasted to WebSocket clients.");
}
