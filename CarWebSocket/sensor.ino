#include <WiFi.h>
#include <ESPmDNS.h>
#include <WebSocketsServer.h>
#include "Wire.h"
#include <MPU6050_light.h>
#include <Adafruit_AMG88xx.h>
#include <PubSubClient.h>
#include <ESP32Servo.h>
#include <TinyGPS++.h>
#include <HardwareSerial.h>

// Wi-Fi & MQTT Config
const char* ssid = "HUNTRIX007 6995";
const char* password = "9817t)X6";
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

TwoWire WireMPU = TwoWire(1);  

WiFiClient espClient;
PubSubClient client(espClient);
Adafruit_AMG88xx amg;
MPU6050 mpu(WireMPU);

HardwareSerial gpsSerial(2); // Use UART2
TinyGPSPlus gps;

unsigned long lastPingTime = 0;
const unsigned long pingInterval = 30000; // 30 seconds
unsigned long lastGpsCheckTime = 0;
const unsigned long gpsCheckInterval = 1000;
int lastStationCount = 0; 
String camIP = "";


bool autonomousMode = false;
double targetLat = 0.0;
double targetLng = 0.0;
// Target gyro orientation for alignment
double targetPitch = 0.0;
double targetRoll = 0.0;
double targetYaw = 0.0;
bool alignmentMode = false;
const double GPS_PRECISION = 0.00001; // ~1 meter precision
const double HEADING_TOLERANCE = 10.0; // degrees
const double GYRO_TOLERANCE = 3.0; // degrees tolerance for alignment (more precise)
unsigned long lastNavigationUpdate = 0;
const unsigned long navigationInterval = 500; // Update every 500ms
unsigned long lastAlignmentUpdate = 0;
const unsigned long alignmentInterval = 800; // Alignment update every 800ms (slower for stability)
int alignmentAttempts = 0;
const int maxAlignmentAttempts = 20; // Prevent infinite alignment loops


double calculateDistance(double lat1, double lng1, double lat2, double lng2) {
  const double R = 6371000; // Earth radius in meters
  double dLat = radians(lat2 - lat1);
  double dLng = radians(lng2 - lng1);
  double a = sin(dLat/2) * sin(dLat/2) + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLng/2) * sin(dLng/2);
  double c = 2 * atan2(sqrt(a), sqrt(1-a));
  return R * c;
}

double calculateBearing(double lat1, double lng1, double lat2, double lng2) {
  double dLng = radians(lng2 - lng1);
  double y = sin(dLng) * cos(radians(lat2));
  double x = cos(radians(lat1)) * sin(radians(lat2)) - sin(radians(lat1)) * cos(radians(lat2)) * cos(dLng);
  double bearing = degrees(atan2(y, x));
  return fmod((bearing + 360), 360); // Normalize to 0-360
}

bool performDeviceAlignment() {
  if (!alignmentMode) return true; // Skip alignment if not in alignment mode
  
  // Safety check: prevent infinite alignment loops
  if (alignmentAttempts >= maxAlignmentAttempts) {
    stopCar();
    alignmentMode = false;
    webSocket.broadcastTXT("ALIGNMENT_TIMEOUT");
    Serial.println("Alignment timeout - maximum attempts reached");
    alignmentAttempts = 0;
    return true;
  }
  
  // Get current gyro readings
  double currentPitch = mpu.getAngleX();
  double currentRoll = mpu.getAngleY();
  double currentYaw = mpu.getAngleZ();
  
  // Calculate alignment errors
  double pitchError = targetPitch - currentPitch;
  double rollError = targetRoll - currentRoll;
  double yawError = targetYaw - currentYaw;
  
  // Normalize yaw error to [-180, 180]
  if (yawError > 180) yawError -= 360;
  if (yawError < -180) yawError += 360;
  
  // Send alignment data for monitoring
  String alignData = "ALIGN_DATA:{";
  alignData += "\"pitchError\":" + String(pitchError, 2) + ",";
  alignData += "\"rollError\":" + String(rollError, 2) + ",";
  alignData += "\"yawError\":" + String(yawError, 2) + ",";
  alignData += "\"currentPitch\":" + String(currentPitch, 2) + ",";
  alignData += "\"currentRoll\":" + String(currentRoll, 2) + ",";
  alignData += "\"currentYaw\":" + String(currentYaw, 2) + ",";
  alignData += "\"attempts\":" + String(alignmentAttempts) + ",";
  alignData += "\"navigationPhase\":\"GYRO_ALIGNMENT\"}";
  webSocket.broadcastTXT(alignData);
  
  // Check if alignment is achieved
  if (abs(pitchError) <= GYRO_TOLERANCE && 
      abs(rollError) <= GYRO_TOLERANCE && 
      abs(yawError) <= GYRO_TOLERANCE) {
    stopCar(); // Ensure car is stopped when aligned
    alignmentMode = false;
    alignmentAttempts = 0;
    webSocket.broadcastTXT("ALIGNMENT_COMPLETE");
    Serial.println("Device alignment complete!");
    return true;
  }
  
  // Perform physical alignment corrections using car movements
  // Priority: Yaw (most critical) > Roll > Pitch
  
  if (abs(yawError) > GYRO_TOLERANCE) {
    // Yaw correction using turning movements
    int turnDuration = map(abs(yawError), GYRO_TOLERANCE, 45, 80, 300); // Scale duration based on error
    turnDuration = constrain(turnDuration, 80, 300); // Limit duration
    
    if (yawError > 0) {
      turnRight();
      Serial.println("Aligning: Turning right for yaw correction (" + String(yawError, 2) + "°) for " + String(turnDuration) + "ms");
    } else {
      turnLeft();
      Serial.println("Aligning: Turning left for yaw correction (" + String(yawError, 2) + "°) for " + String(turnDuration) + "ms");
    }
    delay(turnDuration);
    stopCar();
    
  } else if (abs(rollError) > GYRO_TOLERANCE) {
    // Roll correction using differential movement
    int rollDuration = map(abs(rollError), GYRO_TOLERANCE, 30, 100, 250); // Scale duration
    rollDuration = constrain(rollDuration, 100, 250);
    
    if (rollError > 0) {
      // Need to tilt left - move right side forward, left side backward
      moveForwardRight();
      Serial.println("Aligning: Correcting roll tilt right (" + String(rollError, 2) + "°) for " + String(rollDuration) + "ms");
    } else {
      // Need to tilt right - move left side forward, right side backward  
      moveForwardLeft();
      Serial.println("Aligning: Correcting roll tilt left (" + String(rollError, 2) + "°) for " + String(rollDuration) + "ms");
    }
    delay(rollDuration);
    stopCar();
    
  } else if (abs(pitchError) > GYRO_TOLERANCE) {
    // Pitch correction using forward/backward movement
    int pitchDuration = map(abs(pitchError), GYRO_TOLERANCE, 20, 80, 200); // Scale duration
    pitchDuration = constrain(pitchDuration, 80, 200);
    
    if (pitchError > 0) {
      // Need to pitch up - move backward briefly
      moveBackward();
      Serial.println("Aligning: Correcting pitch down (" + String(pitchError, 2) + "°) for " + String(pitchDuration) + "ms");
    } else {
      // Need to pitch down - move forward briefly
      moveForward();
      Serial.println("Aligning: Correcting pitch up (" + String(pitchError, 2) + "°) for " + String(pitchDuration) + "ms");
    }
    delay(pitchDuration);
    stopCar();
  }
  
  alignmentAttempts++;
  return false; // Alignment not complete, continue adjusting
}

void navigateToTarget() {
  if (!autonomousMode || !gps.location.isValid()) return;
  
  double currentLat = gps.location.lat();
  double currentLng = gps.location.lng();
  double distance = calculateDistance(currentLat, currentLng, targetLat, targetLng);
  
  // Check if we've reached the target location
  if (distance < 2.0) { // Within 2 meters
    stopCar();
    
    // Now perform device alignment if needed (after reaching target)
    if (alignmentMode) {
      bool alignmentComplete = performDeviceAlignment();
      if (!alignmentComplete) {
        return; // Continue alignment at target location
      }
      // Alignment complete, finish autonomous mode
      autonomousMode = false;
      alignmentMode = false;
      webSocket.broadcastTXT("TARGET_REACHED_AND_ALIGNED");
      Serial.println("Target reached and device aligned!");
    } else {
      // No alignment needed, just finish
      autonomousMode = false;
      webSocket.broadcastTXT("TARGET_REACHED");
      Serial.println("Target reached!");
    }
    return;
  }
  
  // Calculate required heading for GPS navigation
  double targetBearing = calculateBearing(currentLat, currentLng, targetLat, targetLng);
  double currentHeading = mpu.getAngleZ() + 180; // Normalize to 0-360
  if (currentHeading < 0) currentHeading += 360;
  if (currentHeading >= 360) currentHeading -= 360;
  
  double headingError = targetBearing - currentHeading;
  if (headingError > 180) headingError -= 360;
  if (headingError < -180) headingError += 360;
  
  // Send navigation data
  String navData = "NAV_DATA:{";
  navData += "\"distance\":" + String(distance, 2) + ",";
  navData += "\"targetBearing\":" + String(targetBearing, 2) + ",";
  navData += "\"currentHeading\":" + String(currentHeading, 2) + ",";
  navData += "\"headingError\":" + String(headingError, 2) + ",";
  navData += "\"alignmentMode\":" + String(alignmentMode ? "true" : "false") + ",";
  navData += "\"navigationPhase\":\"GPS_NAVIGATION\"}";
  webSocket.broadcastTXT(navData);
  
  // GPS Navigation logic - move towards target coordinates
  if (abs(headingError) > HEADING_TOLERANCE) {
    // Need to turn towards target
    if (headingError > 0) {
      turnRight();
    } else {
      turnLeft();
    }
  } else {
    // Heading is good, move forward towards target
    moveForward();
  }
}

void setGPS9600() {
  byte setBaud9600[] = {
    0xB5, 0x62, 0x06, 0x00, 0x14, 0x00, 0x01, 0x00,
    0x00, 0x00, 0xD0, 0x08, 0x00, 0x00, 0x00, 0x96,
    0x00, 0x00, 0x23, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0xA2, 0xB5
  };

  for (int i = 0; i < sizeof(setBaud9600); i++) {
    gpsSerial.write(setBaud9600[i]);
  }
}

void setup() {
  Serial.begin(115200);
  gpsSerial.begin(9600);
  setGPS9600();   // force GPS to use 9600
  
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

  // WiFi.softAP(ssid, password);
  // Serial.print("Creating Wi-Fi network");
  // while (WiFi.softAPgetStationNum() == 0) {
  //   Serial.print(".");
  //   delay(500);
  // }

  // Connect to Wi-Fi (STA mode)
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected to WiFi");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  // Start mDNS
  if (!MDNS.begin("esp32")) {
    Serial.println("Error starting mDNS");
  } else {
    Serial.println("mDNS responder started: http://esp32.local");
  }
  Serial.println("\nNetwork Created!");

  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
  Serial.println("WebSocket server started on port 81");

  Wire.begin(21, 22);     // AMG8833 on default I2C
  WireMPU.begin(26, 27);
  byte status = mpu.begin();

  if (status!=0) {
    Serial.println("❌ MPU6050 not found. Check connections.");
  }
  Serial.println(F("Calculating offsets, do not move MPU6050"));
  mpu.calcOffsets(true,true); // gyro and accelero
  Serial.println("✅ MPU6050 ready.");

  // Initialize AMG8831 sensor
  while (!amg.begin()) {
    Serial.println("Could not find a valid AMG8831 sensor, check wiring!");
    delay(1000); // Retry every 1 second
  }
  Serial.println("AMG8831 sensor initialized.");
  
  gpsSerial.begin(GPS_BAUD, SERIAL_8N1, RXD2, TXD2);
}

void loop() {
  unsigned long currentTime = millis();

  // Handle WebSocket (prioritized)
  webSocket.loop();

  // GPS non-blocking handler
  while (gpsSerial.available()) {
    char c = gpsSerial.read();
    Serial.write(c);  // Print raw NMEA sentence
    gps.encode(c);
  }
  if (gps.location.isUpdated()) {
    Serial.print("Latitude: "); 
    Serial.println(gps.location.lat(), 6);

    Serial.print("Longitude: "); 
    Serial.println(gps.location.lng(), 6);

    Serial.print("Satellites: "); 
    Serial.println(gps.satellites.value());

    Serial.print("HDOP: "); 
    Serial.println(gps.hdop.hdop());

    Serial.println("-------------------");
  } else {
    Serial.println("No GPS fix yet...");
  }

  // Send GPS updates periodically if location is updated
  mpu.update();
  if (currentTime - lastGpsCheckTime >= gpsCheckInterval) {
    sendGPSData();
    sendGyroData();
    lastGpsCheckTime = currentTime;
  }

  // Ping clients every 30s
  if (currentTime - lastPingTime >= pingInterval) {
    webSocket.broadcastTXT("ping");
    lastPingTime = currentTime;
  }

  if (autonomousMode && (currentTime - lastNavigationUpdate >= navigationInterval)) {
    navigateToTarget();
    lastNavigationUpdate = currentTime;
  } else if (alignmentMode && !autonomousMode && (currentTime - lastAlignmentUpdate >= alignmentInterval)) {
    // Perform standalone alignment when not in autonomous mode (for manual testing)
    performDeviceAlignment();
    lastAlignmentUpdate = currentTime;
  }

  // Check client connection state
  checkNewConnections();
}

void sendGPSData() {
  String gpsData = "GPS_DATA:{";
  gpsData += "\"lat\":" + String(gps.location.lat(), 6) + ",";
  gpsData += "\"lng\":" + String(gps.location.lng(), 6) + ",";
  gpsData += "\"speed\":" + String(gps.speed.kmph()) + ",";
  gpsData += "\"alt\":" + String(gps.altitude.meters()) + ",";
  gpsData += "\"hdop\":" + String(gps.hdop.value() / 100.0) + ",";
  gpsData += "\"satellites\":" + String(gps.satellites.value()) + ",";
  gpsData += "\"time\":\"" + String(gps.date.year()) + "/" + String(gps.date.month()) + "/" + String(gps.date.day()) +
             " " + String(gps.time.hour()) + ":" + String(gps.time.minute()) + ":" + String(gps.time.second()) + "\"}";
  
  webSocket.broadcastTXT(gpsData);
}

void sendGyroData() {

  String gyroData = "GYRO_DATA:{";
  gyroData += "\"gyro_x\":" + String(mpu.getGyroX()) + ",";
  gyroData += "\"gyro_y\":" + String(mpu.getGyroY()) + ",";
  gyroData += "\"gyro_z\":" + String(mpu.getGyroZ()) + ",";
  gyroData += "\"accel_x\":" + String(mpu.getAccX()) + ",";
  gyroData += "\"accel_y\":" + String(mpu.getAccY()) + ",";
  gyroData += "\"accel_z\":" + String(mpu.getAccZ()) + ",";
  gyroData += "\"accel_angle_x\":" + String(mpu.getAccAngleX()) + ",";
  gyroData += "\"accel_angle_y\":" + String(mpu.getAccAngleY()) + ",";
  gyroData += "\"angle_x\":" + String(mpu.getAngleX()) + ",";
  gyroData += "\"angle_y\":" + String(mpu.getAngleY()) + ",";
  gyroData += "\"angle_z\":" + String(mpu.getAngleZ()) + ",";
  gyroData += "\"temp\":" + String(mpu.getTemp(), 2) + "}";

  webSocket.broadcastTXT(gyroData);
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
      Serial.println(pos); 
      Serial.println(position); 
      if (position >= 0 && position <= 180) {
        h_servo.write(position);
      } else {
        Serial.println("Invalid horizontal servo position: " + String(position));
      }
    }
    else if (command.startsWith("V_TURN_CAM:")) {
      String pos = command.substring(11);
      int position = pos.toInt();
      Serial.println(pos); 
      Serial.println(position); 
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
    } else if (command.startsWith("SET_TARGET:")) {
      String coords = command.substring(11);
      
      // Parse coordinates and gyro data: lat,lng,pitch,roll,yaw
      int firstComma = coords.indexOf(',');
      int secondComma = coords.indexOf(',', firstComma + 1);
      int thirdComma = coords.indexOf(',', secondComma + 1);
      int fourthComma = coords.indexOf(',', thirdComma + 1);
      
      if (firstComma > 0 && secondComma > 0) {
        targetLat = coords.substring(0, firstComma).toDouble();
        targetLng = coords.substring(firstComma + 1, secondComma).toDouble();
        
        // Parse gyro data if present
        if (thirdComma > 0 && fourthComma > 0) {
          targetPitch = coords.substring(secondComma + 1, thirdComma).toDouble();
          targetRoll = coords.substring(thirdComma + 1, fourthComma).toDouble();
          targetYaw = coords.substring(fourthComma + 1).toDouble();
          alignmentMode = true; // Enable alignment mode when gyro data is provided
          alignmentAttempts = 0; // Reset alignment attempts
          
          Serial.println("Target set with alignment: " + String(targetLat, 6) + ", " + String(targetLng, 6));
          Serial.println("Target orientation - Pitch: " + String(targetPitch, 3) + ", Roll: " + String(targetRoll, 3) + ", Yaw: " + String(targetYaw, 3));
          
          webSocket.broadcastTXT("TARGET_SET_WITH_ALIGNMENT:" + String(targetLat, 6) + "," + String(targetLng, 6) + 
                                "," + String(targetPitch, 3) + "," + String(targetRoll, 3) + "," + String(targetYaw, 3));
        } else {
          alignmentMode = false; // No alignment if gyro data not provided
          alignmentAttempts = 0; // Reset counter
          Serial.println("Target set: " + String(targetLat, 6) + ", " + String(targetLng, 6));
          webSocket.broadcastTXT("TARGET_SET:" + String(targetLat, 6) + "," + String(targetLng, 6));
        }
        
        autonomousMode = true;
      }
    }
    else if (command == "STOP_AUTO") {
      autonomousMode = false;
      alignmentMode = false;
      alignmentAttempts = 0; // Reset counter
      stopCar();
      webSocket.broadcastTXT("AUTO_STOPPED");
      Serial.println("Autonomous mode and alignment stopped");
    }
    else if (command == "GET_TARGET") {
      if (alignmentMode) {
        webSocket.sendTXT(num, "CURRENT_TARGET_WITH_ALIGNMENT:" + String(targetLat, 6) + "," + String(targetLng, 6) + 
                          "," + String(targetPitch, 3) + "," + String(targetRoll, 3) + "," + String(targetYaw, 3));
      } else {
        webSocket.sendTXT(num, "CURRENT_TARGET:" + String(targetLat, 6) + "," + String(targetLng, 6));
      }
    }
    else if (command.startsWith("SET_ALIGNMENT:")) {
      String gyroValues = command.substring(14);
      int firstComma = gyroValues.indexOf(',');
      int secondComma = gyroValues.indexOf(',', firstComma + 1);
      
      if (firstComma > 0 && secondComma > 0) {
        targetPitch = gyroValues.substring(0, firstComma).toDouble();
        targetRoll = gyroValues.substring(firstComma + 1, secondComma).toDouble();
        targetYaw = gyroValues.substring(secondComma + 1).toDouble();
        alignmentMode = true;
        alignmentAttempts = 0; // Reset alignment attempts
        
        Serial.println("Alignment mode activated - Target orientation: Pitch: " + String(targetPitch, 3) + 
                      ", Roll: " + String(targetRoll, 3) + ", Yaw: " + String(targetYaw, 3));
        webSocket.broadcastTXT("ALIGNMENT_MODE_ACTIVATED:" + String(targetPitch, 3) + "," + 
                              String(targetRoll, 3) + "," + String(targetYaw, 3));
      }
    }
    else if (command == "STOP_ALIGNMENT") {
      alignmentMode = false;
      alignmentAttempts = 0; // Reset counter
      stopCar();
      webSocket.broadcastTXT("ALIGNMENT_STOPPED");
      Serial.println("Alignment mode stopped");
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
