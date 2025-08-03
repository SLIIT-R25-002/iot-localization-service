# IoT Localization Service - HeatScape

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/Platform-ESP32-blue.svg)](https://espressif.com/en/products/socs/esp32)
[![Research Project](https://img.shields.io/badge/Research%20Project-SLIIT%20Year%204-green.svg)](https://sliit.lk)

A comprehensive IoT localization and remote monitoring system developed as part of the HeatScape research project at SLIIT. This system combines autonomous navigation, thermal imaging, and real-time streaming capabilities using ESP32-based devices.

## 🚗 Overview

This project implements a dual ESP32 system for autonomous vehicle navigation and thermal monitoring:

- **ESP32-CAM**: Provides real-time video streaming with WebSocket communication
- **ESP32 DevKit**: Controls vehicle movement, sensors, and autonomous navigation

### Key Features

- 🎥 **Real-time Video Streaming** via ESP32-CAM
- 🌡️ **Thermal Imaging** using AMG88xx sensor (8x8 thermal array)
- 📍 **GPS-based Navigation** with autonomous pathfinding
- 🧭 **IMU Sensor Integration** (MPU6050) for orientation tracking
- 🎮 **Remote Control** via WebSocket interface
- 🚗 **Motor Control** with differential steering
- 📷 **Pan-Tilt Camera Control** using servo motors
- 🔄 **Real-time Data Broadcasting** (GPS, gyroscope, temperature)

## 🏗️ System Architecture

```
┌─────────────────┐    WebSocket    ┌─────────────────┐
│   ESP32-CAM     │◄──────────────►│  ESP32 DevKit   │
│  (Camera Server)│                │ (Control Unit)  │
└─────────────────┘                └─────────────────┘
         │                                   │
         │                                   │
         ▼                                   ▼
┌─────────────────┐                ┌─────────────────┐
│ Video Streaming │                │   Sensor Data   │
│   HTTP Server   │                │   WebSocket     │
└─────────────────┘                └─────────────────┘
         │                                   │
         │                                   ▼
         │                         ┌─────────────────┐
         │                         │   Client App    │
         └─────────────────────────┤  (Web/Mobile)   │
                                   └─────────────────┘
```

## 📋 Hardware Requirements

### ESP32-CAM Module
- ESP32-CAM development board
- OV2640 camera module
- MicroSD card (optional)
- USB-to-Serial programmer

### ESP32 DevKit Module
- ESP32 DevKit v1
- MPU6050 (6-axis IMU)
- AMG88xx thermal sensor
- NEO-6M GPS module
- L298N motor driver
- 2x servo motors (SG90)
- 2x DC gear motors
- Jumper wires and breadboard

### Power Supply
- 7.4V Li-Po battery (recommended)
- Power distribution board
- Voltage regulators (3.3V, 5V)

## 📦 Required Libraries

Install these libraries through Arduino IDE Library Manager:

```cpp
// Core Libraries
#include <WiFi.h>
#include <WebSocketsServer.h>
#include <WebSocketsClient.h>
#include <ESPmDNS.h>

// Sensor Libraries
#include <MPU6050_light.h>
#include <Adafruit_AMG88xx.h>
#include <TinyGPS++.h>

// Actuator Libraries
#include <ESP32Servo.h>

// Camera Libraries (ESP32-CAM)
#include "esp_camera.h"
#include "esp_http_server.h"
```

## 🔧 Installation & Setup

### 1. Hardware Assembly

**ESP32-CAM Connections:**
- Camera module → ESP32-CAM (pre-connected)
- External antenna (if using external WiFi antenna)

**ESP32 DevKit Connections:**
```
MPU6050:    VCC→3.3V, GND→GND, SDA→26, SCL→27
AMG88xx:    VCC→3.3V, GND→GND, SDA→21, SCL→22
GPS Module: VCC→3.3V, GND→GND, TX→16, RX→17
L298N:      IN1→19, IN2→18, IN3→23, IN4→4
Servos:     Horizontal→15, Vertical→2
```

### 2. Software Configuration

1. **Clone the repository:**
```bash
git clone https://github.com/SLIIT-R25-002/iot-localization-service.git
cd iot-localization-service
```

2. **Configure WiFi credentials:**

Edit both `CameraWebServer.ino` and `sensor.ino`:
```cpp
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
```

3. **Upload firmware:**
- Upload `CameraWebServer.ino` to ESP32-CAM
- Upload `sensor.ino` to ESP32 DevKit

### 3. Board Configuration

**For ESP32-CAM:**
- Board: ESP32 Wrover Module
- Partition Scheme: Custom (use provided `partitions.csv`)
- PSRAM: Enabled

**For ESP32 DevKit:**
- Board: ESP32 Dev Module
- Flash Size: 4MB
- Partition Scheme: Default

## 🚀 Usage

### Basic Operation

1. **Power on both ESP32 modules**
2. **Connect to WiFi network** (check Serial Monitor for IP addresses)
3. **Access camera stream:** `http://<ESP32-CAM-IP>`
4. **Connect to WebSocket:** `ws://<ESP32-DevKit-IP>:81`

### WebSocket Commands

#### Movement Controls
```javascript
webSocket.send("forward");      // Move forward
webSocket.send("backward");     // Move backward
webSocket.send("left");         // Turn left
webSocket.send("right");        // Turn right
webSocket.send("stop");         // Stop movement
```

#### Camera Controls
```javascript
webSocket.send("H_TURN_CAM:90");  // Horizontal servo (0-180°)
webSocket.send("V_TURN_CAM:45");  // Vertical servo (0-180°)
```

#### Autonomous Navigation
```javascript
webSocket.send("SET_TARGET:6.9271,79.8612");  // Set GPS coordinates
webSocket.send("STOP_AUTO");                  // Stop autonomous mode
```

#### Sensor Data
```javascript
webSocket.send("get_temp");    // Request thermal data
// GPS and IMU data sent automatically every second
```

### Data Formats

**GPS Data:**
```json
{
  "lat": 6.927079,
  "lng": 79.861244,
  "speed": 0.0,
  "alt": 12.3,
  "hdop": 1.2,
  "satellites": 8,
  "time": "2025/08/03 14:30:25"
}
```

**Thermal Data:**
```json
[25.2, 25.1, 24.9, ...] // 64 temperature values (8x8 grid)
```

## 🎯 Features in Detail

### Autonomous Navigation
- **GPS-based waypoint navigation**
- **Automatic heading correction using IMU**
- **Distance calculation with 1-meter precision**
- **Obstacle avoidance (future enhancement)**

### Thermal Monitoring
- **8x8 thermal sensor array (AMG88xx)**
- **Real-time temperature mapping**
- **Heat signature detection**
- **Data logging capabilities**

### Camera System
- **Live video streaming**
- **Pan-tilt camera control**
- **Multiple resolution support**
- **Low-latency WebSocket communication**

## 🛠️ Configuration Options

### Camera Settings (ESP32-CAM)
```cpp
config.frame_size = FRAMESIZE_UXGA;     // Image resolution
config.jpeg_quality = 10;               // JPEG quality (1-63)
config.fb_count = 2;                    // Frame buffers
```

### Navigation Parameters
```cpp
const double GPS_PRECISION = 0.00001;   // ~1 meter precision
const double HEADING_TOLERANCE = 10.0;  // Heading accuracy (degrees)
const unsigned long navigationInterval = 500; // Update rate (ms)
```

## 🔍 Troubleshooting

### Common Issues

**Camera not connecting:**
- Check power supply (minimum 3.3V, 600mA)
- Verify camera module connection
- Use external antenna for better WiFi range

**GPS not acquiring signal:**
- Ensure outdoor testing or near window
- Check baud rate configuration (9600)
- Verify wiring connections

**Motor not responding:**
- Check L298N power supply (7-12V)
- Verify motor driver connections
- Test individual motor channels

**WebSocket connection issues:**
- Confirm both devices on same network
- Check firewall settings
- Verify port 81 is available

### Debug Mode

Enable detailed logging:
```cpp
Serial.setDebugOutput(true);  // ESP32-CAM
// Check Serial Monitor at 115200 baud
```

## 📊 Performance Metrics

- **Video latency:** <200ms over local network
- **GPS accuracy:** ±1 meter (with good signal)
- **Thermal resolution:** 8x8 pixels, ±2°C accuracy
- **Battery life:** 2-3 hours (7.4V 2200mAh Li-Po)
- **WiFi range:** Up to 100m (outdoor, direct line)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Academic Context

This project is part of the HeatScape research initiative at SLIIT (Sri Lanka Institute of Information Technology), focusing on:

- **IoT-based environmental monitoring**
- **Autonomous vehicle navigation systems**
- **Thermal imaging applications**
- **Real-time data acquisition and processing**

### Research Team
- **Institution:** Sri Lanka Institute of Information Technology (SLIIT)
- **Year:** 4th Year Research Project
- **Project Code:** R25-002

## 🆘 Support

For technical support and questions:
- Create an issue in this repository
- Contact the development team
- Refer to ESP32 documentation for hardware-specific issues

## 🔮 Future Enhancements

- [ ] Machine learning integration for thermal analysis
- [ ] Advanced obstacle avoidance using computer vision
- [ ] Mobile application development
- [ ] Cloud data logging and analytics
- [ ] Multi-robot coordination
- [ ] Edge AI processing for real-time decision making

---

**Note:** This is a research project. Use appropriate safety measures when testing autonomous navigation features.