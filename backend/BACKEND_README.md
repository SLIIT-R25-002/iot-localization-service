# SuperGlue Backend Implementation

This repository contains a Python backend implementation for SuperGlue feature matching that can handle:
1. **Image-to-Image matching**: Match features between two static images
2. **Image-to-Video Stream matching**: Match a reference image against frames from a video stream (IP camera, webcam, or video file)

## Features

- **Flexible Input Sources**: Support for webcams, IP cameras, video files, and static images
- **Real-time Processing**: Process video streams frame by frame
- **REST API**: Full Flask-based REST API for web integration
- **Standalone Mode**: Direct Python usage without web dependencies
- **Visualization**: Generate matching visualizations with confidence scores
- **Configurable**: Adjustable SuperPoint and SuperGlue parameters

## Installation

1. **Install the original SuperGlue dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install additional backend dependencies:**
   ```bash
   pip install -r requirements_backend.txt
   ```

3. **Ensure you have the SuperGlue model weights:**
   - The weights should be in `models/weights/` directory
   - Both indoor and outdoor models are supported

## Usage

### Option 1: Standalone Python Implementation

The standalone implementation (`superglue_standalone.py`) provides direct Python access without web dependencies.

#### Basic Image Matching Example:

```python
from superglue_standalone import SuperGlueStandalone

# Initialize
sg = SuperGlueStandalone()

# Set reference image
result = sg.set_reference_image("path/to/reference.jpg")
print(f"Reference set: {result}")

# Match with query image
result = sg.match_with_image("path/to/query.jpg", save_visualization=True)
print(f"Found {result['num_matches']} matches")
```

#### Video Stream Matching Example:

```python
# Process video stream
results = sg.process_video_batch(
    stream_source="0",  # Webcam, or "path/to/video.mp4", or "http://ip:port/stream"
    max_frames=50,
    save_visualizations=True,
    output_dir="output"
)

# Print summary
successful = [r for r in results if r['status'] == 'success']
print(f"Processed {len(successful)} frames")
```

#### Command Line Usage:

```bash
# Image matching
python superglue_standalone.py image --ref_image ref.jpg --query_image query.jpg

# Video stream matching (webcam)
python superglue_standalone.py video --ref_image ref.jpg --stream_source 0

# Video file matching
python superglue_standalone.py video --ref_image ref.jpg --stream_source video.mp4

# IP camera matching
python superglue_standalone.py video --ref_image ref.jpg --stream_source "http://192.168.1.100:8080/video"
```

### Option 2: Flask REST API Backend

The REST API implementation (`superglue_backend.py`) provides a web service for integration with other applications.

#### Start the Backend Server:

```bash
python superglue_backend.py --host localhost --port 5000 --superglue indoor
```

#### API Endpoints:

- **GET** `/api/info` - Get backend status and configuration
- **POST** `/api/set_reference` - Set reference image
- **POST** `/api/match_image` - Match with another image
- **POST** `/api/start_stream` - Start video stream matching
- **GET** `/api/next_frame` - Get next frame matching results
- **POST** `/api/stop_stream` - Stop video stream

#### API Usage Example:

```python
import requests

# Set reference image
response = requests.post('http://localhost:5000/api/set_reference', 
                        json={'image_path': 'ref.jpg'})
print(response.json())

# Match with image
response = requests.post('http://localhost:5000/api/match_image',
                        json={'query_image_path': 'query.jpg', 'visualize': True})
result = response.json()
print(f"Found {result['num_matches']} matches")
```

#### Client Example:

Use the provided `client_example.py` for a complete client implementation:

```bash
python client_example.py --ref_image ref.jpg --query_image query.jpg --mode image
python client_example.py --ref_image ref.jpg --stream_source 0 --mode video
```

## Input Sources

### Image Inputs
- **Local files**: JPG, PNG, JPEG formats
- **Paths**: Absolute or relative paths to image files

### Video Stream Inputs
- **Webcam**: Use integer ID (0, 1, 2, ...)
- **IP Camera**: HTTP/RTSP URLs (e.g., "http://192.168.1.100:8080/video")
- **Video Files**: Local video files (.mp4, .avi, etc.)
- **Image Directories**: Folder containing sequential images

### Examples:
```python
# Different stream sources
sg.start_video_stream_matching("0")  # Default webcam
sg.start_video_stream_matching("1")  # Second webcam
sg.start_video_stream_matching("http://192.168.1.100:8080/video")  # IP camera
sg.start_video_stream_matching("rtsp://user:pass@192.168.1.100/stream")  # RTSP
sg.start_video_stream_matching("path/to/video.mp4")  # Video file
sg.start_video_stream_matching("path/to/image/directory/")  # Image sequence
```

## Configuration

### SuperGlue Model Settings:

```python
config = {
    'superpoint': {
        'nms_radius': 4,                    # Non-maximum suppression radius
        'keypoint_threshold': 0.005,        # Keypoint confidence threshold
        'max_keypoints': 1024               # Maximum keypoints to detect
    },
    'superglue': {
        'weights': 'indoor',                # 'indoor' or 'outdoor'
        'sinkhorn_iterations': 20,          # Sinkhorn algorithm iterations
        'match_threshold': 0.2,             # Match confidence threshold
    },
    'force_cpu': False                      # Force CPU usage
}
```

### Model Selection:
- **Indoor model**: Best for indoor scenes, close-range matching
- **Outdoor model**: Best for outdoor scenes, wide-baseline matching

## Output Format

### Matching Results:
```json
{
    "status": "success",
    "num_keypoints_ref": 512,
    "num_keypoints_query": 487,
    "num_matches": 156,
    "matches": {
        "keypoints_ref": [[x1, y1], [x2, y2], ...],
        "keypoints_query": [[x1, y1], [x2, y2], ...],
        "confidence": [0.95, 0.87, ...]
    },
    "timing": {
        "forward_pass": 0.045
    }
}
```

### Video Stream Results:
```json
{
    "status": "success",
    "frame_id": 42,
    "num_matches": 89,
    "visualization_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
    "timing": {"forward_pass": 0.038}
}
```

## Performance Tips

1. **GPU Usage**: Ensure CUDA is available for faster processing
2. **Image Sizing**: Use appropriate resize parameters (default: 640x480)
3. **Model Selection**: Choose indoor/outdoor model based on your use case
4. **Batch Processing**: Use `process_video_batch()` for offline video processing
5. **Memory Management**: Stop video streams when done to free resources

## Troubleshooting

### Common Issues:

1. **"Could not read image"**: Check file paths and image formats
2. **"No reference image set"**: Call `set_reference_image()` first
3. **GPU out of memory**: Reduce image size or use `--force_cpu`
4. **Video stream fails**: Check camera permissions and stream URLs
5. **Import errors**: Install all requirements from `requirements_backend.txt`

### Debug Mode:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration Examples

### With OpenCV:
```python
import cv2
from superglue_standalone import SuperGlueStandalone

sg = SuperGlueStandalone()
sg.set_reference_image("ref.jpg")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save frame temporarily
    cv2.imwrite("temp_frame.jpg", frame)
    
    # Match with reference
    result = sg.match_with_image("temp_frame.jpg")
    print(f"Matches: {result.get('num_matches', 0)}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

### With Flask Web App:
```python
from flask import Flask, render_template, request
import requests

app = Flask(__name__)
BACKEND_URL = "http://localhost:5000"

@app.route('/match', methods=['POST'])
def match_images():
    ref_path = request.form['reference']
    query_path = request.form['query']
    
    # Set reference
    requests.post(f"{BACKEND_URL}/api/set_reference", 
                 json={'image_path': ref_path})
    
    # Match
    response = requests.post(f"{BACKEND_URL}/api/match_image",
                           json={'query_image_path': query_path})
    
    return response.json()
```

## License

This implementation is based on SuperGlue by Magic Leap (CVPR 2020). Please refer to the original license terms in the main repository.

## Citation

If you use this backend implementation, please cite the original SuperGlue paper:

```bibtex
@inproceedings{sarlin20superglue,
  author    = {Paul-Edouard Sarlin and
               Daniel DeTone and
               Tomasz Malisiewicz and
               Andrew Rabinovich},
  title     = {{SuperGlue}: Learning Feature Matching with Graph Neural Networks},
  booktitle = {CVPR},
  year      = {2020},
  url       = {https://arxiv.org/abs/1911.11763}
}
```
