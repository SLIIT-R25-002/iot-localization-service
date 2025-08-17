#!/usr/bin/env python3
"""
SuperGlue Backend API
A Python backend that uses SuperGlue for feature matching between:
1. Two static images
2. A reference image and frames from a video stream (IP camera or video file)

Author: Backend Implementation
Based on: SuperGlue by Magic Leap (CVPR 2020)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import threading
import queue

import cv2
import numpy as np
import torch
import matplotlib.cm as cm
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import base64
import io
from PIL import Image

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer, make_matching_plot_fast, 
                         frame2tensor, read_image, process_resize)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable gradients for inference
torch.set_grad_enabled(False)


class SuperGlueBackend:
    """SuperGlue backend for image and video stream matching"""
    
    def __init__(self, config: Dict):
        """
        Initialize the SuperGlue backend
        
        Args:
            config: Configuration dictionary for SuperPoint and SuperGlue
        """
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() and not config.get('force_cpu', False) else 'cpu'
        logger.info(f'Running inference on device: {self.device}')
        
        logger.info(f'CUDA version: {torch.version.cuda}')      # should match your CUDA toolkit (e.g. 12.1)
        logger.info(f'CUDA available: {torch.cuda.is_available()}')
        
        if torch.cuda.is_available():
            logger.info(f'GPU device name: {torch.cuda.get_device_name(0)}')
        else:
            logger.info('No CUDA GPU available, running on CPU') 
        
        # Initialize the matching network
        self.matching = Matching(config).eval().to(self.device)
        self.keys = ['keypoints', 'scores', 'descriptors']
        
        # Store reference image data
        self.reference_data = None
        self.reference_image = None
        self.reference_path = None
        
        # Video streaming
        self.video_streamer = None
        self.streaming_active = False
        
        # Video recording
        self.video_writer = None
        self.recording_active = False
        self.recording_thread = None
        self.frame_queue = queue.Queue()
        
        # Timer for performance monitoring
        self.timer = AverageTimer()
        
    def set_reference_image(self, image_path: str, resize: List[int] = [640, 480]) -> Dict:
        """
        Set the reference image for matching from file path
        
        Args:
            image_path: Path to the reference image
            resize: Resize dimensions [width, height]
            
        Returns:
            Dictionary with status and image info
        """
        try:
            # Read and process the reference image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return {"status": "error", "message": f"Could not read image: {image_path}"}
            
            return self._process_reference_image(image, resize, image_path)
            
        except Exception as e:
            logger.error(f"Error setting reference image: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def set_reference_image_from_data(self, image_data: Union[np.ndarray, bytes], 
                                     resize: List[int] = [640, 480], 
                                     filename: str = "uploaded_image") -> Dict:
        """
        Set the reference image for matching from image data
        
        Args:
            image_data: Image data as numpy array or bytes
            resize: Resize dimensions [width, height]
            filename: Name for the uploaded file (for reference)
            
        Returns:
            Dictionary with status and image info
        """
        try:
            # Handle different input formats
            if isinstance(image_data, bytes):
                # Convert bytes to numpy array
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    return {"status": "error", "message": "Could not decode image data"}
            elif isinstance(image_data, np.ndarray):
                # If already a numpy array, convert to grayscale if needed
                if len(image_data.shape) == 3:
                    image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
                else:
                    image = image_data
            else:
                return {"status": "error", "message": "Unsupported image data format"}
            
            return self._process_reference_image(image, resize, filename)
            
        except Exception as e:
            logger.error(f"Error setting reference image from data: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _process_reference_image(self, image: np.ndarray, resize: List[int], source_name: str) -> Dict:
        """
        Common processing logic for reference images
        
        Args:
            image: Grayscale image as numpy array
            resize: Resize dimensions [width, height]
            source_name: Name/path of the image source
            
        Returns:
            Dictionary with status and image info
        """
        try:
            # Resize image
            if len(resize) == 2:
                image = cv2.resize(image, tuple(resize), interpolation=cv2.INTER_AREA)
            elif len(resize) == 1 and resize[0] > 0:
                h, w = image.shape
                scale = resize[0] / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Convert to tensor
            image_tensor = frame2tensor(image, self.device)
            
            # Extract features
            data = self.matching.superpoint({'image': image_tensor})
            self.reference_data = {k+'0': data[k] for k in self.keys}
            self.reference_data['image0'] = image_tensor
            self.reference_image = image
            self.reference_path = source_name
            
            return {
                "status": "success",
                "message": "Reference image set successfully",
                "keypoints_count": len(data['keypoints'][0]),
                "image_shape": image.shape,
                "source": source_name
            }
            
        except Exception as e:
            logger.error(f"Error processing reference image: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def match_with_image(self, query_image_path: str, resize: List[int] = [640, 480], 
                        visualize: bool = True, output_path: Optional[str] = None) -> Dict:
        """
        Match reference image with a query image from file path
        
        Args:
            query_image_path: Path to the query image
            resize: Resize dimensions
            visualize: Whether to create visualization
            output_path: Path to save visualization (optional)
            
        Returns:
            Dictionary with matching results
        """
        if self.reference_data is None:
            return {"status": "error", "message": "No reference image set"}
        
        try:
            # Read and process query image
            query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
            if query_image is None:
                return {"status": "error", "message": f"Could not read image: {query_image_path}"}
            
            return self._match_with_image_data(query_image, resize, visualize, output_path, query_image_path)
            
        except Exception as e:
            logger.error(f"Error in image matching: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def match_with_image_data(self, image_data: Union[np.ndarray, bytes], 
                             resize: List[int] = [640, 480], visualize: bool = True, 
                             output_path: Optional[str] = None, filename: str = "query_image") -> Dict:
        """
        Match reference image with a query image from image data
        
        Args:
            image_data: Query image data as numpy array or bytes
            resize: Resize dimensions
            visualize: Whether to create visualization
            output_path: Path to save visualization (optional)
            filename: Name for the query image (for reference)
            
        Returns:
            Dictionary with matching results
        """
        if self.reference_data is None:
            return {"status": "error", "message": "No reference image set"}
        
        try:
            # Handle different input formats
            if isinstance(image_data, bytes):
                # Convert bytes to numpy array
                nparr = np.frombuffer(image_data, np.uint8)
                query_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                if query_image is None:
                    return {"status": "error", "message": "Could not decode image data"}
            elif isinstance(image_data, np.ndarray):
                # If already a numpy array, convert to grayscale if needed
                if len(image_data.shape) == 3:
                    query_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
                else:
                    query_image = image_data
            else:
                return {"status": "error", "message": "Unsupported image data format"}
            
            return self._match_with_image_data(query_image, resize, visualize, output_path, filename)
            
        except Exception as e:
            logger.error(f"Error in image matching with data: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _match_with_image_data(self, query_image: np.ndarray, resize: List[int], 
                              visualize: bool, output_path: Optional[str], source_name: str) -> Dict:
        """
        Common matching logic for query images
        
        Args:
            query_image: Grayscale query image as numpy array
            resize: Resize dimensions
            visualize: Whether to create visualization
            output_path: Path to save visualization (optional)
            source_name: Name/path of the query image source
            
        Returns:
            Dictionary with matching results
        """
        try:
            # Resize query image
            if len(resize) == 2:
                query_image = cv2.resize(query_image, tuple(resize), interpolation=cv2.INTER_AREA)
            elif len(resize) == 1 and resize[0] > 0:
                h, w = query_image.shape
                scale = resize[0] / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                query_image = cv2.resize(query_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Convert to tensor and extract features
            query_tensor = frame2tensor(query_image, self.device)
            
            # Perform matching
            self.timer.update('data')
            pred = self.matching({**self.reference_data, 'image1': query_tensor})
            self.timer.update('forward')
            
            # Extract results
            kpts0 = self.reference_data['keypoints0'][0].detach().cpu().numpy()
            kpts1 = pred['keypoints1'][0].detach().cpu().numpy()
            matches = pred['matches0'][0].detach().cpu().numpy()
            confidence = pred['matching_scores0'][0].detach().cpu().numpy()
            
            # Filter valid matches
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            match_confidence = confidence[valid]
            
            result = {
                "status": "success",
                "num_keypoints_ref": len(kpts0),
                "num_keypoints_query": len(kpts1),
                "num_matches": len(mkpts0),
                "query_source": source_name,
                "matches": {
                    "keypoints_ref": mkpts0.tolist(),
                    "keypoints_query": mkpts1.tolist(),
                    "confidence": match_confidence.tolist()
                },
                "timing": {
                    "forward_pass": self.timer.times['forward']
                }
            }
            
            # Create visualization if requested
            if visualize:
                color = cm.jet(match_confidence)
                text = [
                    'SuperGlue',
                    f'Keypoints: {len(kpts0)}:{len(kpts1)}',
                    f'Matches: {len(mkpts0)}'
                ]
                
                viz_image = make_matching_plot_fast(
                    self.reference_image, query_image, kpts0, kpts1, mkpts0, mkpts1, 
                    color, text, path=output_path, show_keypoints=True
                )
                
                if output_path:
                    cv2.imwrite(output_path, viz_image)
                    result["visualization_path"] = output_path
                else:
                    # Convert to base64 for API response
                    _, buffer = cv2.imencode('.png', viz_image)
                    viz_base64 = base64.b64encode(buffer).decode('utf-8')
                    result["visualization_base64"] = viz_base64
            
            self.timer.update('viz')
            return result
            
        except Exception as e:
            logger.error(f"Error in matching processing: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def start_video_stream_matching(self, stream_source: str, resize: List[int] = [640, 480]) -> Dict:
        """
        Start matching with a video stream
        
        Args:
            stream_source: Video source (IP camera URL, video file, or webcam ID)
            resize: Resize dimensions
            
        Returns:
            Dictionary with status
        """
        if self.reference_data is None:
            return {"status": "error", "message": "No reference image set"}
        
        try:
            # Initialize video streamer
            self.video_streamer = VideoStreamer(stream_source, resize, skip=1, 
                                              image_glob=['*.png', '*.jpg', '*.jpeg'])
            self.streaming_active = True
            
            return {
                "status": "success",
                "message": "Video stream started successfully",
                "source": stream_source
            }
            
        except Exception as e:
            logger.error(f"Error starting video stream: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_next_frame_match(self, visualize: bool = True) -> Dict:
        """
        Get matching results for the next frame in the video stream
        
        Args:
            visualize: Whether to create visualization
            
        Returns:
            Dictionary with matching results
        """
        if not self.streaming_active or self.video_streamer is None:
            return {"status": "error", "message": "No active video stream"}
        
        if self.reference_data is None:
            return {"status": "error", "message": "No reference image set"}
        
        try:
            # Get next frame
            frame, ret = self.video_streamer.next_frame()
            if not ret:
                self.streaming_active = False
                return {"status": "end", "message": "Video stream ended"}
            
            # Convert frame to tensor
            frame_tensor = frame2tensor(frame, self.device)
            
            # Perform matching
            self.timer.update('data')
            pred = self.matching({**self.reference_data, 'image1': frame_tensor})
            self.timer.update('forward')
            
            # Extract results
            kpts0 = self.reference_data['keypoints0'][0].detach().cpu().numpy()
            kpts1 = pred['keypoints1'][0].detach().cpu().numpy()
            matches = pred['matches0'][0].detach().cpu().numpy()
            confidence = pred['matching_scores0'][0].detach().cpu().numpy()
            
            # Filter valid matches
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            match_confidence = confidence[valid]
            
            result = {
                "status": "success",
                "frame_id": self.video_streamer.i - 1,
                "num_keypoints_ref": len(kpts0),
                "num_keypoints_query": len(kpts1),
                "num_matches": len(mkpts0),
                "matches": {
                    "keypoints_ref": mkpts0.tolist(),
                    "keypoints_query": mkpts1.tolist(),
                    "confidence": match_confidence.tolist()
                },
                "timing": {
                    "forward_pass": self.timer.times['forward']
                }
            }
            
            # Create visualization if requested
            if visualize:
                color = cm.jet(match_confidence)
                text = [
                    'SuperGlue',
                    f'Keypoints: {len(kpts0)}:{len(kpts1)}',
                    f'Matches: {len(mkpts0)}'
                ]
                
                viz_image = make_matching_plot_fast(
                    self.reference_image, frame, kpts0, kpts1, mkpts0, mkpts1, 
                    color, text, show_keypoints=True
                )
                
                # Convert to base64 for API response
                _, buffer = cv2.imencode('.png', viz_image)
                viz_base64 = base64.b64encode(buffer).decode('utf-8')
                result["visualization_base64"] = viz_base64
            
            self.timer.update('viz')
            return result
            
        except Exception as e:
            logger.error(f"Error in video frame matching: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def stop_video_stream(self) -> Dict:
        """Stop the video stream"""
        try:
            if self.video_streamer:
                self.video_streamer.cleanup()
            self.streaming_active = False
            self.video_streamer = None
            
            return {"status": "success", "message": "Video stream stopped"}
            
        except Exception as e:
            logger.error(f"Error stopping video stream: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def start_video_recording(self, stream_source: str, output_path: str, 
                             resize: List[int] = [640, 480], fps: float = 30.0,
                             show_keypoints: bool = True) -> Dict:
        """
        Start recording a video with SuperGlue matches
        
        Args:
            stream_source: Video source (IP camera URL, video file, or webcam ID)
            output_path: Path where to save the output video
            resize: Resize dimensions
            fps: Output video frame rate
            show_keypoints: Whether to show keypoints in visualization
            
        Returns:
            Dictionary with status
        """
        if self.reference_data is None:
            return {"status": "error", "message": "No reference image set"}
        
        if self.recording_active:
            return {"status": "error", "message": "Recording already active"}
        
        try:
            # Initialize video streamer
            self.video_streamer = VideoStreamer(stream_source, resize, skip=1, 
                                              image_glob=['*.png', '*.jpg', '*.jpeg'])
            
            # Get first frame to determine video dimensions
            frame, ret = self.video_streamer.next_frame()
            if not ret:
                return {"status": "error", "message": "Could not read first frame"}
            
            # Create visualization for first frame to get output dimensions
            frame_tensor = frame2tensor(frame, self.device)
            pred = self.matching({**self.reference_data, 'image1': frame_tensor})
            
            kpts0 = self.reference_data['keypoints0'][0].detach().cpu().numpy()
            kpts1 = pred['keypoints1'][0].detach().cpu().numpy()
            matches = pred['matches0'][0].detach().cpu().numpy()
            confidence = pred['matching_scores0'][0].detach().cpu().numpy()
            
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            match_confidence = confidence[valid]
            
            color = cm.jet(match_confidence)
            text = [
                'SuperGlue',
                f'Keypoints: {len(kpts0)}:{len(kpts1)}',
                f'Matches: {len(mkpts0)}'
            ]
            
            viz_frame = make_matching_plot_fast(
                self.reference_image, frame, kpts0, kpts1, mkpts0, mkpts1, 
                color, text, show_keypoints=show_keypoints
            )
            
            # Initialize video writer
            height, width = viz_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not self.video_writer.isOpened():
                return {"status": "error", "message": "Could not initialize video writer"}
            
            # Write first frame
            self.video_writer.write(viz_frame)
            
            # Start recording thread
            self.recording_active = True
            self.recording_thread = threading.Thread(
                target=self._recording_worker, 
                args=(show_keypoints,),
                daemon=True
            )
            self.recording_thread.start()
            
            return {
                "status": "success",
                "message": "Video recording started",
                "output_path": output_path,
                "video_dimensions": (width, height),
                "fps": fps
            }
            
        except Exception as e:
            logger.error(f"Error starting video recording: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _recording_worker(self, show_keypoints: bool):
        """Worker thread for video recording"""
        try:
            while self.recording_active and self.video_streamer:
                frame, ret = self.video_streamer.next_frame()
                if not ret:
                    break
                
                # Convert frame to tensor
                frame_tensor = frame2tensor(frame, self.device)
                
                # Perform matching
                pred = self.matching({**self.reference_data, 'image1': frame_tensor})
                
                # Extract results
                kpts0 = self.reference_data['keypoints0'][0].detach().cpu().numpy()
                kpts1 = pred['keypoints1'][0].detach().cpu().numpy()
                matches = pred['matches0'][0].detach().cpu().numpy()
                confidence = pred['matching_scores0'][0].detach().cpu().numpy()
                
                # Filter valid matches
                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                match_confidence = confidence[valid]
                
                # Create visualization
                color = cm.jet(match_confidence) if len(match_confidence) > 0 else []
                text = [
                    'SuperGlue',
                    f'Keypoints: {len(kpts0)}:{len(kpts1)}',
                    f'Matches: {len(mkpts0)}',
                    f'Frame: {self.video_streamer.i - 1}'
                ]
                
                viz_frame = make_matching_plot_fast(
                    self.reference_image, frame, kpts0, kpts1, mkpts0, mkpts1, 
                    color, text, show_keypoints=show_keypoints
                )
                
                # Write frame to video
                if self.video_writer and self.video_writer.isOpened():
                    self.video_writer.write(viz_frame)
                
        except Exception as e:
            logger.error(f"Error in recording worker: {str(e)}")
        
        finally:
            self.recording_active = False
    
    def stop_video_recording(self) -> Dict:
        """Stop video recording"""
        try:
            self.recording_active = False
            
            # Wait for recording thread to finish
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=5.0)
            
            # Release video writer
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            # Stop video streamer
            if self.video_streamer:
                self.video_streamer.cleanup()
                self.video_streamer = None
            
            self.streaming_active = False
            
            return {"status": "success", "message": "Video recording stopped"}
            
        except Exception as e:
            logger.error(f"Error stopping video recording: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_recording_status(self) -> Dict:
        """Get current recording status"""
        return {
            "recording_active": self.recording_active,
            "streaming_active": self.streaming_active,
            "video_writer_active": self.video_writer is not None and self.video_writer.isOpened() if self.video_writer else False
        }
    
    def generate_stream_frames(self, show_keypoints: bool = True):
        """
        Generator function that yields processed frames for video streaming
        
        Args:
            show_keypoints: Whether to show keypoints in visualization
            
        Yields:
            Processed frame bytes for streaming
        """
        if not self.streaming_active or self.video_streamer is None:
            return
        
        if self.reference_data is None:
            return
        
        try:
            while self.streaming_active and self.video_streamer:
                frame, ret = self.video_streamer.next_frame()
                if not ret:
                    break
                
                # Convert frame to tensor
                frame_tensor = frame2tensor(frame, self.device)
                
                # Perform matching
                pred = self.matching({**self.reference_data, 'image1': frame_tensor})
                
                # Extract results
                kpts0 = self.reference_data['keypoints0'][0].detach().cpu().numpy()
                kpts1 = pred['keypoints1'][0].detach().cpu().numpy()
                matches = pred['matches0'][0].detach().cpu().numpy()
                confidence = pred['matching_scores0'][0].detach().cpu().numpy()
                
                # Filter valid matches
                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                match_confidence = confidence[valid]
                
                # Create visualization
                color = cm.jet(match_confidence) if len(match_confidence) > 0 else []
                text = [
                    'SuperGlue',
                    f'Keypoints: {len(kpts0)}:{len(kpts1)}',
                    f'Matches: {len(mkpts0)}',
                    f'Frame: {self.video_streamer.i - 1}'
                ]
                
                # Add threshold information
                k_thresh = self.matching.superpoint.config['keypoint_threshold']
                m_thresh = self.matching.superglue.config['match_threshold']
                small_text = [
                    f'Keypoint Threshold: {k_thresh:.4f}',
                    f'Match Threshold: {m_thresh:.2f}',
                    f'Frame: {self.video_streamer.i - 1:06d}'
                ]
                
                viz_frame = make_matching_plot_fast(
                    self.reference_image, frame, kpts0, kpts1, mkpts0, mkpts1, 
                    color, text, show_keypoints=show_keypoints, small_text=small_text
                )
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', viz_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                # Yield frame in multipart format for streaming
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
        except Exception as e:
            logger.error(f"Error in stream generation: {str(e)}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "device": self.device,
            "config": self.config,
            "reference_image_set": self.reference_data is not None,
            "reference_image_path": self.reference_path,
            "streaming_active": self.streaming_active,
            "recording_active": self.recording_active
        }


# Flask API for the backend
app = Flask(__name__)
CORS(app)

# Global backend instance
backend = None


@app.route('/api/info', methods=['GET'])
def get_info():
    """Get backend information"""
    if backend is None:
        return jsonify({"status": "error", "message": "Backend not initialized"})
    return jsonify(backend.get_model_info())


@app.route('/api/set_reference', methods=['POST'])
def set_reference():
    """Set reference image from file upload or image path"""
    if backend is None:
        return jsonify({"status": "error", "message": "Backend not initialized"})
    
    try:
        # Check if this is a file upload
        if 'image' in request.files:
            # Handle file upload
            file = request.files['image']
            if file.filename == '':
                return jsonify({"status": "error", "message": "No file selected"})
            
            # Get resize parameter from form data
            resize_str = request.form.get('resize', '[640, 480]')
            try:
                resize = json.loads(resize_str) if isinstance(resize_str, str) else resize_str
            except (json.JSONDecodeError, TypeError):
                resize = [640, 480]
            
            # Read file data
            file_data = file.read()
            filename = file.filename or "uploaded_image"
            
            result = backend.set_reference_image_from_data(file_data, resize, filename)
            return jsonify(result)
        
        # Handle traditional JSON request with image path
        elif request.is_json:
            data = request.get_json()
            image_path = data.get('image_path')
            resize = data.get('resize', [640, 480])
            
            if not image_path:
                return jsonify({"status": "error", "message": "image_path required when not uploading file"})
            
            result = backend.set_reference_image(image_path, resize)
            return jsonify(result)
        
        # Handle base64 encoded image in JSON
        elif request.is_json:
            data = request.get_json()
            image_base64 = data.get('image_base64')
            resize = data.get('resize', [640, 480])
            filename = data.get('filename', 'base64_image')
            
            if image_base64:
                try:
                    # Decode base64 image
                    image_data = base64.b64decode(image_base64)
                    result = backend.set_reference_image_from_data(image_data, resize, filename)
                    return jsonify(result)
                except Exception as e:
                    return jsonify({"status": "error", "message": f"Failed to decode base64 image: {str(e)}"})
        
        return jsonify({"status": "error", "message": "Either upload a file, provide image_path, or image_base64"})
        
    except Exception as e:
        logger.error(f"Error in set_reference endpoint: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/match_image', methods=['POST'])
def match_image():
    """Match with another image from file upload or image path"""
    if backend is None:
        return jsonify({"status": "error", "message": "Backend not initialized"})
    
    try:
        # Check if this is a file upload
        if 'image' in request.files:
            # Handle file upload
            file = request.files['image']
            if file.filename == '':
                return jsonify({"status": "error", "message": "No file selected"})
            
            # Get parameters from form data
            resize_str = request.form.get('resize', '[640, 480]')
            try:
                resize = json.loads(resize_str) if isinstance(resize_str, str) else resize_str
            except (json.JSONDecodeError, TypeError):
                resize = [640, 480]
            
            visualize = request.form.get('visualize', 'true').lower() == 'true'
            output_path = request.form.get('output_path')
            
            # Read file data
            file_data = file.read()
            filename = file.filename or "query_image"
            
            result = backend.match_with_image_data(file_data, resize, visualize, output_path, filename)
            return jsonify(result)
        
        # Handle traditional JSON request with image path
        elif request.is_json:
            data = request.get_json()
            query_path = data.get('query_image_path')
            resize = data.get('resize', [640, 480])
            visualize = data.get('visualize', True)
            output_path = data.get('output_path')
            
            if query_path:
                result = backend.match_with_image(query_path, resize, visualize, output_path)
                return jsonify(result)
            
            # Handle base64 encoded image in JSON
            image_base64 = data.get('image_base64')
            filename = data.get('filename', 'base64_query')
            
            if image_base64:
                try:
                    # Decode base64 image
                    image_data = base64.b64decode(image_base64)
                    result = backend.match_with_image_data(image_data, resize, visualize, output_path, filename)
                    return jsonify(result)
                except Exception as e:
                    return jsonify({"status": "error", "message": f"Failed to decode base64 image: {str(e)}"})
        
        return jsonify({"status": "error", "message": "Either upload a file, provide query_image_path, or image_base64"})
        
    except Exception as e:
        logger.error(f"Error in match_image endpoint: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/start_stream', methods=['POST'])
def start_stream():
    """Start video stream matching"""
    if backend is None:
        return jsonify({"status": "error", "message": "Backend not initialized"})
    
    data = request.get_json()
    stream_source = data.get('stream_source')
    resize = data.get('resize', [640, 480])
    
    if not stream_source:
        return jsonify({"status": "error", "message": "stream_source required"})
    
    result = backend.start_video_stream_matching(stream_source, resize)
    return jsonify(result)


@app.route('/api/next_frame', methods=['GET'])
def next_frame():
    """Get next frame matching results"""
    if backend is None:
        return jsonify({"status": "error", "message": "Backend not initialized"})
    
    visualize = request.args.get('visualize', 'true').lower() == 'true'
    result = backend.get_next_frame_match(visualize)
    return jsonify(result)


@app.route('/api/stop_stream', methods=['POST'])
def stop_stream():
    """Stop video stream"""
    if backend is None:
        return jsonify({"status": "error", "message": "Backend not initialized"})
    
    result = backend.stop_video_stream()
    return jsonify(result)


@app.route('/api/start_recording', methods=['POST'])
def start_recording():
    """Start video recording with SuperGlue matches"""
    if backend is None:
        return jsonify({"status": "error", "message": "Backend not initialized"})
    
    data = request.get_json()
    stream_source = data.get('stream_source')
    output_path = data.get('output_path')
    resize = data.get('resize', [640, 480])
    fps = data.get('fps', 30.0)
    show_keypoints = data.get('show_keypoints', True)
    
    if not stream_source:
        return jsonify({"status": "error", "message": "stream_source required"})
    
    if not output_path:
        return jsonify({"status": "error", "message": "output_path required"})
    
    result = backend.start_video_recording(stream_source, output_path, resize, fps, show_keypoints)
    return jsonify(result)


@app.route('/api/stop_recording', methods=['POST'])
def stop_recording():
    """Stop video recording"""
    if backend is None:
        return jsonify({"status": "error", "message": "Backend not initialized"})
    
    result = backend.stop_video_recording()
    return jsonify(result)


@app.route('/api/recording_status', methods=['GET'])
def recording_status():
    """Get recording status"""
    if backend is None:
        return jsonify({"status": "error", "message": "Backend not initialized"})
    
    result = backend.get_recording_status()
    return jsonify(result)


@app.route('/api/video_stream')
def video_stream():
    """Stream processed video with SuperGlue matches"""
    if backend is None:
        return jsonify({"status": "error", "message": "Backend not initialized"})
    
    if not backend.streaming_active:
        return jsonify({"status": "error", "message": "No active video stream. Start streaming first with /api/start_stream"})
    
    show_keypoints = request.args.get('show_keypoints', 'true').lower() == 'true'
    
    return Response(
        backend.generate_stream_frames(show_keypoints),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for container orchestration"""
    try:
        # Basic health checks
        health_status = {
            "status": "healthy",
            "service": "iot-localization-service",
            "backend_initialized": backend is not None,
            "device": backend.device if backend else "unknown",
            "reference_image_set": backend.reference_data is not None if backend else False,
            "streaming_active": backend.streaming_active if backend else False,
            "model_config": backend.config if backend else None
        }
        return jsonify(health_status), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "service": "iot-localization-service",
            "error": str(e)
        }), 500


@app.route('/', methods=['GET'])
def index():
    """Simple test page for file uploads"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>SuperGlue Backend - File Upload Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .form-group { margin: 20px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select { padding: 8px; margin: 5px 0; width: 300px; }
            button { padding: 10px 20px; background: #007cba; color: white; border: none; cursor: pointer; }
            button:hover { background: #005a87; }
            .result { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; }
            .error { background: #ffebee; color: #c62828; }
            .success { background: #e8f5e8; color: #2e7d32; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>SuperGlue Backend - File Upload Test</h1>
            
            <h2>Set Reference Image</h2>
            <form id="referenceForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="refImage">Reference Image:</label>
                    <input type="file" id="refImage" name="image" accept="image/*" required>
                </div>
                <div class="form-group">
                    <label for="refResize">Resize (width,height):</label>
                    <input type="text" id="refResize" name="resize" value="[640, 480]" placeholder="[640, 480]">
                </div>
                <button type="submit">Set Reference Image</button>
            </form>
            <div id="referenceResult" class="result" style="display:none;"></div>
            
            <h2>Match Query Image</h2>
            <form id="queryForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="queryImage">Query Image:</label>
                    <input type="file" id="queryImage" name="image" accept="image/*" required>
                </div>
                <div class="form-group">
                    <label for="queryResize">Resize (width,height):</label>
                    <input type="text" id="queryResize" name="resize" value="[640, 480]" placeholder="[640, 480]">
                </div>
                <div class="form-group">
                    <label for="visualize">Create Visualization:</label>
                    <select id="visualize" name="visualize">
                        <option value="true">Yes</option>
                        <option value="false">No</option>
                    </select>
                </div>
                <button type="submit">Match Images</button>
            </form>
            <div id="queryResult" class="result" style="display:none;"></div>
            
            <h2>Video Recording</h2>
            <form id="recordingForm">
                <div class="form-group">
                    <label for="streamSource">Stream Source:</label>
                    <input type="text" id="streamSource" name="stream_source" value="0" placeholder="0 for webcam, or video file path">
                </div>
                <div class="form-group">
                    <label for="outputPath">Output Video Path:</label>
                    <input type="text" id="outputPath" name="output_path" value="output_matches.mp4" placeholder="output_matches.mp4">
                </div>
                <div class="form-group">
                    <label for="recordResize">Resize (width,height):</label>
                    <input type="text" id="recordResize" name="resize" value="[640, 480]" placeholder="[640, 480]">
                </div>
                <div class="form-group">
                    <label for="fps">FPS:</label>
                    <input type="number" id="fps" name="fps" value="30" min="1" max="60">
                </div>
                <div class="form-group">
                    <label for="showKeypoints">Show Keypoints:</label>
                    <select id="showKeypoints" name="show_keypoints">
                        <option value="true">Yes</option>
                        <option value="false">No</option>
                    </select>
                </div>
                <button type="button" onclick="startRecording()">Start Recording</button>
                <button type="button" onclick="stopRecording()">Stop Recording</button>
                <button type="button" onclick="getRecordingStatus()">Get Status</button>
            </form>
            <div id="recordingResult" class="result" style="display:none;"></div>
            
            <h2>Live Video Stream</h2>
            <form id="streamForm">
                <div class="form-group">
                    <label for="streamSourceLive">Stream Source:</label>
                    <input type="text" id="streamSourceLive" name="stream_source" value="0" placeholder="0 for webcam, or video file path">
                </div>
                <div class="form-group">
                    <label for="streamResize">Resize (width,height):</label>
                    <input type="text" id="streamResize" name="resize" value="[640, 480]" placeholder="[640, 480]">
                </div>
                <div class="form-group">
                    <label for="showKeypointsStream">Show Keypoints:</label>
                    <select id="showKeypointsStream" name="show_keypoints">
                        <option value="true">Yes</option>
                        <option value="false">No</option>
                    </select>
                </div>
                <button type="button" onclick="startLiveStream()">Start Live Stream</button>
                <button type="button" onclick="stopLiveStream()">Stop Live Stream</button>
            </form>
            <div id="streamResult" class="result" style="display:none;"></div>
            <div id="videoContainer" style="display:none; margin-top: 20px;">
                <h3>Live SuperGlue Matches Stream</h3>
                <img id="videoStream" style="max-width: 100%; border: 2px solid #007cba;" alt="Live Stream">
                <p><strong>Stream URL:</strong> <span id="streamUrl"></span></p>
            </div>
            
            <h2>Backend Info</h2>
            <button onclick="getInfo()">Get Backend Info</button>
            <div id="infoResult" class="result" style="display:none;"></div>
        </div>
        
        <script>
            document.getElementById('referenceForm').onsubmit = function(e) {
                e.preventDefault();
                const formData = new FormData(this);
                submitForm('/api/set_reference', formData, 'referenceResult');
            };
            
            document.getElementById('queryForm').onsubmit = function(e) {
                e.preventDefault();
                const formData = new FormData(this);
                submitForm('/api/match_image', formData, 'queryResult');
            };
            
            function submitForm(url, formData, resultId) {
                const resultDiv = document.getElementById(resultId);
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = 'Processing...';
                resultDiv.className = 'result';
                
                fetch(url, {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        resultDiv.className = 'result success';
                        let content = '<strong>Success!</strong><br>';
                        content += JSON.stringify(data, null, 2);
                        
                        // Show visualization if available
                        if (data.visualization_base64) {
                            content += '<br><br><strong>Visualization:</strong><br>';
                            content += '<img src="data:image/png;base64,' + data.visualization_base64 + '" style="max-width:100%; border:1px solid #ccc;">';
                        }
                        
                        resultDiv.innerHTML = '<pre>' + content + '</pre>';
                    } else {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = '<strong>Error:</strong><br><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    }
                })
                .catch(error => {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = '<strong>Network Error:</strong><br>' + error.message;
                });
            }
            
            function getInfo() {
                const resultDiv = document.getElementById('infoResult');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = 'Loading...';
                resultDiv.className = 'result';
                
                fetch('/api/info')
                .then(response => response.json())
                .then(data => {
                    resultDiv.className = 'result success';
                    resultDiv.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                })
                .catch(error => {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = '<strong>Error:</strong><br>' + error.message;
                });
            }
            
            function startLiveStream() {
                const form = document.getElementById('streamForm');
                const formData = new FormData(form);
                const data = {
                    stream_source: formData.get('stream_source'),
                    resize: JSON.parse(formData.get('resize') || '[640, 480]')
                };
                
                const resultDiv = document.getElementById('streamResult');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = 'Starting stream...';
                resultDiv.className = 'result';
                
                fetch('/api/start_stream', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        resultDiv.className = 'result success';
                        resultDiv.innerHTML = '<strong>Stream started successfully!</strong>';
                        
                        // Show video container and start streaming
                        const videoContainer = document.getElementById('videoContainer');
                        const videoStream = document.getElementById('videoStream');
                        const streamUrl = document.getElementById('streamUrl');
                        const showKeypoints = document.getElementById('showKeypointsStream').value;
                        
                        const streamEndpoint = '/api/video_stream?show_keypoints=' + showKeypoints;
                        videoStream.src = streamEndpoint;
                        streamUrl.textContent = window.location.origin + streamEndpoint;
                        videoContainer.style.display = 'block';
                        
                    } else {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = '<strong>Error:</strong><br><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    }
                })
                .catch(error => {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = '<strong>Network Error:</strong><br>' + error.message;
                });
            }
            
            function stopLiveStream() {
                const resultDiv = document.getElementById('streamResult');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = 'Stopping stream...';
                resultDiv.className = 'result';
                
                fetch('/api/stop_stream', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        resultDiv.className = 'result success';
                        resultDiv.innerHTML = '<strong>Stream stopped successfully!</strong>';
                        
                        // Hide video container
                        const videoContainer = document.getElementById('videoContainer');
                        const videoStream = document.getElementById('videoStream');
                        videoStream.src = '';
                        videoContainer.style.display = 'none';
                        
                    } else {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = '<strong>Error:</strong><br><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    }
                })
                .catch(error => {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = '<strong>Network Error:</strong><br>' + error.message;
                });
            }
            
            function startRecording() {
                const form = document.getElementById('recordingForm');
                const formData = new FormData(form);
                const data = {
                    stream_source: formData.get('stream_source'),
                    output_path: formData.get('output_path'),
                    resize: JSON.parse(formData.get('resize') || '[640, 480]'),
                    fps: parseFloat(formData.get('fps')),
                    show_keypoints: formData.get('show_keypoints') === 'true'
                };
                
                const resultDiv = document.getElementById('recordingResult');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = 'Starting recording...';
                resultDiv.className = 'result';
                
                fetch('/api/start_recording', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        resultDiv.className = 'result success';
                        resultDiv.innerHTML = '<strong>Recording started!</strong><br><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    } else {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = '<strong>Error:</strong><br><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    }
                })
                .catch(error => {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = '<strong>Network Error:</strong><br>' + error.message;
                });
            }
            
            function stopRecording() {
                const resultDiv = document.getElementById('recordingResult');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = 'Stopping recording...';
                resultDiv.className = 'result';
                
                fetch('/api/stop_recording', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        resultDiv.className = 'result success';
                        resultDiv.innerHTML = '<strong>Recording stopped!</strong><br><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    } else {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = '<strong>Error:</strong><br><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    }
                })
                .catch(error => {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = '<strong>Network Error:</strong><br>' + error.message;
                });
            }
            
            function getRecordingStatus() {
                const resultDiv = document.getElementById('recordingResult');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = 'Getting status...';
                resultDiv.className = 'result';
                
                fetch('/api/recording_status')
                .then(response => response.json())
                .then(data => {
                    resultDiv.className = 'result success';
                    resultDiv.innerHTML = '<strong>Recording Status:</strong><br><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                })
                .catch(error => {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = '<strong>Error:</strong><br>' + error.message;
                });
            }
        </script>
    </body>
    </html>
    '''


def main():
    """Main function to run the backend"""
    parser = argparse.ArgumentParser(
        description='SuperGlue Backend API',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '--host', type=str, default='localhost',
        help='Host to run the API server on')
    parser.add_argument(
        '--port', type=int, default=5000,
        help='Port to run the API server on')
    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by SuperPoint')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'superpoint': {
            'nms_radius': args.nms_radius,
            'keypoint_threshold': args.keypoint_threshold,
            'max_keypoints': args.max_keypoints
        },
        'superglue': {
            'weights': args.superglue,
            'sinkhorn_iterations': args.sinkhorn_iterations,
            'match_threshold': args.match_threshold,
        },
        'force_cpu': args.force_cpu
    }
    
    # Initialize backend
    global backend
    backend = SuperGlueBackend(config)
    
    logger.info(f"Starting SuperGlue Backend API on {args.host}:{args.port}")
    logger.info(f"Model: {args.superglue}, Device: {backend.device}")
    
    # Run the Flask app
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
