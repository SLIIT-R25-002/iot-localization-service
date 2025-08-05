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

import cv2
import numpy as np
import torch
import matplotlib.cm as cm
from flask import Flask, request, jsonify, send_file
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
        
        # Timer for performance monitoring
        self.timer = AverageTimer()
        
    def set_reference_image(self, image_path: str, resize: List[int] = [640, 480]) -> Dict:
        """
        Set the reference image for matching
        
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
            self.reference_path = image_path
            
            return {
                "status": "success",
                "message": "Reference image set successfully",
                "keypoints_count": len(data['keypoints'][0]),
                "image_shape": image.shape
            }
            
        except Exception as e:
            logger.error(f"Error setting reference image: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def match_with_image(self, query_image_path: str, resize: List[int] = [640, 480], 
                        visualize: bool = True, output_path: Optional[str] = None) -> Dict:
        """
        Match reference image with a query image
        
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
            kpts0 = self.reference_data['keypoints0'][0].cpu().numpy()
            kpts1 = pred['keypoints1'][0].cpu().numpy()
            matches = pred['matches0'][0].cpu().numpy()
            confidence = pred['matching_scores0'][0].cpu().numpy()
            
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
            logger.error(f"Error in image matching: {str(e)}")
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
            kpts0 = self.reference_data['keypoints0'][0].cpu().numpy()
            kpts1 = pred['keypoints1'][0].cpu().numpy()
            matches = pred['matches0'][0].cpu().numpy()
            confidence = pred['matching_scores0'][0].cpu().numpy()
            
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
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "device": self.device,
            "config": self.config,
            "reference_image_set": self.reference_data is not None,
            "reference_image_path": self.reference_path,
            "streaming_active": self.streaming_active
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
    """Set reference image"""
    if backend is None:
        return jsonify({"status": "error", "message": "Backend not initialized"})
    
    data = request.get_json()
    image_path = data.get('image_path')
    resize = data.get('resize', [640, 480])
    
    if not image_path:
        return jsonify({"status": "error", "message": "image_path required"})
    
    result = backend.set_reference_image(image_path, resize)
    return jsonify(result)


@app.route('/api/match_image', methods=['POST'])
def match_image():
    """Match with another image"""
    if backend is None:
        return jsonify({"status": "error", "message": "Backend not initialized"})
    
    data = request.get_json()
    query_path = data.get('query_image_path')
    resize = data.get('resize', [640, 480])
    visualize = data.get('visualize', True)
    output_path = data.get('output_path')
    
    if not query_path:
        return jsonify({"status": "error", "message": "query_image_path required"})
    
    result = backend.match_with_image(query_path, resize, visualize, output_path)
    return jsonify(result)


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
