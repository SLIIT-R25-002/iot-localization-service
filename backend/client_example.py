#!/usr/bin/env python3
"""
SuperGlue Backend Client Example
Demonstrates how to use the SuperGlue backend API for image and video stream matching.
"""

import requests
import json
import time
import base64
import cv2
import numpy as np
from pathlib import Path
import argparse


class SuperGlueClient:
    """Client for SuperGlue Backend API"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Initialize the client
        
        Args:
            base_url: Base URL of the SuperGlue backend API
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def get_info(self):
        """Get backend information"""
        response = self.session.get(f"{self.base_url}/api/info")
        return response.json()
    
    def set_reference_image(self, image_path: str, resize: list = [640, 480]):
        """
        Set the reference image
        
        Args:
            image_path: Path to the reference image
            resize: Resize dimensions [width, height]
        """
        data = {
            "image_path": image_path,
            "resize": resize
        }
        response = self.session.post(f"{self.base_url}/api/set_reference", json=data)
        return response.json()
    
    def match_with_image(self, query_image_path: str, resize: list = [640, 480], 
                        visualize: bool = True, output_path: str = None):
        """
        Match reference image with a query image
        
        Args:
            query_image_path: Path to the query image
            resize: Resize dimensions
            visualize: Whether to create visualization
            output_path: Path to save visualization
        """
        data = {
            "query_image_path": query_image_path,
            "resize": resize,
            "visualize": visualize,
            "output_path": output_path
        }
        response = self.session.post(f"{self.base_url}/api/match_image", json=data)
        return response.json()
    
    def start_video_stream(self, stream_source: str, resize: list = [640, 480]):
        """
        Start video stream matching
        
        Args:
            stream_source: Video source (IP camera URL, video file, or webcam ID)
            resize: Resize dimensions
        """
        data = {
            "stream_source": stream_source,
            "resize": resize
        }
        response = self.session.post(f"{self.base_url}/api/start_stream", json=data)
        return response.json()
    
    def get_next_frame_match(self, visualize: bool = True):
        """
        Get matching results for next frame
        
        Args:
            visualize: Whether to create visualization
        """
        params = {"visualize": str(visualize).lower()}
        response = self.session.get(f"{self.base_url}/api/next_frame", params=params)
        return response.json()
    
    def stop_video_stream(self):
        """Stop video stream"""
        response = self.session.post(f"{self.base_url}/api/stop_stream")
        return response.json()
    
    def save_visualization_from_base64(self, base64_data: str, output_path: str):
        """
        Save visualization image from base64 data
        
        Args:
            base64_data: Base64 encoded image data
            output_path: Path to save the image
        """
        image_data = base64.b64decode(base64_data)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(output_path, image)
        return output_path


def demo_image_matching(client: SuperGlueClient, ref_image: str, query_image: str):
    """
    Demonstrate image-to-image matching
    
    Args:
        client: SuperGlue client instance
        ref_image: Path to reference image
        query_image: Path to query image
    """
    print("\\n=== Image-to-Image Matching Demo ===")
    
    # Check if images exist
    if not Path(ref_image).exists():
        print(f"Reference image not found: {ref_image}")
        return
    
    if not Path(query_image).exists():
        print(f"Query image not found: {query_image}")
        return
    
    # Set reference image
    print(f"Setting reference image: {ref_image}")
    result = client.set_reference_image(ref_image)
    if result["status"] != "success":
        print(f"Error setting reference image: {result['message']}")
        return
    
    print(f"Reference image set successfully. Keypoints: {result['keypoints_count']}")
    
    # Match with query image
    print(f"Matching with query image: {query_image}")
    result = client.match_with_image(query_image, visualize=True)
    if result["status"] != "success":
        print(f"Error in matching: {result['message']}")
        return
    
    print(f"Matching completed:")
    print(f"  - Reference keypoints: {result['num_keypoints_ref']}")
    print(f"  - Query keypoints: {result['num_keypoints_query']}")
    print(f"  - Matches found: {result['num_matches']}")
    print(f"  - Forward pass time: {result['timing']['forward_pass']:.4f}s")
    
    # Save visualization
    if "visualization_base64" in result:
        output_path = "image_matching_result.png"
        client.save_visualization_from_base64(result["visualization_base64"], output_path)
        print(f"  - Visualization saved to: {output_path}")


def demo_video_stream_matching(client: SuperGlueClient, ref_image: str, stream_source: str, max_frames: int = 10):
    """
    Demonstrate video stream matching
    
    Args:
        client: SuperGlue client instance
        ref_image: Path to reference image
        stream_source: Video stream source
        max_frames: Maximum number of frames to process
    """
    print("\\n=== Video Stream Matching Demo ===")
    
    # Check if reference image exists
    if not Path(ref_image).exists():
        print(f"Reference image not found: {ref_image}")
        return
    
    # Set reference image
    print(f"Setting reference image: {ref_image}")
    result = client.set_reference_image(ref_image)
    if result["status"] != "success":
        print(f"Error setting reference image: {result['message']}")
        return
    
    print(f"Reference image set successfully. Keypoints: {result['keypoints_count']}")
    
    # Start video stream
    print(f"Starting video stream: {stream_source}")
    result = client.start_video_stream(stream_source)
    if result["status"] != "success":
        print(f"Error starting video stream: {result['message']}")
        return
    
    print("Video stream started successfully")
    
    try:
        frame_count = 0
        output_dir = Path("video_matching_results")
        output_dir.mkdir(exist_ok=True)
        
        while frame_count < max_frames:
            # Get next frame match
            result = client.get_next_frame_match(visualize=True)
            
            if result["status"] == "end":
                print("Video stream ended")
                break
            elif result["status"] != "success":
                print(f"Error in frame matching: {result['message']}")
                break
            
            frame_count += 1
            print(f"Frame {result['frame_id']}: {result['num_matches']} matches found")
            
            # Save visualization
            if "visualization_base64" in result:
                output_path = output_dir / f"frame_{result['frame_id']:06d}_matches.png"
                client.save_visualization_from_base64(result["visualization_base64"], str(output_path))
                print(f"  - Saved: {output_path}")
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.1)
    
    finally:
        # Stop video stream
        print("Stopping video stream...")
        client.stop_video_stream()
        print("Video stream stopped")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='SuperGlue Backend Client Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '--backend_url', type=str, default='http://localhost:5000',
        help='URL of the SuperGlue backend API')
    parser.add_argument(
        '--mode', choices=['image', 'video', 'both'], default='both',
        help='Demo mode to run')
    parser.add_argument(
        '--ref_image', type=str, required=True,
        help='Path to reference image')
    parser.add_argument(
        '--query_image', type=str,
        help='Path to query image (for image mode)')
    parser.add_argument(
        '--stream_source', type=str,
        help='Video stream source: webcam ID (0,1,..), IP camera URL, or video file path')
    parser.add_argument(
        '--max_frames', type=int, default=10,
        help='Maximum frames to process in video mode')
    
    args = parser.parse_args()
    
    # Create client
    client = SuperGlueClient(args.backend_url)
    
    # Check backend status
    try:
        info = client.get_info()
        print(f"Connected to SuperGlue Backend")
        print(f"  - Device: {info.get('device', 'unknown')}")
        print(f"  - Model: {info.get('config', {}).get('superglue', {}).get('weights', 'unknown')}")
    except Exception as e:
        print(f"Error connecting to backend: {e}")
        print(f"Make sure the backend is running at {args.backend_url}")
        return
    
    # Run demos based on mode
    if args.mode in ['image', 'both']:
        if args.query_image:
            demo_image_matching(client, args.ref_image, args.query_image)
        else:
            print("Query image required for image matching demo")
    
    if args.mode in ['video', 'both']:
        if args.stream_source:
            demo_video_stream_matching(client, args.ref_image, args.stream_source, args.max_frames)
        else:
            print("Stream source required for video matching demo")


if __name__ == '__main__':
    main()
