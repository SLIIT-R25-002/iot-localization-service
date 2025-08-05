#!/usr/bin/env python3
"""
SuperGlue Standalone Backend
A simplified Python backend that uses SuperGlue for feature matching without Flask dependency.
Can be used directly in Python scripts or as a module.

Author: Backend Implementation
Based on: SuperGlue by Magic Leap (CVPR 2020)
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

import cv2
import numpy as np
import torch
import matplotlib.cm as cm

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer, make_matching_plot_fast, 
                         frame2tensor, read_image, process_resize)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable gradients for inference
torch.set_grad_enabled(False)


class SuperGlueStandalone:
    """Standalone SuperGlue backend for image and video stream matching"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the SuperGlue backend
        
        Args:
            config: Configuration dictionary for SuperPoint and SuperGlue
        """
        if config is None:
            config = self._get_default_config()
            
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
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            },
            'force_cpu': False
        }
        
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
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
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
            self.reference_path = str(image_path)
            
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
                        save_visualization: bool = False, output_path: Optional[str] = None) -> Dict:
        """
        Match reference image with a query image
        
        Args:
            query_image_path: Path to the query image
            resize: Resize dimensions
            save_visualization: Whether to save visualization
            output_path: Path to save visualization (optional)
            
        Returns:
            Dictionary with matching results
        """
        if self.reference_data is None:
            return {"status": "error", "message": "No reference image set"}
        
        try:
            # Read and process query image
            query_image = cv2.imread(str(query_image_path), cv2.IMREAD_GRAYSCALE)
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
                    "forward_pass": self.timer.times.get('forward', 0)
                }
            }
            
            # Create visualization if requested
            if save_visualization:
                if output_path is None:
                    output_path = f"match_result_{int(time.time())}.png"
                
                color = cm.jet(match_confidence)
                text = [
                    'SuperGlue',
                    f'Keypoints: {len(kpts0)}:{len(kpts1)}',
                    f'Matches: {len(mkpts0)}'
                ]
                
                viz_image = make_matching_plot_fast(
                    self.reference_image, query_image, kpts0, kpts1, mkpts0, mkpts1, 
                    color, text, path=None, show_keypoints=True
                )
                
                cv2.imwrite(output_path, viz_image)
                result["visualization_path"] = output_path
                logger.info(f"Visualization saved to: {output_path}")
            
            self.timer.update('viz')
            return result
            
        except Exception as e:
            logger.error(f"Error in image matching: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def start_video_stream_matching(self, stream_source: Union[str, int], resize: List[int] = [640, 480]) -> Dict:
        """
        Start matching with a video stream
        
        Args:
            stream_source: Video source (IP camera URL, video file, or webcam ID as int)
            resize: Resize dimensions
            
        Returns:
            Dictionary with status
        """
        if self.reference_data is None:
            return {"status": "error", "message": "No reference image set"}
        
        try:
            # Initialize video streamer
            self.video_streamer = VideoStreamer(str(stream_source), resize, skip=1, 
                                              image_glob=['*.png', '*.jpg', '*.jpeg'])
            self.streaming_active = True
            
            return {
                "status": "success",
                "message": "Video stream started successfully",
                "source": str(stream_source)
            }
            
        except Exception as e:
            logger.error(f"Error starting video stream: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_next_frame_match(self, save_visualization: bool = False, output_dir: str = "output") -> Dict:
        """
        Get matching results for the next frame in the video stream
        
        Args:
            save_visualization: Whether to save visualization
            output_dir: Directory to save visualizations
            
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
            
            frame_id = self.video_streamer.i - 1
            
            result = {
                "status": "success",
                "frame_id": frame_id,
                "num_keypoints_ref": len(kpts0),
                "num_keypoints_query": len(kpts1),
                "num_matches": len(mkpts0),
                "matches": {
                    "keypoints_ref": mkpts0.tolist(),
                    "keypoints_query": mkpts1.tolist(),
                    "confidence": match_confidence.tolist()
                },
                "timing": {
                    "forward_pass": self.timer.times.get('forward', 0)
                }
            }
            
            # Create visualization if requested
            if save_visualization:
                Path(output_dir).mkdir(exist_ok=True)
                output_path = Path(output_dir) / f"frame_{frame_id:06d}_matches.png"
                
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
                
                cv2.imwrite(str(output_path), viz_image)
                result["visualization_path"] = str(output_path)
            
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
    
    def process_video_batch(self, stream_source: Union[str, int], max_frames: int = 100, 
                           save_visualizations: bool = True, output_dir: str = "video_output") -> List[Dict]:
        """
        Process a batch of video frames and return all results
        
        Args:
            stream_source: Video source
            max_frames: Maximum number of frames to process
            save_visualizations: Whether to save visualizations
            output_dir: Output directory for visualizations
            
        Returns:
            List of matching results for each frame
        """
        if self.reference_data is None:
            return [{"status": "error", "message": "No reference image set"}]
        
        # Start video stream
        start_result = self.start_video_stream_matching(stream_source)
        if start_result["status"] != "success":
            return [start_result]
        
        results = []
        try:
            for i in range(max_frames):
                result = self.get_next_frame_match(save_visualizations, output_dir)
                results.append(result)
                
                if result["status"] == "end":
                    logger.info(f"Video ended after {i+1} frames")
                    break
                elif result["status"] != "success":
                    logger.warning(f"Error in frame {i}: {result.get('message', 'Unknown error')}")
                    break
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1} frames")
        
        finally:
            self.stop_video_stream()
        
        return results


def demo_image_matching():
    """Demo function for image-to-image matching"""
    # Initialize SuperGlue
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'indoor',  # or 'outdoor'
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    
    sg = SuperGlueStandalone(config)
    
    # Example paths - replace with your actual image paths
    ref_image = "assets/scannet_sample_images/scene0711_00_frame-001680.jpg"
    query_image = "assets/scannet_sample_images/scene0711_00_frame-001995.jpg"
    
    # Check if example images exist
    if not Path(ref_image).exists() or not Path(query_image).exists():
        print("Example images not found. Please provide valid image paths.")
        return
    
    # Set reference image
    print(f"Setting reference image: {ref_image}")
    result = sg.set_reference_image(ref_image)
    print(f"Reference set: {result}")
    
    # Match with query image
    print(f"\\nMatching with query image: {query_image}")
    result = sg.match_with_image(query_image, save_visualization=True, 
                                output_path="demo_image_match.png")
    print(f"Match result: {result}")


def demo_video_matching():
    """Demo function for video stream matching"""
    # Initialize SuperGlue
    sg = SuperGlueStandalone()
    
    # Example paths - replace with your actual paths
    ref_image = "assets/scannet_sample_images/scene0711_00_frame-001680.jpg"
    video_source = 0  # Webcam, or use a video file path
    
    if not Path(ref_image).exists():
        print("Reference image not found. Please provide a valid image path.")
        return
    
    # Set reference image
    print(f"Setting reference image: {ref_image}")
    result = sg.set_reference_image(ref_image)
    print(f"Reference set: {result}")
    
    # Process video frames
    print(f"\\nProcessing video from: {video_source}")
    results = sg.process_video_batch(video_source, max_frames=20, 
                                   save_visualizations=True, 
                                   output_dir="demo_video_output")
    
    # Print summary
    successful_frames = [r for r in results if r.get("status") == "success"]
    total_matches = sum(r.get("num_matches", 0) for r in successful_frames)
    print(f"\\nProcessed {len(successful_frames)} frames successfully")
    print(f"Total matches found: {total_matches}")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description='SuperGlue Standalone Backend',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        'mode', choices=['image', 'video'], 
        help='Mode: image matching or video stream matching')
    parser.add_argument(
        '--ref_image', type=str, required=True,
        help='Path to reference image')
    parser.add_argument(
        '--query_image', type=str,
        help='Path to query image (for image mode)')
    parser.add_argument(
        '--stream_source', type=str,
        help='Video stream source (for video mode)')
    parser.add_argument(
        '--max_frames', type=int, default=50,
        help='Maximum frames to process (for video mode)')
    parser.add_argument(
        '--output_dir', type=str, default='output',
        help='Output directory for visualizations')
    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force CPU mode')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': args.superglue,
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        },
        'force_cpu': args.force_cpu
    }
    
    # Initialize SuperGlue
    sg = SuperGlueStandalone(config)
    
    # Set reference image
    print(f"Setting reference image: {args.ref_image}")
    result = sg.set_reference_image(args.ref_image)
    if result["status"] != "success":
        print(f"Error: {result['message']}")
        return
    print(f"Reference image set successfully. Keypoints: {result['keypoints_count']}")
    
    if args.mode == 'image':
        if not args.query_image:
            print("Error: --query_image required for image mode")
            return
        
        print(f"\\nMatching with query image: {args.query_image}")
        result = sg.match_with_image(args.query_image, save_visualization=True,
                                   output_path=f"{args.output_dir}/image_match.png")
        if result["status"] == "success":
            print(f"Matches found: {result['num_matches']}")
            print(f"Visualization saved to: {result.get('visualization_path', 'N/A')}")
        else:
            print(f"Error: {result['message']}")
    
    elif args.mode == 'video':
        if not args.stream_source:
            print("Error: --stream_source required for video mode")
            return
        
        print(f"\\nProcessing video stream: {args.stream_source}")
        results = sg.process_video_batch(args.stream_source, args.max_frames, 
                                       save_visualizations=True, output_dir=args.output_dir)
        
        successful_frames = [r for r in results if r.get("status") == "success"]
        print(f"Processed {len(successful_frames)} frames successfully")
        if successful_frames:
            avg_matches = np.mean([r["num_matches"] for r in successful_frames])
            print(f"Average matches per frame: {avg_matches:.1f}")


if __name__ == '__main__':
    main()
