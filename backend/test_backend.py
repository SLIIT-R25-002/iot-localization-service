#!/usr/bin/env python3
"""
Test script for SuperGlue Backend
Verifies that all components are working correctly.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name()}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    return True

def test_superglue_models():
    """Test if SuperGlue models can be imported"""
    print("\\nTesting SuperGlue models...")
    
    try:
        from models.matching import Matching
        from models.superpoint import SuperPoint
        from models.superglue import SuperGlue
        from models.utils import VideoStreamer, frame2tensor
        print("✓ SuperGlue models imported successfully")
        return True
    except ImportError as e:
        print(f"✗ SuperGlue models import failed: {e}")
        print("Make sure you're running from the SuperGlue directory")
        return False

def test_model_weights():
    """Test if model weights are available"""
    print("\\nTesting model weights...")
    
    weights_dir = Path("models/weights")
    if not weights_dir.exists():
        print(f"✗ Weights directory not found: {weights_dir}")
        return False
    
    indoor_weights = weights_dir / "superglue_indoor.pth"
    outdoor_weights = weights_dir / "superglue_outdoor.pth"
    
    if indoor_weights.exists():
        print("✓ Indoor weights found")
    else:
        print(f"✗ Indoor weights not found: {indoor_weights}")
    
    if outdoor_weights.exists():
        print("✓ Outdoor weights found")
    else:
        print(f"✗ Outdoor weights not found: {outdoor_weights}")
    
    return indoor_weights.exists() or outdoor_weights.exists()

def test_standalone_backend():
    """Test the standalone backend"""
    print("\\nTesting standalone backend...")
    
    try:
        from superglue_standalone import SuperGlueStandalone
        
        # Create instance
        sg = SuperGlueStandalone()
        print("✓ SuperGlueStandalone created successfully")
        
        # Test with sample images if available
        sample_images = list(Path("assets").glob("**/*.jpg")) + list(Path("assets").glob("**/*.png"))
        if len(sample_images) >= 2:
            ref_img = str(sample_images[0])
            query_img = str(sample_images[1])
            
            print(f"Testing with sample images:")
            print(f"  Reference: {ref_img}")
            print(f"  Query: {query_img}")
            
            # Set reference
            result = sg.set_reference_image(ref_img)
            if result["status"] == "success":
                print(f"✓ Reference image set ({result['keypoints_count']} keypoints)")
                
                # Test matching
                result = sg.match_with_image(query_img, save_visualization=False)
                if result["status"] == "success":
                    print(f"✓ Image matching successful ({result['num_matches']} matches)")
                else:
                    print(f"✗ Image matching failed: {result['message']}")
            else:
                print(f"✗ Failed to set reference: {result['message']}")
        else:
            print("⚠ No sample images found for testing")
        
        return True
        
    except Exception as e:
        print(f"✗ Standalone backend test failed: {e}")
        traceback.print_exc()
        return False

def test_flask_backend():
    """Test if Flask backend can be imported"""
    print("\\nTesting Flask backend...")
    
    try:
        import flask
        import flask_cors
        print(f"✓ Flask {flask.__version__} available")
        
        # Try importing the backend (but don't run it)
        try:
            import superglue_backend
            print("✓ Flask backend imports successfully")
            return True
        except ImportError as e:
            print(f"✗ Flask backend import failed: {e}")
            return False
            
    except ImportError as e:
        print(f"⚠ Flask not available: {e}")
        print("  Install with: pip install flask flask-cors")
        return False

def test_webcam():
    """Test webcam access"""
    print("\\nTesting webcam access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✓ Webcam accessible (frame shape: {frame.shape})")
                result = True
            else:
                print("✗ Webcam opened but no frame received")
                result = False
            cap.release()
        else:
            print("✗ Cannot open webcam")
            result = False
        return result
    except Exception as e:
        print(f"✗ Webcam test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("SuperGlue Backend Test Suite")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("SuperGlue Models", test_superglue_models),
        ("Model Weights", test_model_weights),
        ("Standalone Backend", test_standalone_backend),
        ("Flask Backend", test_flask_backend),
        ("Webcam Access", test_webcam),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\\n{test_name}")
        print("-" * len(test_name))
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\\n" + "=" * 40)
    print("Test Summary:")
    print("=" * 40)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\\n🎉 All tests passed! Your SuperGlue backend is ready to use.")
    else:
        print("\\n⚠ Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
