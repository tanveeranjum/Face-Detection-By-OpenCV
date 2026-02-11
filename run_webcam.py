#!/usr/bin/env python3
"""
Main launcher for Face Detection System
Fixed version that works with your structure
"""

import sys
import os

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import cv2
import numpy as np
import time
from pathlib import Path

def check_opencv():
    """Check if OpenCV is installed"""
    try:
        print(f"✅ OpenCV version: {cv2.__version__}")
        return True
    except AttributeError:
        print(f"✅ OpenCV is installed")
        return True
    except ImportError:
        print("❌ OpenCV is not installed!")
        print("Please run: pip install opencv-python")
        return False

def download_models():
    """Download face detection models if not present"""
    import urllib.request
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_urls = {
        "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }
    
    all_files_exist = True
    
    for filename, url in model_urls.items():
        filepath = models_dir / filename
        
        if not filepath.exists():
            all_files_exist = False
            print(f"📥 Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"  ✅ Downloaded")
            except Exception as e:
                print(f"  ❌ Failed to download {filename}: {e}")
                return False
    
    if all_files_exist:
        print("✅ All model files are present")
    
    return True

class SimpleFaceDetector:
    """Simple face detector for webcam"""
    
    def __init__(self, confidence=0.5):
        self.confidence = confidence
        self.net = None
        self.load_model()
    
    def load_model(self):
        """Load the face detection model"""
        config_path = "models/deploy.prototxt"
        weights_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
        
        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            print("❌ Model files not found!")
            if download_models():
                # Try again after download
                pass
            else:
                return False
        
        try:
            self.net = cv2.dnn.readNetFromCaffe(config_path, weights_path)
            print("✅ Face detection model loaded")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def detect_faces(self, frame):
        """Detect faces in a frame"""
        if frame is None or self.net is None:
            return frame, []
        
        (h, w) = frame.shape[:2]
        
        # Create blob for face detection
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False,
            crop=False
        )
        
        # Detect faces
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        
        # Process detections
        for i in range(detections.shape[2]):
            confidence_score = detections[0, 0, i, 2]
            
            if confidence_score > self.confidence:
                # Get bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Ensure within bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w-1, x2), min(h-1, y2)
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw confidence
                label = f"{confidence_score*100:.1f}%"
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                faces.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(confidence_score)
                })
        
        return frame, faces

def run_webcam(camera_id=0, confidence=0.5):
    """Run webcam face detection"""
    
    print("="*60)
    print("🎭 WEB CAM FACE DETECTION")
    print("="*60)
    
    # Check OpenCV
    if not check_opencv():
        return
    
    # Create detector
    detector = SimpleFaceDetector(confidence)
    
    if detector.net is None:
        print("❌ Could not initialize face detector")
        return
    
    # Open webcam
    print(f"\n📷 Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Could not open camera {camera_id}")
        print("Trying camera 1...")
        cap = cv2.VideoCapture(1)
        
        if not cap.isOpened():
            print("❌ No camera available!")
            print("Please check:")
            print("1. Webcam is connected")
            print("2. No other app is using the webcam")
            return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Camera opened successfully!")
    
    # Create output directory
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🎮 CONTROLS:")
    print("• Press 'Q' to quit")
    print("• Press 'S' to save snapshot")
    print("• Press '+' to increase confidence")
    print("• Press '-' to decrease confidence")
    print("\n👀 Looking for faces...")
    
    # FPS calculation
    fps_start = time.time()
    frame_count = 0
    fps = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Could not read frame")
                break
            
            # Detect faces
            processed_frame, faces = detector.detect_faces(frame)
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - fps_start
            
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()
            
            # Add overlay info
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Faces: {len(faces)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Conf: {detector.confidence:.2f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, "Press 'Q' to quit", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Face Detection - Press Q to quit', processed_frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\n⏹️ Stopping...")
                break
            elif key == ord('s') or key == ord('S'):
                # Save snapshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = output_dir / f"snapshot_{timestamp}.jpg"
                cv2.imwrite(str(filename), processed_frame)
                print(f"📸 Saved: {filename}")
            elif key == ord('+'):
                detector.confidence = min(0.9, detector.confidence + 0.05)
                print(f"📈 Confidence: {detector.confidence:.2f}")
            elif key == ord('-'):
                detector.confidence = max(0.1, detector.confidence - 0.05)
                print(f"📉 Confidence: {detector.confidence:.2f}")
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopped by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Face detection ended")
        print(f"💾 Snapshots saved in: {output_dir}/")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Detection System')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID (0 for default)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (0.1-0.9)')
    parser.add_argument('--list-cameras', action='store_true',
                       help='List available cameras')
    
    args = parser.parse_args()
    
    if args.list_cameras:
        # List available cameras
        print("📷 Checking available cameras...")
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"  Camera {i}: Available")
                cap.release()
            else:
                print(f"  Camera {i}: Not available")
        return
    
    # Run webcam detection
    run_webcam(camera_id=args.camera, confidence=args.confidence)

if __name__ == "__main__":
    main()