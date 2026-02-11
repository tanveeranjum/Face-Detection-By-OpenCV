"""
Webcam Face Detection Processor
Real-time face detection from webcam
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple
from .face_detector import FaceDetector

class WebcamFaceDetector:
    """
    Real-time face detection from webcam
    """
    
    def __init__(self, camera_id=0, confidence=0.5):
        """
        Initialize webcam detector
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
            confidence: Detection confidence threshold
        """
        self.camera_id = camera_id
        self.confidence = confidence
        self.is_running = False
        self.cap = None
        
        # Initialize face detector
        self.detector = FaceDetector(confidence_threshold=confidence)
        
        # Statistics
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        self.total_faces = 0
        
        print(f"✅ Webcam detector initialized (Camera ID: {camera_id})")
    
    def start_camera(self):
        """Start webcam capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.cap.isOpened():
                print("❌ Could not open webcam")
                return False
            
            self.is_running = True
            print("✅ Webcam started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error starting webcam: {e}")
            return False
    
    def stop_camera(self):
        """Stop webcam capture"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.is_running = False
        print("✅ Webcam stopped")
    
    def get_frame(self):
        """Capture a frame from webcam"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            print("⚠️ Could not read frame from webcam")
            return None
        
        return frame
    
    def process_frame(self, frame):
        """
        Process a single frame for face detection
        
        Args:
            frame: Input frame from webcam
        
        Returns:
            processed_frame: Frame with face detection overlay
            faces: List of detected faces
        """
        if frame is None:
            return None, []
        
        # Detect faces
        processed_frame, faces = self.detector.detect_faces(frame)
        
        # Update statistics
        self.frame_count += 1
        self.total_faces += len(faces)
        
        # Calculate FPS
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        
        # Add overlay information
        if processed_frame is not None:
            self._add_overlay(processed_frame, faces)
        
        return processed_frame, faces
    
    def _add_overlay(self, frame, faces):
        """Add information overlay to frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)  # White
        thickness = 2
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 40), 
                   font, font_scale, color, thickness)
        
        # Face count
        cv2.putText(frame, f"Faces: {len(faces)}", (20, 70), 
                   font, font_scale, color, thickness)
        
        # Confidence threshold
        cv2.putText(frame, f"Confidence: {self.confidence}", (20, 100), 
                   font, font_scale, color, thickness)
        
        # Total faces detected
        cv2.putText(frame, f"Total: {self.total_faces}", (w - 150, 40), 
                   font, font_scale, color, thickness)
    
    def run_realtime(self):
        """
        Run real-time face detection in a window
        Press 'q' to quit, 's' to save snapshot
        """
        if not self.start_camera():
            return
        
        print("\n🎥 Starting real-time face detection...")
        print("   Press 'q' to quit")
        print("   Press 's' to save snapshot")
        print("   Press '+' to increase confidence")
        print("   Press '-' to decrease confidence")
        
        while self.is_running:
            # Get frame
            frame = self.get_frame()
            if frame is None:
                break
            
            # Process frame
            processed_frame, faces = self.process_frame(frame)
            
            if processed_frame is not None:
                # Display frame
                cv2.imshow('Face Detection - Webcam', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):  # Quit
                    break
                elif key == ord('s'):  # Save snapshot
                    self._save_snapshot(processed_frame)
                elif key == ord('+'):  # Increase confidence
                    self.confidence = min(0.9, self.confidence + 0.05)
                    self.detector.update_confidence_threshold(self.confidence)
                    print(f"📈 Confidence increased to {self.confidence}")
                elif key == ord('-'):  # Decrease confidence
                    self.confidence = max(0.1, self.confidence - 0.05)
                    self.detector.update_confidence_threshold(self.confidence)
                    print(f"📉 Confidence decreased to {self.confidence}")
        
        # Cleanup
        self.stop_camera()
        cv2.destroyAllWindows()
        print("\n✅ Face detection session ended")
        print(f"   Total frames processed: {self.frame_count}")
        print(f"   Total faces detected: {self.total_faces}")
    
    def _save_snapshot(self, frame):
        """Save current frame as image"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"data/output/snapshot_{timestamp}.jpg"
        
        cv2.imwrite(filename, frame)
        print(f"📸 Snapshot saved: {filename}")
        return filename
    
    def get_statistics(self):
        """Get current statistics"""
        return {
            'fps': self.fps,
            'total_faces': self.total_faces,
            'confidence': self.confidence,
            'is_running': self.is_running
        }

def run_webcam_detection(camera_id=0, confidence=0.5):
    """
    Simple function to run webcam face detection
    
    Args:
        camera_id: Camera device ID
        confidence: Detection confidence threshold
    """
    detector = WebcamFaceDetector(camera_id=camera_id, confidence=confidence)
    detector.run_realtime()

if __name__ == "__main__":
    # Run webcam detection
    import argparse
    
    parser = argparse.ArgumentParser(description='Webcam Face Detection')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    print("🚀 Starting Webcam Face Detection...")
    run_webcam_detection(camera_id=args.camera, confidence=args.confidence)