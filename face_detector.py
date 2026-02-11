"""
Face Detector Module
Core face detection using OpenCV DNN
"""

import cv2
import numpy as np
from pathlib import Path

class FaceDetector:
    """
    Face detector using OpenCV's DNN module
    """
    
    def __init__(self, model_type="caffe", confidence_threshold=0.5):
        """
        Initialize face detector
        
        Args:
            model_type: "caffe" or "tensorflow" (caffe recommended)
            confidence_threshold: Minimum confidence for detection (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self.model_type = model_type
        
        # Load model
        self.net = self._load_model()
        
        # Detection parameters
        self.input_size = (300, 300)
        self.mean_values = (104.0, 177.0, 123.0)  # BGR mean subtraction
        self.scale_factor = 1.0
        
        print(f"✅ Face detector initialized (model: {model_type})")
    
    def _load_model(self):
        """Load the face detection model"""
        models_dir = Path("models")
        
        if self.model_type == "caffe":
            config_path = models_dir / "deploy.prototxt"
            weights_path = models_dir / "res10_300x300_ssd_iter_140000.caffemodel"
            
            if not config_path.exists() or not weights_path.exists():
                raise FileNotFoundError(
                    "Caffe model files not found. Run setup.py first."
                )
            
            return cv2.dnn.readNetFromCaffe(str(config_path), str(weights_path))
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def detect_faces(self, frame, draw_boxes=True):
        """
        Detect faces in a frame
        
        Args:
            frame: Input image/frame (numpy array)
            draw_boxes: Whether to draw bounding boxes on the frame
        
        Returns:
            processed_frame: Frame with bounding boxes (if draw_boxes=True)
            faces: List of detected faces with details
        """
        if frame is None:
            return None, []
        
        # Get frame dimensions
        (h, w) = frame.shape[:2]
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(
            frame,
            self.scale_factor,
            self.input_size,
            self.mean_values,
            swapRB=False,  # OpenCV uses BGR by default
            crop=False
        )
        
        # Pass blob through network
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # Process detections
        faces = []
        processed_frame = frame.copy() if draw_boxes else None
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter out weak detections
            if confidence > self.confidence_threshold:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure bounding boxes are within frame dimensions
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w - 1, endX)
                endY = min(h - 1, endY)
                
                # Calculate face area
                face_width = endX - startX
                face_height = endY - startY
                face_area = face_width * face_height
                
                # Store face information
                face_info = {
                    'bbox': (startX, startY, endX, endY),
                    'confidence': float(confidence),
                    'area': face_area,
                    'center': ((startX + endX) // 2, (startY + endY) // 2),
                    'dimensions': (face_width, face_height)
                }
                faces.append(face_info)
                
                # Draw bounding box if requested
                if draw_boxes:
                    self._draw_face_box(processed_frame, face_info)
        
        return processed_frame, faces
    
    def _draw_face_box(self, frame, face_info):
        """Draw bounding box and label for a face"""
        startX, startY, endX, endY = face_info['bbox']
        confidence = face_info['confidence']
        
        # Draw rectangle
        color = (0, 255, 0)  # Green
        thickness = 2
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, thickness)
        
        # Draw confidence label
        label = f"{confidence * 100:.1f}%"
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # Draw background for text
        cv2.rectangle(
            frame,
            (startX, startY - text_height - 8),
            (startX + text_width, startY),
            color,
            -1  # Filled rectangle
        )
        
        # Draw text
        cv2.putText(
            frame,
            label,
            (startX, startY - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),  # Black text
            2
        )
    
    def update_confidence_threshold(self, new_threshold):
        """Update the confidence threshold"""
        if 0 <= new_threshold <= 1:
            self.confidence_threshold = new_threshold
            print(f"✅ Confidence threshold updated to {new_threshold}")
        else:
            print("⚠️ Confidence threshold must be between 0 and 1")
    
    def get_statistics(self, faces):
        """Get statistics about detected faces"""
        if not faces:
            return {
                'count': 0,
                'avg_confidence': 0,
                'avg_area': 0
            }
        
        confidences = [face['confidence'] for face in faces]
        areas = [face['area'] for face in faces]
        
        return {
            'count': len(faces),
            'avg_confidence': np.mean(confidences),
            'avg_area': np.mean(areas),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'min_area': np.min(areas),
            'max_area': np.max(areas)
        }

# Simple function for quick usage
def detect_faces_simple(image_path, confidence=0.5, output_path=None):
    """
    Simple function to detect faces in an image
    
    Args:
        image_path: Path to input image
        confidence: Detection confidence threshold
        output_path: Optional path to save result
    
    Returns:
        faces: List of detected faces
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return []
    
    # Create detector
    detector = FaceDetector(confidence_threshold=confidence)
    
    # Detect faces
    result, faces = detector.detect_faces(image)
    
    # Save result if requested
    if output_path and result is not None:
        cv2.imwrite(output_path, result)
        print(f"✅ Result saved to: {output_path}")
    
    # Print statistics
    stats = detector.get_statistics(faces)
    print(f"📊 Detected {stats['count']} face(s)")
    print(f"   Average confidence: {stats['avg_confidence']:.2%}")
    
    return faces

if __name__ == "__main__":
    # Test the detector
    import sys
    
    if len(sys.argv) > 1:
        # Test with provided image
        image_path = sys.argv[1]
        detect_faces_simple(image_path, output_path="test_output.jpg")
    else:
        print("Usage: python face_detector.py <image_path>")
        print("Example: python face_detector.py test_image.jpg")