"""
Utility functions for face detection system
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def convert_image_format(image_input, target_format="BGR"):
    """
    Convert image between different formats
    
    Args:
        image_input: Input image (file path, PIL Image, or numpy array)
        target_format: "BGR", "RGB", or "GRAY"
    
    Returns:
        image_np: Converted numpy array
    """
    # Load image if path is provided
    if isinstance(image_input, (str, Path)):
        image = cv2.imread(str(image_input))
        if image is None:
            raise ValueError(f"Could not load image: {image_input}")
    elif isinstance(image_input, Image.Image):
        # Convert PIL to numpy
        image = np.array(image_input)
        # PIL uses RGB, OpenCV uses BGR
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        image = image_input.copy()
    else:
        raise TypeError("Unsupported image input type")
    
    # Convert to target format
    if target_format == "BGR":
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif target_format == "RGB":
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:  # BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:  # RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif target_format == "GRAY":
        if len(image.shape) == 3:  # Color to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image

def resize_image(image, max_width=800):
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image (numpy array)
        max_width: Maximum width for resizing
    
    Returns:
        resized_image: Resized image
    """
    h, w = image.shape[:2]
    
    if w <= max_width:
        return image
    
    # Calculate new dimensions
    ratio = max_width / w
    new_height = int(h * ratio)
    
    return cv2.resize(image, (max_width, new_height))

def create_test_image(num_faces=3, image_size=(600, 400)):
    """
    Create a test image with synthetic faces
    
    Args:
        num_faces: Number of faces to create
        image_size: (width, height) of image
    
    Returns:
        test_image: Generated test image
    """
    width, height = image_size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add gradient background
    for i in range(height):
        color = int(200 * i / height)
        image[i, :] = [color, color//2, 255-color//2]
    
    # Generate random face positions
    np.random.seed(42)  # For reproducibility
    positions = []
    
    for _ in range(num_faces):
        x = np.random.randint(100, width - 100)
        y = np.random.randint(100, height - 100)
        positions.append((x, y))
    
    # Draw faces
    colors = [
        (255, 200, 150),  # Light skin
        (200, 255, 200),  # Light green
        (200, 200, 255),  # Light blue
        (255, 255, 200),  # Light yellow
        (255, 200, 255),  # Light purple
    ]
    
    for i, (x, y) in enumerate(positions):
        color = colors[i % len(colors)]
        size = np.random.randint(40, 80)
        
        # Draw face circle
        cv2.circle(image, (x, y), size, color, -1)
        
        # Draw eyes
        eye_size = size // 8
        cv2.circle(image, (x - size//3, y - size//4), eye_size, (0, 0, 0), -1)
        cv2.circle(image, (x + size//3, y - size//4), eye_size, (0, 0, 0), -1)
        
        # Draw mouth
        mouth_width = size // 3
        mouth_height = size // 6
        cv2.ellipse(image, (x, y + size//4), (mouth_width, mouth_height), 
                   0, 0, 180, (0, 0, 0), 2)
    
    return image

def display_image(image, title="Image", figsize=(10, 6)):
    """
    Display image using matplotlib
    
    Args:
        image: Input image (numpy array in BGR format)
        title: Plot title
        figsize: Figure size
    """
    # Convert BGR to RGB for matplotlib
    if len(image.shape) == 3:
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_image = image
    
    plt.figure(figsize=figsize)
    plt.imshow(display_image)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_results(image, faces, output_path="results.jpg"):
    """
    Save detection results with annotations
    
    Args:
        image: Original image
        faces: List of detected faces
        output_path: Path to save results
    
    Returns:
        output_path: Path where results were saved
    """
    # Draw boxes on image
    result_image = image.copy()
    
    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        confidence = face['confidence']
        
        # Draw rectangle
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw confidence
        label = f"{confidence:.1%}"
        cv2.putText(result_image, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Add summary text
    summary = f"Faces: {len(faces)}"
    cv2.putText(result_image, summary, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save image
    cv2.imwrite(output_path, result_image)
    print(f"✅ Results saved to: {output_path}")
    
    return output_path

def check_camera_available(camera_id=0):
    """
    Check if camera is available
    
    Args:
        camera_id: Camera device ID
    
    Returns:
        available: True if camera is available
        message: Status message
    """
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        cap.release()
        return False, f"Camera {camera_id} is not available"
    
    # Test capturing a frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return False, f"Camera {camera_id} cannot capture frames"
    
    return True, f"Camera {camera_id} is ready (Resolution: {frame.shape[1]}x{frame.shape[0]})"

if __name__ == "__main__":
    # Test utility functions
    print("🧪 Testing utility functions...")
    
    # Create test image
    test_image = create_test_image(num_faces=3)
    print(f"✅ Created test image: {test_image.shape}")
    
    # Save test image
    cv2.imwrite("data/test_images/test_faces.jpg", test_image)
    print("✅ Test image saved")
    
    # Check camera
    available, message = check_camera_available()
    print(f"📷 Camera status: {message}")
    
    # Display test image
    display_image(test_image, title="Test Image with Synthetic Faces")