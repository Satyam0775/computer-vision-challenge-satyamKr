import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_rotation_angle(image_path, rotated_image_path=None, rotation_angle=None):
    """
    Detects the rotation angle of a satellite port image by finding the circular marker.
    
    Args:
        image_path: Path to the reference image
        rotated_image_path: Path to save rotated image (optional)
        rotation_angle: Angle to rotate the image for testing (optional)
    
    Returns:
        detected_angle: The detected rotation angle in degrees
    """
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Create outputs directory
    outputs_dir = r"C:\Users\satya\Downloads\computer-vision-challenge-satyamKr-main\satellite-port-cv\outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    
    # If rotation angle is provided, rotate the image for testing
    if rotation_angle is not None:
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))
        
        if rotated_image_path:
            rotated_path = os.path.join(outputs_dir, os.path.basename(rotated_image_path))
            cv2.imwrite(rotated_path, img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # 🔑 Try circle detection with tuned parameters
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=100,
        param2=20,
        minRadius=5,
        maxRadius=150
    )
    
    if circles is None:
        print("✗ No circles detected - try adjusting HoughCircles params or check image quality")
        return None
    
    circles = np.uint16(np.around(circles))
    
    if len(circles[0]) > 0:
        x, y, r = circles[0][0]
        
        # Find the center of the image
        h, w = gray.shape
        center_x, center_y = w // 2, h // 2
        
        dx = x - center_x
        dy = y - center_y
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        
        reference_angle = 225
        detected_rotation = (angle_deg - reference_angle) % 360
        if detected_rotation > 180:
            detected_rotation -= 360
        
        # Draw visualization
        img_display = img.copy()
        cv2.circle(img_display, (x, y), r, (0, 255, 0), 2)
        cv2.circle(img_display, (x, y), 2, (0, 0, 255), 3)
        cv2.circle(img_display, (center_x, center_y), 5, (255, 0, 0), -1)
        cv2.line(img_display, (center_x, center_y), (x, y), (255, 0, 0), 2)
        
        # Save annotated image into outputs folder
        output_path = os.path.join(outputs_dir, "rotation_detection_output.png")
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Rotation: {detected_rotation:.2f}°')
        plt.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Circle detected at: ({x}, {y}) with radius {r}")
        print(f"✓ Angle from center: {angle_deg:.2f}°")
        print(f"✓ Detected rotation: {detected_rotation:.2f}°")
        print(f"✓ Saved visualization to: {output_path}")
        
        return detected_rotation
    
    return None


def main():
    print("Testing rotation detection on WhatsApp image...")
    
    image_path = r"C:\Users\satya\Downloads\computer-vision-challenge-satyamKr-main\satellite-port-cv\data\WhatsApp Image 2025-09-28 at 09.28.13_c24f16cc.jpg"
    
    angle = detect_rotation_angle(image_path)
    print(f"Final Detected Angle: {angle}")


if __name__ == "__main__":
    main()
