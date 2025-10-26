import cv2
import numpy as np

def detect_circle_in_crop(image):
    """
    Detects if a circle is present in the cropped image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=100
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        return True, (x, y, r)
    
    return False, None


def detect_squares(image):
    """
    Detects the concentric squares in the image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    squares = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4 and cv2.contourArea(contour) > 100:
            squares.append(approx)
    
    return squares


def navigate_to_circle(reference_image_path, cropped_image_path, step_size=50):
    """
    Determines camera movements needed to bring the circle into view
    """
    ref_img = cv2.imread(reference_image_path)
    crop_img = cv2.imread(cropped_image_path)
    
    if ref_img is None or crop_img is None:
        raise ValueError("Could not load images")
    
    # Check if circle already visible in cropped
    circle_visible, circle_info = detect_circle_in_crop(crop_img)
    if circle_visible:
        print("Circle is already visible in the cropped image!")
        print(f"Circle location: {circle_info}")
        return []
    
    # Find circle in reference
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    ref_blurred = cv2.GaussianBlur(ref_gray, (9, 9), 2)
    ref_circles = cv2.HoughCircles(
        ref_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=100
    )
    
    if ref_circles is None:
        print("Could not find circle in reference image")
        return None
    
    ref_circles = np.uint16(np.around(ref_circles))
    circle_x, circle_y, circle_r = ref_circles[0][0]
    print(f"Circle in reference image at: ({circle_x}, {circle_y})")
    
    # Template matching to locate crop in reference
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(ref_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    
    crop_x, crop_y = max_loc
    crop_h, crop_w = crop_img.shape[:2]
    print(f"Cropped region in reference: top-left ({crop_x}, {crop_y})")
    print(f"Crop dimensions: {crop_w}x{crop_h}")
    
    # Camera movement calculation
    movements = []
    current_x, current_y = crop_x, crop_y
    target_x, target_y = circle_x, circle_y
    
    while True:
        if (current_x <= target_x <= current_x + crop_w and
            current_y <= target_y <= current_y + crop_h):
            print("Circle is now in view!")
            break
        
        dx = target_x - (current_x + crop_w // 2)
        dy = target_y - (current_y + crop_h // 2)
        
        if abs(dx) > abs(dy):
            if dx > 0:
                move_x = min(step_size, abs(dx))
                movements.append(f"Move RIGHT {move_x} pixels")
                current_x += move_x
            else:
                move_x = min(step_size, abs(dx))
                movements.append(f"Move LEFT {move_x} pixels")
                current_x -= move_x
        else:
            if dy > 0:
                move_y = min(step_size, abs(dy))
                movements.append(f"Move DOWN {move_y} pixels")
                current_y += move_y
            else:
                move_y = min(step_size, abs(dy))
                movements.append(f"Move UP {move_y} pixels")
                current_y -= move_y
        
        if len(movements) > 100:
            print("Too many movements, stopping")
            break
    
    print("\nCamera movements needed:")
    for i, movement in enumerate(movements, 1):
        print(f"Step {i}: {movement}")
    
    return movements


def main():
    print("Camera Navigation to Circle")
    print("=" * 50)
    
    # Your actual file paths
    reference_image = r"C:\Users\satya\Downloads\computer-vision-challenge-satyamKr-main\satellite-port-cv\data\WhatsApp Image 2025-09-28 at 09.28.13_c24f16cc.jpg"
    cropped_image = r"C:\Users\satya\Downloads\computer-vision-challenge-satyamKr-main\satellite-port-cv\data\cropped_port.png"
    
    # Run navigation
    movements = navigate_to_circle(reference_image, cropped_image, step_size=50)
    print(movements)


if __name__ == "__main__":
    main()
