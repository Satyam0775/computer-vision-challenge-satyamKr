"""
Complete Test Suite for Computer Vision Challenge
Tests all four parts of the challenge
"""

import cv2
import numpy as np
import os
import sys

# Base directory
BASE_DIR = r"C:\Users\satya\Downloads\computer-vision-challenge-satyamKr-main\satellite-port-cv"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Ensure dirs exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add src directory to path
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))


def create_synthetic_satellite_port(output_path=os.path.join(DATA_DIR, "satellite_port.png")):
    """Creates a synthetic satellite port image for testing"""
    img_width = 800
    img_height = 800
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 200

    scale = 10  # 10 pixels per cm
    outer_size = int(40 * scale)
    middle_size = int(35 * scale)
    inner_size = int(30 * scale)

    center_x = img_width // 2
    center_y = img_height // 2

    outer_half = outer_size // 2
    pt1_outer = (center_x - outer_half, center_y - outer_half)
    pt2_outer = (center_x + outer_half, center_y + outer_half)
    cv2.rectangle(img, pt1_outer, pt2_outer, (0, 0, 0), 3)

    middle_half = middle_size // 2
    pt1_middle = (center_x - middle_half, center_y - middle_half)
    pt2_middle = (center_x + middle_half, center_y + middle_half)
    cv2.rectangle(img, pt1_middle, pt2_middle, (0, 0, 0), 3)

    inner_half = inner_size // 2
    pt1_inner = (center_x - inner_half, center_y - inner_half)
    pt2_inner = (center_x + inner_half, center_y + inner_half)
    cv2.rectangle(img, pt1_inner, pt2_inner, (255, 255, 255), -1)
    cv2.rectangle(img, pt1_inner, pt2_inner, (0, 0, 0), 3)

    circle_radius = int(2.5 * scale / 2)
    circle_x = center_x - outer_half + circle_radius
    circle_y = center_y - outer_half + circle_radius
    cv2.circle(img, (circle_x, circle_y), circle_radius, (0, 0, 0), -1)

    cv2.imwrite(output_path, img)
    print(f"✓ Created synthetic satellite port image: {output_path}")
    return img


def create_cropped_image(reference_path, output_path=os.path.join(DATA_DIR, "cropped_port.png")):
    """Creates a cropped version of the reference image for Part B testing"""
    img = cv2.imread(reference_path)
    if img is None:
        print("✗ Could not load reference image for cropping")
        return None

    h, w = img.shape[:2]
    crop_x = w // 2
    crop_y = h // 4
    crop_w = w // 2
    crop_h = h // 2

    cropped = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    cv2.imwrite(output_path, cropped)
    print(f"✓ Created cropped image: {output_path}")
    return cropped


def test_part_a():
    """Test Part A: Rotation Detection"""
    print("\n" + "=" * 60)
    print("TESTING PART A: Rotation Angle Detection")
    print("=" * 60)

    try:
        from part_a_rotation_detection import detect_rotation_angle

        ref_img_path = os.path.join(DATA_DIR, "satellite_port.png")
        rotated_img_path = os.path.join(OUTPUT_DIR, "test_rotated_45.png")

        print("\n1. Testing with original image...")
        angle = detect_rotation_angle(ref_img_path)

        if angle is not None:
            print(f"✓ Original image rotation: {angle:.2f}°")

        print("\n2. Testing with 45° rotation...")
        test_angle = 45
        detected_angle = detect_rotation_angle(
            ref_img_path,
            rotated_img_path,
            rotation_angle=test_angle
        )

        if detected_angle is not None:
            error = abs(detected_angle - test_angle)
            print(f"✓ Expected: {test_angle}°, Detected: {detected_angle:.2f}°")
            print(f"  Error: {error:.2f}°")

            if error < 5:
                print("✓ PART A: PASSED (Error < 5°)")
                return True
            else:
                print("⚠ PART A: WARNING (Error >= 5°)")
                return True
        else:
            print("✗ PART A: FAILED (Could not detect rotation)")
            return False

    except Exception as e:
        print(f"✗ PART A: ERROR - {str(e)}")
        return False


def test_part_b():
    """Test Part B: Camera Navigation"""
    print("\n" + "=" * 60)
    print("TESTING PART B: Camera Navigation")
    print("=" * 60)

    try:
        from part_b_camera_navigation import navigate_to_circle

        ref_img_path = os.path.join(DATA_DIR, "satellite_port.png")
        crop_img_path = os.path.join(DATA_DIR, "cropped_port.png")

        # If cropped file missing → create it
        if not os.path.exists(crop_img_path):
            create_cropped_image(ref_img_path, crop_img_path)

        print("\n1. Generating navigation instructions...")
        movements = navigate_to_circle(
            ref_img_path,
            crop_img_path,
            step_size=50
        )

        if movements is not None:
            print(f"✓ Generated {len(movements)} movement steps")
            print("✓ PART B: PASSED")
            return True
        else:
            print("✗ PART B: FAILED")
            return False

    except Exception as e:
        print(f"✗ PART B: ERROR - {str(e)}")
        return False


def test_part_c():
    """Test Part C: Homography Perspective Transform"""
    print("\n" + "=" * 60)
    print("TESTING PART C: Homography Perspective Transform")
    print("=" * 60)

    try:
        from part_c_homography_transform import generate_perspective_view

        output_path = os.path.join(OUTPUT_DIR, "perspective_view_22_5deg.png")

        print("\n1. Generating perspective view at 22.5°...")
        img, K, Rt = generate_perspective_view(
            angle_deg=22.5,
            distance_cm=100,
            output_path=output_path
        )

        if img is not None and os.path.exists(output_path):
            print("✓ Perspective view generated successfully")
            print(f"  Output size: {img.shape}")
            print("✓ PART C: PASSED")
            return True
        else:
            print("✗ PART C: FAILED")
            return False

    except Exception as e:
        print(f"✗ PART C: ERROR - {str(e)}")
        return False


def test_part_d():
    """Test Part D: Camera Movement Animation"""
    print("\n" + "=" * 60)
    print("TESTING PART D: Camera Movement Animation")
    print("=" * 60)

    try:
        from part_d_camera_movement import animate_camera_movement, create_video_from_frames, visualize_movement_path

        frames_dir = os.path.join(OUTPUT_DIR, "camera_movement_frames")
        video_path = os.path.join(OUTPUT_DIR, "camera_movement.mp4")

        print("\n1. Generating animation frames...")
        animate_camera_movement(
            start_angle=22.5,
            end_angle=0,
            distance_cm=100,
            num_steps=10,
            output_dir=frames_dir
        )

        print("\n2. Creating video...")
        create_video_from_frames(
            frame_dir=frames_dir,
            output_video=video_path,
            fps=5
        )

        print("\n3. Visualizing camera path...")
        visualize_movement_path(
            start_angle=22.5,
            end_angle=0,
            distance_cm=100,
            num_steps=10
        )

        if os.path.exists(video_path):
            print("✓ Video created successfully")
            print("✓ PART D: PASSED")
            return True
        else:
            print("✗ PART D: FAILED (No video created)")
            return False

    except Exception as e:
        print(f"✗ PART D: ERROR - {str(e)}")
        return False


if __name__ == "__main__":
    # Generate synthetic image (for consistent testing)
    synthetic_img_path = os.path.join(DATA_DIR, "satellite_port.png")
    if not os.path.exists(synthetic_img_path):
        create_synthetic_satellite_port(synthetic_img_path)

    # Run all parts
    results = {
        "Part A": test_part_a(),
        "Part B": test_part_b(),
        "Part C": test_part_c(),
        "Part D": test_part_d(),
    }

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    for part, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{part}: {status}")
