import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define your base output folder
BASE_OUTPUT_DIR = r"C:\Users\satya\Downloads\computer-vision-challenge-satyamKr-main\satellite-port-cv\outputs"

def create_3d_coordinates():
    outer_square_size = 40.0
    inner_square_size = 30.0
    spacing = 2.5
    middle_square_size = inner_square_size + 2 * spacing

    outer_half = outer_square_size / 2
    outer_corners = np.array([
        [-outer_half, outer_half, 0],
        [outer_half, outer_half, 0],
        [outer_half, -outer_half, 0],
        [-outer_half, -outer_half, 0]
    ])

    middle_half = middle_square_size / 2
    middle_corners = np.array([
        [-middle_half, middle_half, 0],
        [middle_half, middle_half, 0],
        [middle_half, -middle_half, 0],
        [-middle_half, -middle_half, 0]
    ])

    inner_half = inner_square_size / 2
    inner_corners = np.array([
        [-inner_half, inner_half, 0],
        [inner_half, inner_half, 0],
        [inner_half, -inner_half, 0],
        [-inner_half, -inner_half, 0]
    ])

    circle_radius = spacing / 2
    circle_center = np.array([-outer_half + circle_radius, outer_half - circle_radius, 0])

    return {
        'outer': outer_corners,
        'middle': middle_corners,
        'inner': inner_corners,
        'circle': circle_center,
        'circle_radius': circle_radius
    }


def camera_matrix(focal_length, image_width, image_height):
    cx = image_width / 2
    cy = image_height / 2
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])
    return K


def compute_projection_matrix(angle_deg, distance_cm):
    angle_rad = np.radians(angle_deg)
    camera_pos = np.array([
        distance_cm * np.sin(angle_rad),
        0,
        distance_cm * np.cos(angle_rad)
    ])
    forward = -camera_pos / np.linalg.norm(camera_pos)
    up = np.array([0, 1, 0])
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)

    R_cw = np.column_stack((right, up, forward))
    R = R_cw.T
    t = -R @ camera_pos
    Rt = np.column_stack((R, t))
    return Rt


def project_3d_to_2d(points_3d, K, Rt):
    n = points_3d.shape[0]
    points_3d_h = np.column_stack((points_3d, np.ones(n)))
    P = K @ Rt
    points_2d_h = (P @ points_3d_h.T).T
    points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:3]
    return points_2d


def draw_satellite_port(img, coords_2d):
    pts_outer = coords_2d['outer'].astype(np.int32)
    cv2.polylines(img, [pts_outer], True, (0, 0, 0), 3)

    pts_middle = coords_2d['middle'].astype(np.int32)
    cv2.polylines(img, [pts_middle], True, (0, 0, 0), 3)

    pts_inner = coords_2d['inner'].astype(np.int32)
    cv2.fillPoly(img, [pts_inner], (255, 255, 255))
    cv2.polylines(img, [pts_inner], True, (0, 0, 0), 3)

    circle_center = tuple(coords_2d['circle'].astype(np.int32))
    circle_radius = int(coords_2d['circle_radius'])
    cv2.circle(img, circle_center, circle_radius, (0, 0, 0), -1)


def generate_frame(angle_deg, distance_cm, image_width, image_height, focal_length, coords_3d):
    K = camera_matrix(focal_length, image_width, image_height)
    Rt = compute_projection_matrix(angle_deg, distance_cm)

    coords_2d = {}
    for key in ['outer', 'middle', 'inner']:
        coords_2d[key] = project_3d_to_2d(coords_3d[key], K, Rt)

    coords_2d['circle'] = project_3d_to_2d(coords_3d['circle'].reshape(1, -1), K, Rt)[0]

    circle_edge = coords_3d['circle'] + np.array([coords_3d['circle_radius'], 0, 0])
    circle_edge_2d = project_3d_to_2d(circle_edge.reshape(1, -1), K, Rt)[0]
    coords_2d['circle_radius'] = np.linalg.norm(circle_edge_2d - coords_2d['circle'])

    img = np.ones((image_height, image_width, 3), dtype=np.uint8) * 200
    draw_satellite_port(img, coords_2d)
    return img


def animate_camera_movement(start_angle=22.5, end_angle=0, distance_cm=100,
                           num_steps=15, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(BASE_OUTPUT_DIR, "camera_movement_frames")

    os.makedirs(output_dir, exist_ok=True)

    image_width = 800
    image_height = 800
    focal_length = 1000
    coords_3d = create_3d_coordinates()
    angles = np.linspace(start_angle, end_angle, num_steps)

    print(f"Generating {num_steps} frames...")
    images = []

    for i, angle in enumerate(angles):
        img = generate_frame(angle, distance_cm, image_width, image_height,
                             focal_length, coords_3d)
        text = f"Angle: {angle:.2f} deg | Distance: {distance_cm} cm"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 0, 255), 2)
        frame_path = os.path.join(output_dir, f'frame_{i:03d}.png')
        cv2.imwrite(frame_path, img)
        images.append(img)
        print(f"Frame {i+1}/{num_steps}: Angle = {angle:.2f}° - Saved to {frame_path}")

    print(f"All frames saved to '{output_dir}'")
    return images


def create_video_from_frames(frame_dir=None, output_video=None, fps=5):
    if frame_dir is None:
        frame_dir = os.path.join(BASE_OUTPUT_DIR, "camera_movement_frames")
    if output_video is None:
        output_video = os.path.join(BASE_OUTPUT_DIR, "camera_movement.mp4")

    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
    if not frame_files:
        print("No frames found!")
        return

    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print(f"\nCreating video from {len(frame_files)} frames...")
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        video.write(frame)

    video.release()
    print(f"Video saved to: {output_video}")


def visualize_movement_path(start_angle=22.5, end_angle=0, distance_cm=100, num_steps=15):
    angles = np.linspace(start_angle, end_angle, num_steps)
    positions = []
    for angle in angles:
        angle_rad = np.radians(angle)
        x = distance_cm * np.sin(angle_rad)
        z = distance_cm * np.cos(angle_rad)
        positions.append([x, 0, z])
    positions = np.array(positions)

    plt.figure(figsize=(10, 8))
    plt.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2, label='Camera Path')
    plt.plot(positions[:, 0], positions[:, 2], 'ro', markersize=6)
    plt.plot(positions[0, 0], positions[0, 2], 'go', markersize=12, label='Start')
    plt.plot(positions[-1, 0], positions[-1, 2], 'rs', markersize=12, label='End (Front View)')
    plt.plot(0, 0, 'k*', markersize=15, label='Port Center')

    circle = plt.Circle((0, 0), distance_cm, fill=False, linestyle='--',
                       color='gray', label=f'{distance_cm} cm radius')
    plt.gca().add_patch(circle)

    plt.xlabel('X (cm) - Horizontal')
    plt.ylabel('Z (cm) - Distance from Port')
    plt.title('Camera Movement Path (Top View)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()

    path_file = os.path.join(BASE_OUTPUT_DIR, "camera_path_visualization.png")
    plt.savefig(path_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nCamera path visualization saved to: {path_file}")


def main():
    print("Camera Movement Animation")
    print("=" * 60)

    images = animate_camera_movement(
        start_angle=22.5,
        end_angle=0,
        distance_cm=100,
        num_steps=15
    )

    create_video_from_frames(fps=5)

    visualize_movement_path(
        start_angle=22.5,
        end_angle=0,
        distance_cm=100,
        num_steps=15
    )

    print("\nComplete! Check the outputs directory for all frames and video.")


if __name__ == "__main__":
    main()
