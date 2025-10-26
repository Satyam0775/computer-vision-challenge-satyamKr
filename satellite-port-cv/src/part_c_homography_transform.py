import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def create_3d_coordinates():
    # Same as before...
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


def draw_satellite_port(img, coords_2d, image_width, image_height):
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


def generate_perspective_view(angle_deg=22.5, distance_cm=100, output_path=None):
    image_width = 800
    image_height = 800
    coords_3d = create_3d_coordinates()
    focal_length_pixels = 1000
    K = camera_matrix(focal_length_pixels, image_width, image_height)
    Rt = compute_projection_matrix(angle_deg, distance_cm)

    coords_2d = {}
    for key in ['outer', 'middle', 'inner']:
        coords_2d[key] = project_3d_to_2d(coords_3d[key], K, Rt)

    coords_2d['circle'] = project_3d_to_2d(coords_3d['circle'].reshape(1, -1), K, Rt)[0]
    circle_edge = coords_3d['circle'] + np.array([coords_3d['circle_radius'], 0, 0])
    circle_edge_2d = project_3d_to_2d(circle_edge.reshape(1, -1), K, Rt)[0]
    coords_2d['circle_radius'] = np.linalg.norm(circle_edge_2d - coords_2d['circle'])

    img = np.ones((image_height, image_width, 3), dtype=np.uint8) * 200
    draw_satellite_port(img, coords_2d, image_width, image_height)

    if output_path is None:
        output_path = r"C:\Users\satya\Downloads\computer-vision-challenge-satyamKr-main\satellite-port-cv\outputs\perspective_view_22_5deg.png"

    cv2.imwrite(output_path, img)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'Perspective View: {angle_deg}° from right, {distance_cm} cm away')
    plt.axis('off')
    plt.savefig(output_path.replace('.png', '_display.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Perspective view generated: {output_path}")
    return img, K, Rt


def main():
    print("Generating Perspective View")
    print("=" * 50)
    generate_perspective_view(angle_deg=22.5, distance_cm=100)


if __name__ == "__main__":
    main()
