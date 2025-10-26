# 🛰️ Autonomous Satellite Docking Vision System  
*A complete Computer Vision Challenge implementation by Satyam Kumar*

---

## 🚀 Project Overview

This repository contains a **Computer Vision system** simulating autonomous satellite docking operations.  
It demonstrates detection, localization, rotation estimation, and camera movement planning using **OpenCV**, **NumPy**, and **Matplotlib**.

The project is structured into four core parts:

| Part | Module | Description |
|------|---------|-------------|
| 🌀 **A** | `part_a_rotation_detection.py` | Detects the **rotation angle** of the satellite port using circular marker detection |
| 🎯 **B** | `part_b_camera_navigation.py` | Simulates **camera navigation** to bring the docking port into view |
| 🔺 **C** | `part_c_homography_transform.py` | Generates **perspective projections** using homography and 3D-to-2D transformations |
| 🎥 **D** | `part_d_camera_movement.py` | Animates **camera movement** and creates a video + path visualization |

All modules are tested using `tests/test_all_parts.py` — a complete suite that generates results in the `outputs/` folder.

---

## 🧠 Key Features

- ✅ Circle & square detection using **Hough Transform** and **Canny Edges**
- ✅ Rotation angle estimation from satellite port geometry  
- ✅ Camera navigation & movement path generation  
- ✅ Homography-based 3D → 2D projection  
- ✅ Video animation of camera movement  
- ✅ Auto test pipeline to verify all components

