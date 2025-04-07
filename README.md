# 📌 Computer Vision Challenge – Satyam Kumar

This repository contains my solution for the Computer Vision Challenge. The goal was to apply computer vision techniques like circle detection, translation, rotation, ORB feature matching, and homography to a set of image processing tasks using Python and OpenCV.

---

## 🔧 Technologies Used
- Python
- OpenCV
- NumPy
- Matplotlib

---

## ✅ Tasks Completed

### 🔴 Task A – Circle Detection, Rotation, and Translation
- **Circle Detection**: Used `cv2.HoughCircles` to detect a circle in the image.
- **Translation**: Moved the detected circle to the top-left using `cv2.warpAffine`.
- **Rotation**: Rotated the image by 30 degrees using `cv2.getRotationMatrix2D`.

📁 **Saved Images**:
- `circle_detected_output.png`
- `Translated_lmage_Circle_Top_Left.png`
- `Rotated_lmage_30_deg.png`

---

### 🟠 Task B – ORB Feature Matching and Affine Transformation
- **Feature Extraction**: Detected keypoints using ORB (`cv2.ORB_create`).
- **Matching**: Used `cv2.BFMatcher` with Hamming distance.
- **Transformation**: Calculated translation vector using `cv2.estimateAffinePartial2D`.

📁 **Saved Image**:
- `ORB_Feature_Matching.png`

---

### 🔵 Task C – Homography Transformation
- Performed homography using `cv2.findHomography` from a side-view to simulate a 22.5-degree change in perspective.

📁 **Saved Image**:
- `task_c_homography_22deg.png`

---

### 🟣 Task D – Incremental Camera View Transformation
- Generated multiple incremental views (step-by-step transformation) from the side view to the front view using homography interpolation.

📁 **Saved Images**:
- `incremental_steps_1.png`
- `incremental_steps_2.png`
- `incremental_steps_3.png`
- `incremental_steps_4.png`
- `incremental_steps_5.png`
- `incremental_steps_6.png`

---

## 📂 Folder Structure

