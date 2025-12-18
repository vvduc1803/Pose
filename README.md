
# VM Grasp Generation (ROS 2)

![ROS 2](https://img.shields.io/badge/ROS2-Humble%2FIron-22314E.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Device](https://img.shields.io/badge/Device-RealSense_D435i-blue)

A high-performance **6-DoF Grasp Detection** package for **ROS 2**, utilizing a **coarse-to-fine deep learning pipeline**  
(**AnchorNet + PointMultiGraspNet**).

This node is specifically optimized for **Intel RealSense D435/D435i** cameras running in **native 1280×720 mode**.  
It addresses common issues found in open-source grasping pipelines such as **ghost grasps** (grasps inside the object volume) and **frame misalignment** between model and robot coordinates.

---

## ✨ Key Features

- 🎯 **Native Resolution (1280×720)**  
  Processes raw images without resizing or cropping, preserving the exact camera intrinsics for accurate pixel-to-point projection.

- 🛡️ **Ghost Grasp Fix**  
  Uses a robust invalid-depth padding strategy (`-100.0`) to prevent hallucinated grasps inside object centers—a common failure mode with standard `-1.0` padding.

- 🤖 **Standard ROS Frame Convention**  
  Automatically converts GraspNet model coordinates (**X-Approach**) to standard robot gripper coordinates (**Z-Approach**).

- ⚡ **Service-Based Architecture**  
  Grasps are generated on-demand for specific segmented instances using ROS 2 services.

- 🔍 **Advanced Debugging Tools**  
  Publishes color-coded RViz markers and a cleaned collision point cloud, making it easy to verify what the network actually “sees.”

---
## 🚀 Usage

### 1. Launch Camera (**Crucial Step**)

The network assumes a **1280×720** input.
Using a different resolution will cause geometric distortion and incorrect grasps.

```bash
ros2 launch realsense2_camera rs_launch.py \
    align_depth.enable:=true \
    pointcloud.enable:=false \
    depth_module.depth_profile:=1280x720x15 \
    rgb_camera.color_profile:=1280x720x15
```

---

### 2. Run Segmentation Node

```bash
ros2 run vm_scene_understanding_openset instance_segmentation_server_subcribe
```
---

### 3. Run Grasp Generation Node

```bash
ros2 run vm_grasp_generation vm_grasp_generation_node
```

---

### 4. Call Grasp Service

Trigger detection from the command line or another ROS 2 node:

```bash
ros2 run vm_grasp_generation grasp_client
```

---

## 📡 Topics & API

### Subscribed Topics

| Topic                                             | Type                     | Description                   |
| ------------------------------------------------- | ------------------------ | ----------------------------- |
| `/camera/camera/color/image_raw`                  | `sensor_msgs/Image`      | RGB input (1280×720 required) |
| `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image`      | Aligned depth image           |
| `/camera/camera/color/camera_info`                | `sensor_msgs/CameraInfo` | Camera intrinsics             |

### Published Topics

| Topic                    | Type                             | Description                                              |
| ------------------------ | -------------------------------- | -------------------------------------------------------- |
| `/grasp_markers`         | `visualization_msgs/MarkerArray` | 3D gripper markers (Red = low score, Green = high score) |
| `/grasp_vis`             | `sensor_msgs/Image`              | 2D debug image showing grasp centers                     |
| `/debug_collision_cloud` | `sensor_msgs/PointCloud2`        | Cleaned point cloud used for collision checking          |

---

## 🧠 Technical Troubleshooting

If you modify the code, keep these **golden rules** in mind.

### 1. Padding Value (`-100.0`)

**Issue:**
Grasps sink into objects or ignore the surface.

**Cause:**
Standard `-1.0` padding is interpreted as *1 meter behind the camera*, corrupting feature aggregation.

**Fix:**
Padding is set to `-100.0`, effectively pushing invalid points to infinity and forcing the network to rely only on visible surface geometry.

---

### 2. Frame Rotation (Model → ROS)

**Issue:**
In RViz, the gripper appears sideways.

**Cause:**

* **Model Frame:**

  * X = Approach
  * Y = Width

* **ROS Gripper Frame:**

  * Z = Approach
  * Y = Width

**Fix:**
Apply the following rotation before publishing `PoseStamped`:

```python
rot_convert = np.array([
    [0, 0, -1],  # ROS X = -Model Z
    [0, 1,  0],  # ROS Y =  Model Y
    [1, 0,  0]   # ROS Z =  Model X
])
```

---

## 👁️ Visualizing in RViz

1. Set **Fixed Frame** to:

   ```text
   camera_color_optical_frame
   ```

2. Add:

   * `MarkerArray` → `/grasp_markers`
   * `PointCloud2` → `/debug_collision_cloud`

> **Tip**
> If the cloud looks sparse or has holes, check camera lighting and message synchronization settings.

---
```
