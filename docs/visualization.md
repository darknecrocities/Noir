# Real-Time 3D Neural Network Visualization

## 1. Perspective Vector Projection

Project NOIR implements a custom 3D perspective projection pipeline inside `noir.visualization.visualizer_3d.NeuralVisualizer3D`. This delivers 60 FPS hardware-accelerated rendering with a guaranteed, crash-free CPU fallback.

### Coordinate Transform Equations

Given world coordinates $(x, y, z)$, camera azimuth $\theta_{\text{az}}$, elevation $\phi_{\text{el}}$, distance $d$, and screen dimensions $(W, H)$:

1. **Azimuth Rotation (around Z axis)**:
   $$x_1 = x \cos \theta_{\text{az}} - y \sin \theta_{\text{az}}$$
   $$y_1 = x \sin \theta_{\text{az}} + y \cos \theta_{\text{az}}$$
   $$z_1 = z$$

2. **Elevation Rotation (around X axis)**:
   $$x_2 = x_1$$
   $$y_2 = y_1 \cos \phi_{\text{el}} - z_1 \sin \phi_{\text{el}}$$
   $$z_2 = y_1 \sin \phi_{\text{el}} + z_1 \cos \phi_{\text{el}}$$

3. **Perspective Division to Screen Space**:
   $$z_{\text{cam}} = z_2 + d$$
   $$x_{\text{screen}} = \frac{W}{2} + \left(\frac{x_2 + \text{pan}_x}{z_{\text{cam}}}\right) \cdot S_{\text{fov}}$$
   $$y_{\text{screen}} = \frac{H}{2} - \left(\frac{y_2 + \text{pan}_y}{z_{\text{cam}}}\right) \cdot S_{\text{fov}}$$

---

## 2. Visual Encoding Mapping

| Tensor Property | Visual Representation | Visual Metric |
| :--- | :--- | :--- |
| **Activation Magnitude** | Node Radius & Radial Glow | $r = (4.0 + 8.0 \cdot a) \cdot \text{scale}_z$ |
| **Positive Synaptic Weight** | Cyan Line ($\#00e5ff$) | Thickness & Opacity $\propto \|W_{ij}\|$ |
| **Negative Synaptic Weight** | Magenta Line ($\#ff3c9e$) | Thickness & Opacity $\propto \|W_{ij}\|$ |
| **Gradient Flow Signal** | Traveling Particle Pulse | Phase $\phi(t)$ along connection vector |
| **Surprise Event** | Spatial Perturbation & Shockwave | High-frequency sinusoidal displacement |
| **Reward Event** | Energy Burst Glow | Radial intensity expansion |

---

## 3. Camera Interaction Controls

- **Left Mouse Click + Drag**: Orbit rotate around azimuth and elevation.
- **Right Mouse Click + Drag**: Pan viewport along $(X, Y)$ plane.
- **Mouse Scroll Wheel**: Smooth zoom in / zoom out.
- **Double Click**: Instant camera orientation and distance reset.
- **Node Hover / Probe**: Highlights selected node and displays live numerical activation in HUD.
