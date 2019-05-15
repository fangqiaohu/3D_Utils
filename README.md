# 3D_Utils
Utilities for 3D computer vision/graphics.

#### BlenderKmzToObj  
Blender python script, from `*.kmz` to `*.obj`.

#### BlenderSkpImporter
Blender python script, for `*.skp` files import.

#### MeshToDistanceField
Convert a triangular mesh (`$M \in \{V, F\}$`, e.g., a `*.obj` file) to a unsigned distance field (`$V \in \mathbb{R}_{+}^{n \times n \times n}$`).

#### RenderPointCloud & points3D_reprojection
Known a 3D point cloud (a point set `$\{P_i\}_{i=1}^{N}$`), given camera parameters, render a re-projected 2D point cloud in an image (not a point set) `$I \in \mathbb{N}_{+}^{h \times w}$` (**non-differentiable**).

#### ShapenetSpider
For crawling [ShapeNet](https://www.shapenet.org) dataset with multi-processing.
