import os
import json
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from scipy.spatial import ConvexHull

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_SCAN_DIR = r"C:\\Users\\deoat\\Desktop\\Construct\\data\\scan_001"
DEFAULT_OUTPUT_DIR = r"C:\\Users\\deoat\\Desktop\\Construct\\output"

# Point Cloud Parameters
DEPTH_TRUNC = 8.0
VOXEL_SIZE = 0.02
CONFIDENCE_THRESH = 0.5

# Plane Detection Parameters
RANSAC_DISTANCE_THRESH = 0.03
RANSAC_N_POINTS = 3
RANSAC_ITERATIONS = 2000
MIN_PLANE_POINTS = 300

# Blueprint Parameters
WALL_THICKNESS = 0.15
MIN_WALL_HEIGHT = 2.0
MIN_WALL_LENGTH = 0.5

# Blueprint Style
BLUEPRINT_BLUE = (13, 71, 161)  # RGB: Deep blue
BLUEPRINT_GRID = (21, 101, 192)  # Lighter blue for grid
LINE_COLOR = (255, 255, 255)  # White
HIGHLIGHT_COLOR = (0, 255, 255)  # Cyan for openings
GRID_SIZE = 0.5  # meters

SHOW_VISUALIZATIONS = True

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_transforms(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def create_output_dirs(output_dir: str):
    dirs = ["pointcloud", "blueprint_wireframe", "visualizations"]
    for d in dirs:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)

def depth_to_pointcloud(depth_map, color_img, intrinsic_matrix, confidence_map=None):
    H, W = depth_map.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u, v, depth = u.flatten(), v.flatten(), depth_map.flatten()
    
    if confidence_map is not None:
        conf = confidence_map.flatten()
        valid = (depth > 0) & (depth < DEPTH_TRUNC) & (conf > CONFIDENCE_THRESH)
    else:
        valid = (depth > 0) & (depth < DEPTH_TRUNC)
    
    u, v, depth = u[valid], v[valid], depth[valid]
    
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    
    points = np.stack([X, Y, Z], axis=1)
    colors = color_img[v, u] / 255.0
    
    return points, colors

def transform_pointcloud(points, c2w_matrix):
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    return (c2w_matrix @ points_h.T).T[:, :3]

# ============================================================================
# PHASE A: POINT CLOUD FUSION
# ============================================================================

def fuse_pointclouds(scan_dir: str, output_dir: str):
    print("\n" + "="*60)
    print("PHASE A: POINT CLOUD FUSION")
    print("="*60)
    
    transforms = load_transforms(os.path.join(scan_dir, "transforms.json"))
    frames = transforms['frames']
    print(f"Loading {len(frames)} frames...")
    
    all_points, all_colors = [], []
    
    for idx, frame in enumerate(frames):
        depth_map = np.load(os.path.join(scan_dir, frame['depth_path']))
        color_img = cv2.imread(os.path.join(scan_dir, frame['file_path']))
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        
        if color_img.shape[:2] != depth_map.shape:
            color_img = cv2.resize(color_img, (depth_map.shape[1], depth_map.shape[0]))
        
        confidence_map = None
        if frame.get('confidence_path'):
            conf_path = os.path.join(scan_dir, frame['confidence_path'])
            if os.path.exists(conf_path):
                confidence_map = np.load(conf_path)
        
        intrinsic = np.array(frame['intrinsic_matrix'])
        c2w = np.array(frame['transform_matrix'])
        
        points_cam, colors = depth_to_pointcloud(depth_map, color_img, intrinsic, confidence_map)
        points_world = transform_pointcloud(points_cam, c2w)
        
        all_points.append(points_world)
        all_colors.append(colors)
        
        if (idx + 1) % 5 == 0:
            print(f"  Processed {idx+1}/{len(frames)} frames")
    
    print("\nMerging and filtering...")
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)
    
    print(f"✓ Final point cloud: {len(pcd.points):,} points")
    return pcd

# ============================================================================
# PHASE B: STRUCTURAL PLANE DETECTION
# ============================================================================

def detect_structural_planes(pcd):
    print("\n" + "="*60)
    print("PHASE B: STRUCTURAL PLANE DETECTION")
    print("="*60)
    
    planes = []
    remaining = pcd
    
    for idx in range(20):
        if len(remaining.points) < MIN_PLANE_POINTS:
            break
        
        model, inliers = remaining.segment_plane(
            distance_threshold=RANSAC_DISTANCE_THRESH,
            ransac_n=RANSAC_N_POINTS,
            num_iterations=RANSAC_ITERATIONS
        )
        
        if len(inliers) < MIN_PLANE_POINTS:
            break
        
        a, b, c, d = model
        normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
        
        angle = np.arccos(np.abs(normal[2])) * 180 / np.pi
        
        if angle < 20:
            ptype = "floor" if d < 0 else "ceiling"
        elif angle > 70:
            ptype = "wall"
        else:
            ptype = "sloped"
        
        plane_pcd = remaining.select_by_index(inliers)
        points = np.asarray(plane_pcd.points)
        
        print(f"Plane {idx + 1}: {ptype} | {len(inliers):,} points")
        
        planes.append({
            'id': idx,
            'type': ptype,
            'normal': normal,
            'distance': d,
            'points': points,
            'centroid': points.mean(axis=0)
        })
        
        remaining = remaining.select_by_index(inliers, invert=True)
    
    print(f"\n✓ Detected {len(planes)} planes")
    return planes

# ============================================================================
# PHASE C: EXTRACT WIREFRAME EDGES
# ============================================================================

def extract_wall_wireframe(wall_plane):
    """Extract wireframe edges for a wall."""
    points = wall_plane['points']
    normal = wall_plane['normal']
    centroid = wall_plane['centroid']
    
    # Local coordinate system
    z_axis = normal
    y_axis = np.array([0, 0, 1])
    if np.abs(np.dot(z_axis, y_axis)) > 0.99:
        y_axis = np.array([0, 1, 0])
    
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    # Transform to local coords
    points_local = (points - centroid) @ np.column_stack([x_axis, y_axis, z_axis])
    
    x_min, y_min, z_min = points_local.min(axis=0)
    x_max, y_max, z_max = points_local.max(axis=0)
    
    width = x_max - x_min
    height = y_max - y_min
    
    if height < MIN_WALL_HEIGHT or width < MIN_WALL_LENGTH:
        return None
    
    # Create corner points
    corners_local = np.array([
        [x_min, y_min, 0],
        [x_max, y_min, 0],
        [x_max, y_max, 0],
        [x_min, y_max, 0]
    ])
    
    corners_world = corners_local @ np.column_stack([x_axis, y_axis, z_axis]).T + centroid
    
    # Create edges (front face)
    edges = [
        [corners_world[0], corners_world[1]],
        [corners_world[1], corners_world[2]],
        [corners_world[2], corners_world[3]],
        [corners_world[3], corners_world[0]]
    ]
    
    # Add thickness
    back = corners_world - normal * WALL_THICKNESS
    
    # Back face edges
    edges.extend([
        [back[0], back[1]],
        [back[1], back[2]],
        [back[2], back[3]],
        [back[3], back[0]]
    ])
    
    # Connecting edges
    for i in range(4):
        edges.append([corners_world[i], back[i]])
    
    return {
        'edges': edges,
        'corners': corners_world,
        'width': width,
        'height': height,
        'centroid': centroid
    }

def extract_floor_wireframe(floor_plane):
    """Extract wireframe for floor/ceiling."""
    points = floor_plane['points']
    points_2d = points[:, :2]
    
    try:
        hull = ConvexHull(points_2d)
        hull_points_2d = points_2d[hull.vertices]
        z_height = floor_plane['centroid'][2]
        hull_3d = np.column_stack([hull_points_2d, np.full(len(hull_points_2d), z_height)])
        
        edges = []
        for i in range(len(hull_3d)):
            j = (i + 1) % len(hull_3d)
            edges.append([hull_3d[i], hull_3d[j]])
        
        return {'edges': edges, 'centroid': floor_plane['centroid']}
    except:
        return None

def generate_wireframe_model(planes):
    """Generate complete wireframe model from planes."""
    print("\n" + "="*60)
    print("PHASE C: GENERATING WIREFRAME MODEL")
    print("="*60)
    
    wireframe_data = {
        'walls': [],
        'floors': [],
        'ceilings': [],
        'all_edges': []
    }
    
    # Process walls
    wall_planes = [p for p in planes if p['type'] == 'wall']
    print(f"\nProcessing {len(wall_planes)} walls...")
    
    for wall_plane in wall_planes:
        wall_wf = extract_wall_wireframe(wall_plane)
        if wall_wf:
            wireframe_data['walls'].append(wall_wf)
            wireframe_data['all_edges'].extend(wall_wf['edges'])
            print(f"  Wall: {wall_wf['width']:.2f}m x {wall_wf['height']:.2f}m")
    
    # Process floors
    floor_planes = [p for p in planes if p['type'] == 'floor']
    print(f"\nProcessing {len(floor_planes)} floors...")
    
    for floor_plane in floor_planes:
        floor_wf = extract_floor_wireframe(floor_plane)
        if floor_wf:
            wireframe_data['floors'].append(floor_wf)
            wireframe_data['all_edges'].extend(floor_wf['edges'])
    
    # Process ceilings
    ceiling_planes = [p for p in planes if p['type'] == 'ceiling']
    print(f"\nProcessing {len(ceiling_planes)} ceilings...")
    
    for ceiling_plane in ceiling_planes:
        ceiling_wf = extract_floor_wireframe(ceiling_plane)
        if ceiling_wf:
            wireframe_data['ceilings'].append(ceiling_wf)
            wireframe_data['all_edges'].extend(ceiling_wf['edges'])
    
    print(f"\n✓ Generated {len(wireframe_data['all_edges'])} wireframe edges")
    return wireframe_data

# ============================================================================
# PHASE D: BLUEPRINT VISUALIZATION
# ============================================================================

def create_blueprint_3d_view(wireframe_data, output_dir: str, view_angle='perspective'):
    """Create 3D blueprint wireframe visualization."""
    print(f"\nGenerating 3D blueprint ({view_angle} view)...")
    
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor(tuple(c/255 for c in BLUEPRINT_BLUE))
    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(tuple(c/255 for c in BLUEPRINT_BLUE))
    
    # Style the axes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(tuple(c/255 for c in BLUEPRINT_GRID))
    ax.yaxis.pane.set_edgecolor(tuple(c/255 for c in BLUEPRINT_GRID))
    ax.zaxis.pane.set_edgecolor(tuple(c/255 for c in BLUEPRINT_GRID))
    
    ax.grid(True, color=tuple(c/255 for c in BLUEPRINT_GRID), linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(colors='white', labelsize=10)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    
    # Draw all edges
    edge_collection = Line3DCollection(
        wireframe_data['all_edges'],
        colors=tuple(c/255 for c in LINE_COLOR),
        linewidths=2,
        alpha=0.95
    )
    ax.add_collection3d(edge_collection)
    
    # Set view angle
    if view_angle == 'isometric':
        ax.view_init(elev=30, azim=45)
    elif view_angle == 'front':
        ax.view_init(elev=0, azim=0)
    elif view_angle == 'side':
        ax.view_init(elev=0, azim=90)
    else:  # perspective
        ax.view_init(elev=25, azim=135)
    
    # Set labels
    ax.set_xlabel('X (meters)', fontsize=14, weight='bold', color='white', labelpad=10)
    ax.set_ylabel('Y (meters)', fontsize=14, weight='bold', color='white', labelpad=10)
    ax.set_zlabel('Z (meters)', fontsize=14, weight='bold', color='white', labelpad=10)
    
    title = f'3D ARCHITECTURAL BLUEPRINT - {view_angle.upper()} VIEW'
    ax.set_title(title, fontsize=20, weight='bold', color='white', pad=30)
    
    # Equal aspect ratio
    all_points = []
    for edge in wireframe_data['all_edges']:
        all_points.extend(edge)
    
    if all_points:
        all_points = np.array(all_points)
        max_range = np.ptp(all_points, axis=0).max() / 2.0
        mid = all_points.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    # Add scale reference
    if wireframe_data['walls']:
        wall = wireframe_data['walls'][0]
        text = f"DIMENSIONS: {wall['width']:.2f}m × {wall['height']:.2f}m"
        ax.text2D(0.05, 0.95, text, transform=ax.transAxes,
                 color='white', fontsize=12, weight='bold',
                 bbox=dict(boxstyle='round', facecolor=tuple(c/255 for c in BLUEPRINT_BLUE), 
                          edgecolor='white', linewidth=2))
    
    plt.tight_layout()
    
    filename = f"blueprint_3d_{view_angle}.png"
    output_path = os.path.join(output_dir, "visualizations", filename)
    plt.savefig(output_path, dpi=300, facecolor=tuple(c/255 for c in BLUEPRINT_BLUE), 
               edgecolor='none', bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    
    if SHOW_VISUALIZATIONS and view_angle == 'perspective':
        plt.show()
    
    plt.close()

def create_blueprint_multiview(wireframe_data, output_dir: str):
    """Create multi-angle blueprint visualization."""
    print("\nGenerating multi-view blueprint...")
    
    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor(tuple(c/255 for c in BLUEPRINT_BLUE))
    
    views = [
        ('perspective', 25, 135, 221),
        ('isometric', 30, 45, 222),
        ('front', 0, 0, 223),
        ('side', 0, 90, 224)
    ]
    
    for view_name, elev, azim, subplot_pos in views:
        ax = fig.add_subplot(subplot_pos, projection='3d')
        ax.set_facecolor(tuple(c/255 for c in BLUEPRINT_BLUE))
        
        # Style
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, color=tuple(c/255 for c in BLUEPRINT_GRID), linestyle='-', linewidth=0.5, alpha=0.3)
        ax.tick_params(colors='white', labelsize=8)
        
        # Draw edges
        edge_collection = Line3DCollection(
            wireframe_data['all_edges'],
            colors=tuple(c/255 for c in LINE_COLOR),
            linewidths=1.5,
            alpha=0.95
        )
        ax.add_collection3d(edge_collection)
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(view_name.upper(), fontsize=14, weight='bold', color='white', pad=10)
        
        # Equal aspect
        all_points = []
        for edge in wireframe_data['all_edges']:
            all_points.extend(edge)
        
        if all_points:
            all_points = np.array(all_points)
            max_range = np.ptp(all_points, axis=0).max() / 2.0
            mid = all_points.mean(axis=0)
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    plt.suptitle('3D ARCHITECTURAL BLUEPRINT - MULTI-VIEW', 
                fontsize=24, weight='bold', color='white', y=0.98)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "visualizations", "blueprint_multiview.png")
    plt.savefig(output_path, dpi=300, facecolor=tuple(c/255 for c in BLUEPRINT_BLUE))
    print(f"✓ Saved: blueprint_multiview.png")
    
    if SHOW_VISUALIZATIONS:
        plt.show()
    
    plt.close()

def export_wireframe_to_obj(wireframe_data, output_dir: str):
    """Export wireframe as OBJ file (edges only, no faces)."""
    print("\nExporting wireframe to OBJ...")
    
    vertices = []
    edges = []
    vertex_map = {}
    
    for edge in wireframe_data['all_edges']:
        for point in edge:
            point_tuple = tuple(point)
            if point_tuple not in vertex_map:
                vertex_map[point_tuple] = len(vertices)
                vertices.append(point)
    
    for edge in wireframe_data['all_edges']:
        v1 = vertex_map[tuple(edge[0])]
        v2 = vertex_map[tuple(edge[1])]
        edges.append((v1, v2))
    
    obj_path = os.path.join(output_dir, "blueprint_wireframe", "wireframe_model.obj")
    with open(obj_path, 'w') as f:
        f.write("# 3D Blueprint Wireframe Model\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Edges: {len(edges)}\n\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write("\n")
        for e in edges:
            f.write(f"l {e[0]+1} {e[1]+1}\n")
    
    print(f"✓ Exported wireframe OBJ: {len(vertices)} vertices, {len(edges)} edges")
    print(f"  File: {obj_path}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_blueprint_generation(scan_dir: str, output_dir: str, show_visualizations: bool = True):
    global SHOW_VISUALIZATIONS
    SHOW_VISUALIZATIONS = show_visualizations
    
    print("\n" + "="*70)
    print("  🏗️  3D BLUEPRINT WIREFRAME GENERATOR")
    print("     Classic Technical Drawing Style")
    print("="*70)
    
    create_output_dirs(output_dir)
    
    # Generate point cloud
    pcd = fuse_pointclouds(scan_dir, output_dir)
    
    # Detect planes
    planes = detect_structural_planes(pcd)
    
    # Generate wireframe
    wireframe_data = generate_wireframe_model(planes)
    
    # Create visualizations
    print("\n" + "="*60)
    print("PHASE D: CREATING BLUEPRINT VISUALIZATIONS")
    print("="*60)
    
    create_blueprint_3d_view(wireframe_data, output_dir, 'perspective')
    create_blueprint_3d_view(wireframe_data, output_dir, 'isometric')
    create_blueprint_3d_view(wireframe_data, output_dir, 'front')
    create_blueprint_3d_view(wireframe_data, output_dir, 'side')
    create_blueprint_multiview(wireframe_data, output_dir)
    
    # Export wireframe
    export_wireframe_to_obj(wireframe_data, output_dir)
    
    print("\n" + "="*70)
    print("✓ BLUEPRINT GENERATION COMPLETE!")
    print("="*70)
    print(f"\n📐 Generated Files:")
    print(f"   Visualizations:")
    print(f"     • blueprint_3d_perspective.png")
    print(f"     • blueprint_3d_isometric.png")
    print(f"     • blueprint_3d_front.png")
    print(f"     • blueprint_3d_side.png")
    print(f"     • blueprint_multiview.png")
    print(f"   3D Model:")
    print(f"     • wireframe_model.obj (edge-only, no mesh)")
    print(f"\n   Location: {output_dir}/visualizations/")
    print(f"\n   The OBJ file contains ONLY edges (lines), no faces!")
    print(f"   Open in Blender/MeshLab to view the wireframe.\n")

def main():
    print("\n" + "="*70)
    print("  🏗️  3D ARCHITECTURAL BLUEPRINT GENERATOR")
    print("     White Wireframe on Blue Grid")
    print("="*70)
    run_blueprint_generation(DEFAULT_SCAN_DIR, DEFAULT_OUTPUT_DIR)

if __name__ == "__main__":
    main()
