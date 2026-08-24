import os
import json
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path


DEFAULT_SCAN_DIR = r"C:\\Users\\deoat\\Desktop\\Construct\\data\\scan_001"
DEFAULT_OUTPUT_DIR = r"C:\\Users\\deoat\\Desktop\\Construct\\output"

# Point Cloud Parameters
DEPTH_TRUNC = 8.0
VOXEL_SIZE = 0.015  # Fine detail to capture furniture
CONFIDENCE_THRESH = 0.5

# Blueprint Style
BLUEPRINT_BLUE = (13, 71, 161)  # Deep blue background
BLUEPRINT_GRID = (21, 101, 192)  # Grid lines
LINE_COLOR = (255, 255, 255)    # White edges
EDGE_THICKNESS = 0.8             # Line thickness

SHOW_VISUALIZATION = True



def load_transforms(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def create_output_dirs(output_dir: str):
    os.makedirs(os.path.join(output_dir, "pointcloud"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "mesh"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "blueprint"), exist_ok=True)

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


def fuse_pointclouds(scan_dir: str, output_dir: str):
    print("\n" + "="*60)
    print("PHASE A: 3D RECONSTRUCTION")
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
    
    print("\nMerging and cleaning...")
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    print(f"Total points: {len(pcd.points):,}")
    print(f"Downsampling (voxel size: {VOXEL_SIZE}m)...")
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    
    print("Removing outliers...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)
    
    pcd_path = os.path.join(output_dir, "pointcloud", "scene.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)
    
    print(f"✓ Final point cloud: {len(pcd.points):,} points")
    return pcd


def generate_mesh(pcd, output_dir: str):
    print("\n" + "="*60)
    print("PHASE B: MESH GENERATION")
    print("="*60)
    
    print("Running Poisson surface reconstruction...")
    print("(This captures all objects: walls, furniture, etc.)")
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, width=0, scale=1.1, linear_fit=False
    )
    
    # Remove low-density artifacts
    print("Cleaning mesh...")
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Smooth for cleaner edges
    mesh = mesh.filter_smooth_simple(number_of_iterations=3)
    mesh.compute_vertex_normals()
    
    print(f"✓ Mesh generated:")
    print(f"  Vertices: {len(mesh.vertices):,}")
    print(f"  Triangles: {len(mesh.triangles):,}")
    
    mesh_path = os.path.join(output_dir, "mesh", "room_model.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"✓ Saved: {mesh_path}")
    
    return mesh



def extract_visible_edges(mesh, edge_angle_threshold=30):

    print("\n" + "="*60)
    print("PHASE C: EXTRACTING VISIBLE EDGES")
    print("="*60)
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Compute face normals
    print("Computing face normals...")
    mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals)
    
    # Build edge to face mapping
    print("Building edge-face relationships...")
    edge_to_faces = {}
    
    for face_idx, tri in enumerate(triangles):
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]]))
        ]
        for edge in edges:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(face_idx)
    
    # Extract feature edges (where angle between faces is significant)
    print(f"Filtering edges (angle threshold: {edge_angle_threshold}°)...")
    feature_edges = []
    
    for edge, faces in edge_to_faces.items():
        if len(faces) == 2:  # Internal edge
            # Calculate angle between normals
            normal1 = normals[faces[0]]
            normal2 = normals[faces[1]]
            angle = np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0))
            angle_deg = np.degrees(angle)
            
            # Keep edge if angle is significant (indicates a corner/boundary)
            if angle_deg > edge_angle_threshold:
                feature_edges.append(edge)
        elif len(faces) == 1:  # Boundary edge
            feature_edges.append(edge)
    
    # Convert to 3D line segments
    edge_lines = []
    for edge in feature_edges:
        edge_lines.append([vertices[edge[0]], vertices[edge[1]]])
    
    print(f"✓ Extracted {len(edge_lines):,} feature edges")
    print(f"  (These highlight walls, furniture boundaries, etc.)")
    
    return edge_lines


def render_blueprint(mesh, edges, output_dir: str):

    print("\n" + "="*60)
    print("PHASE D: RENDERING BLUEPRINT")
    print("="*60)
    
    # Create figure
    fig = plt.figure(figsize=(24, 18))
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
    
    # Grid
    ax.grid(True, color=tuple(c/255 for c in BLUEPRINT_GRID), linestyle='-', linewidth=0.5, alpha=0.4)
    
    # Axis styling
    ax.tick_params(colors='white', labelsize=12, pad=8)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    
    # Draw all edges as white lines
    print("Drawing edges...")
    edge_collection = Line3DCollection(
        edges,
        colors=tuple(c/255 for c in LINE_COLOR),
        linewidths=EDGE_THICKNESS,
        alpha=0.9
    )
    ax.add_collection3d(edge_collection)
    
    # Set good viewing angle
    ax.view_init(elev=20, azim=135)
    
    # Labels with clear font
    ax.set_xlabel('X AXIS (meters)', fontsize=16, weight='bold', color='white', labelpad=15)
    ax.set_ylabel('Y AXIS (meters)', fontsize=16, weight='bold', color='white', labelpad=15)
    ax.set_zlabel('Z AXIS (meters)', fontsize=16, weight='bold', color='white', labelpad=15)
    
    # Title
    ax.set_title('3D ROOM MODEL - BLUEPRINT WIREFRAME', 
                fontsize=24, weight='bold', color='white', pad=40)
    
    # Equal aspect ratio for accurate proportions
    vertices = np.asarray(mesh.vertices)
    max_range = np.ptp(vertices, axis=0).max() / 2.0
    mid = vertices.mean(axis=0)
    
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    # Add information box
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    
    info_text = f"ROOM DIMENSIONS:\n"
    info_text += f"  Width (X):  {extent[0]:.2f} m\n"
    info_text += f"  Depth (Y):  {extent[1]:.2f} m\n"
    info_text += f"  Height (Z): {extent[2]:.2f} m\n\n"
    info_text += f"MESH DETAILS:\n"
    info_text += f"  Vertices:  {len(mesh.vertices):,}\n"
    info_text += f"  Triangles: {len(mesh.triangles):,}\n"
    info_text += f"  Edges:     {len(edges):,}"
    
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
             color='white', fontsize=12, weight='bold', va='top',
             family='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor=tuple(c/255 for c in BLUEPRINT_BLUE), 
                      edgecolor='white', linewidth=2))
    
    plt.tight_layout()
    
    # Save high resolution
    output_path = os.path.join(output_dir, "blueprint", "3d_blueprint_model.png")
    print(f"Saving blueprint (high resolution)...")
    plt.savefig(output_path, dpi=300, facecolor=tuple(c/255 for c in BLUEPRINT_BLUE), 
               edgecolor='none', bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    if SHOW_VISUALIZATION:
        print("\nDisplaying blueprint (close window to finish)...")
        plt.show()
    
    plt.close()


def run_blueprint_generation(scan_dir: str, output_dir: str):
    
    create_output_dirs(output_dir)
    
    # Step 1: Create point cloud from video frames
    pcd = fuse_pointclouds(scan_dir, output_dir)
    
    # Step 2: Generate 3D mesh (captures all objects)
    mesh = generate_mesh(pcd, output_dir)
    
    # Step 3: Extract clear edges (highlights walls, furniture, etc.)
    edges = extract_visible_edges(mesh, edge_angle_threshold=30)
    
    # Step 4: Render blueprint
    render_blueprint(mesh, edges, output_dir)
    

def main():
    run_blueprint_generation(DEFAULT_SCAN_DIR, DEFAULT_OUTPUT_DIR)

if __name__ == "__main__":
    main()
