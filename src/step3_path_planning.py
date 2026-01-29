import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import binary_dilation
import heapq

import open3d as o3d


class OccupancyGrid:
    
    def __init__(self, resolution=0.05, z_slice_height=1.0, z_tolerance=0.3):

        self.resolution = resolution
        self.z_slice_height = z_slice_height
        self.z_tolerance = z_tolerance
        self.grid = None
        self.origin = None
        self.grid_size = None
        
    def from_point_cloud(self, points, safety_margin=2):
        """
        Create occupancy grid from 3D point cloud
        
        Args:
            points: Nx3 numpy array of xyz coordinates
            safety_margin: Dilation kernel size for obstacle inflation (grid cells)
        """
        # Filter points near desired height (for floorplan)
        z_min = self.z_slice_height - self.z_tolerance
        z_max = self.z_slice_height + self.z_tolerance
        mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        slice_points = points[mask]
        
        if len(slice_points) == 0:
            raise ValueError(f"No points found at height {self.z_slice_height}m")
        
        # Determine grid bounds
        min_bounds = np.min(slice_points[:, :2], axis=0)
        max_bounds = np.max(slice_points[:, :2], axis=0)
        self.origin = min_bounds
        
        # Calculate grid dimensions
        grid_dims = np.ceil((max_bounds - min_bounds) / self.resolution).astype(int)
        self.grid_size = tuple(grid_dims)
        
        # Initialize empty grid (0 = free, 1 = occupied)
        self.grid = np.zeros(self.grid_size, dtype=np.uint8)
        
        # Mark occupied cells
        grid_coords = ((slice_points[:, :2] - self.origin) / self.resolution).astype(int)
        grid_coords = np.clip(grid_coords, [0, 0], np.array(self.grid_size) - 1)
        self.grid[grid_coords[:, 0], grid_coords[:, 1]] = 1
        
        # Inflate obstacles for safety
        if safety_margin > 0:
            kernel = np.ones((safety_margin * 2 + 1, safety_margin * 2 + 1), dtype=np.uint8)
            self.grid = binary_dilation(self.grid, kernel).astype(np.uint8)
        
        return self.grid
    
    def world_to_grid(self, world_coords):
        """Convert world coordinates (x, y) to grid indices"""
        grid_coords = ((world_coords - self.origin) / self.resolution).astype(int)
        return tuple(grid_coords)
    
    def grid_to_world(self, grid_coords):
        """Convert grid indices to world coordinates (x, y)"""
        return np.array(grid_coords) * self.resolution + self.origin
    
    def is_valid(self, grid_coords):
        """Check if grid coordinates are within bounds and not occupied"""
        if not (0 <= grid_coords[0] < self.grid_size[0] and 
                0 <= grid_coords[1] < self.grid_size[1]):
            return False
        return self.grid[grid_coords[0], grid_coords[1]] == 0


class AStarPlanner:
    """A* pathfinding with 8-connectivity"""
    
    def __init__(self, occupancy_grid):
        self.grid = occupancy_grid
        
    def heuristic(self, a, b):
        """Euclidean distance heuristic"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, node):
        """8-connected neighbors"""
        neighbors = []
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            neighbor = (node[0] + dx, node[1] + dy)
            if self.grid.is_valid(neighbor):
                # Diagonal cost is sqrt(2), straight is 1
                cost = np.sqrt(2) if abs(dx) + abs(dy) == 2 else 1.0
                neighbors.append((neighbor, cost))
        return neighbors
    
    def plan(self, start_world, goal_world):
        """
        Find path from start to goal in world coordinates
        
        Returns:
            path: List of world coordinates [(x,y), ...] or None if no path
            stats: Dictionary with planning statistics
        """
        start = self.grid.world_to_grid(start_world)
        goal = self.grid.world_to_grid(goal_world)
        
        if not self.grid.is_valid(start):
            raise ValueError(f"Start position {start_world} is invalid/occupied")
        if not self.grid.is_valid(goal):
            raise ValueError(f"Goal position {goal_world} is invalid/occupied")
        
        # Priority queue: (f_score, counter, node)
        open_set = [(0, 0, start)]
        counter = 1
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        closed_set = set()
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            closed_set.add(current)
            
            if current == goal:
                # Reconstruct path
                path_grid = [current]
                while current in came_from:
                    current = came_from[current]
                    path_grid.append(current)
                path_grid.reverse()
                
                # Convert to world coordinates
                path_world = [self.grid.grid_to_world(p) for p in path_grid]
                
                stats = {
                    'nodes_explored': len(closed_set),
                    'path_length_grid': len(path_grid),
                    'path_cost': g_score[goal],
                    'algorithm': 'A*'
                }
                return path_world, stats
            
            for neighbor, edge_cost in self.get_neighbors(current):
                tentative_g = g_score[current] + edge_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1
        
        return None, {'nodes_explored': len(closed_set), 'algorithm': 'A*', 'success': False}


class RRTPlanner:
    """Rapidly-exploring Random Tree for path planning"""
    
    def __init__(self, occupancy_grid, max_iterations=5000, step_size=0.5, goal_sample_rate=0.1):
        self.grid = occupancy_grid
        self.max_iterations = max_iterations
        self.step_size = step_size  # meters
        self.goal_sample_rate = goal_sample_rate
        
    def sample_free(self, goal_world):
        """Sample random free configuration, biased towards goal"""
        if np.random.random() < self.goal_sample_rate:
            return goal_world
        
        # Sample random point in world space
        while True:
            rand_grid = np.random.randint(0, self.grid.grid_size[0]), \
                        np.random.randint(0, self.grid.grid_size[1])
            if self.grid.is_valid(rand_grid):
                return self.grid.grid_to_world(rand_grid)
    
    def nearest(self, tree, point):
        """Find nearest node in tree to point"""
        min_dist = float('inf')
        nearest_node = None
        for node in tree:
            dist = np.linalg.norm(np.array(node) - np.array(point))
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        return nearest_node
    
    def steer(self, from_node, to_point):
        """Steer from from_node towards to_point by step_size"""
        direction = np.array(to_point) - np.array(from_node)
        distance = np.linalg.norm(direction)
        if distance <= self.step_size:
            return tuple(to_point) if isinstance(to_point, np.ndarray) else to_point
        else:
            direction = direction / distance
            result = np.array(from_node) + direction * self.step_size
            return tuple(result)
    
    def collision_free(self, from_node, to_node):
        """Check if straight line path is collision-free"""
        from_np = np.array(from_node)
        to_np = np.array(to_node)
        dist = np.linalg.norm(to_np - from_np)
        steps = int(dist / (self.grid.resolution * 0.5)) + 1
        
        for i in range(steps + 1):
            t = i / steps
            point = from_np + t * (to_np - from_np)
            grid_coords = self.grid.world_to_grid(point)
            if not self.grid.is_valid(grid_coords):
                return False
        return True
    
    def plan(self, start_world, goal_world):
        """
        RRT path planning
        
        Returns:
            path: List of world coordinates or None
            stats: Planning statistics
        """
        tree = {tuple(start_world): None}  # node -> parent
        
        for iteration in range(self.max_iterations):
            # Sample
            rand_point = self.sample_free(goal_world)
            
            # Nearest
            nearest_node = self.nearest(tree.keys(), rand_point)
            
            # Steer
            new_node = self.steer(nearest_node, rand_point)
            
            # Check collision
            if self.collision_free(nearest_node, new_node):
                tree[new_node] = nearest_node
                
                # Check if goal reached
                if np.linalg.norm(np.array(new_node) - np.array(goal_world)) < self.step_size:
                    # Add goal to tree
                    if self.collision_free(new_node, tuple(goal_world)):
                        tree[tuple(goal_world)] = new_node
                        
                        # Reconstruct path
                        path = [tuple(goal_world)]
                        current = tuple(goal_world)
                        while tree[current] is not None:
                            current = tree[current]
                            path.append(current)
                        path.reverse()
                        
                        stats = {
                            'iterations': iteration + 1,
                            'nodes_in_tree': len(tree),
                            'path_length': len(path),
                            'algorithm': 'RRT'
                        }
                        return path, stats
        
        return None, {'iterations': self.max_iterations, 'algorithm': 'RRT', 'success': False}


class PathVisualizer:
    """Animated visualization of path planning results"""
    
    def __init__(self, occupancy_grid, figsize=(15, 12)):
        self.grid = occupancy_grid
        self.figsize = figsize
        
    def animate_planning(self, path, stats, start_world, goal_world, 
                        exploration_points=None, save_path=None, show=True):
        """
        Create animated visualization of path planning
        
        Args:
            path: List of waypoints in world coordinates
            stats: Dictionary of planning statistics
            start_world: Start position (x, y)
            goal_world: Goal position (x, y)
            exploration_points: Optional list of explored nodes for visualization
            save_path: Optional path to save animation (mp4 or gif)
            show: If False, do not call plt.show() (for headless saving)
        """
        fig = plt.figure(figsize=self.figsize)
        
        # Create 2D plot (top view)
        ax2d = plt.subplot(2, 2, (1, 3))
        ax2d.set_title(f"{stats['algorithm']} Path Planning - 2D Top View", fontsize=14, weight='bold')
        ax2d.set_xlabel('X (meters)', fontsize=11)
        ax2d.set_ylabel('Y (meters)', fontsize=11)
        ax2d.set_aspect('equal')
        ax2d.grid(True, alpha=0.3)
        
        # Show occupancy grid
        extent = [self.grid.origin[0], 
                  self.grid.origin[0] + self.grid.grid_size[0] * self.grid.resolution,
                  self.grid.origin[1], 
                  self.grid.origin[1] + self.grid.grid_size[1] * self.grid.resolution]
        
        ax2d.imshow(self.grid.grid.T, origin='lower', extent=extent, 
                   cmap='RdYlGn_r', alpha=0.6, interpolation='nearest')
        
        # Plot start and goal
        ax2d.plot(start_world[0], start_world[1], 'go', markersize=15, 
                 label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
        ax2d.plot(goal_world[0], goal_world[1], 'r*', markersize=20, 
                 label='Goal', markeredgecolor='darkred', markeredgewidth=2)
        
        # Initialize path line
        path_line, = ax2d.plot([], [], 'b-', linewidth=3, alpha=0.8, label='Path')
        path_points, = ax2d.plot([], [], 'co', markersize=6, alpha=0.6)
        
        # Create 3D plot
        ax3d = plt.subplot(2, 2, 2, projection='3d')
        ax3d.set_title('3D Path Visualization', fontsize=12, weight='bold')
        ax3d.set_xlabel('X (m)', fontsize=9)
        ax3d.set_ylabel('Y (m)', fontsize=9)
        ax3d.set_zlabel('Z (m)', fontsize=9)
        
        # Plot 3D path (with constant height)
        path_3d = np.array(path)
        z_height = np.full(len(path), self.grid.z_slice_height)
        ax3d.plot(start_world[0], start_world[1], self.grid.z_slice_height, 
                 'go', markersize=10, label='Start')
        ax3d.plot(goal_world[0], goal_world[1], self.grid.z_slice_height, 
                 'r*', markersize=15, label='Goal')
        
        path_line_3d, = ax3d.plot([], [], [], 'b-', linewidth=2, alpha=0.8)
        
        # Stats panel
        ax_stats = plt.subplot(2, 2, 4)
        ax_stats.axis('off')
        
        stats_text = f"""
        ╔══════════════════════════════════════╗
        ║     PATH PLANNING STATISTICS         ║
        ╠══════════════════════════════════════╣
        ║ Algorithm: {stats['algorithm']:<25} ║
        ║ Success: {'✓ Yes' if path else '✗ No':<27} ║
        ║                                      ║
        """
        
        if path:
            path_length_m = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) 
                               for i in range(len(path)-1))
            stats_text += f"""║ Path Length: {len(path)} waypoints{' '*(16-len(str(len(path))))}║
        ║ Path Distance: {path_length_m:.2f} meters{' '*(15-len(f'{path_length_m:.2f}'))}║
        """
        
        if 'nodes_explored' in stats:
            stats_text += f"║ Nodes Explored: {stats['nodes_explored']:<21} ║\n"
        if 'iterations' in stats:
            stats_text += f"║ Iterations: {stats['iterations']:<25} ║\n"
        if 'path_cost' in stats:
            stats_text += f"║ Path Cost: {stats['path_cost']:.2f}{' '*(26-len(f'{stats['path_cost']:.2f}'))}║\n"
        
        stats_text += """╚══════════════════════════════════════╝
        """
        
        ax_stats.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                     verticalalignment='center', bbox=dict(boxstyle='round', 
                     facecolor='wheat', alpha=0.3))
        
        ax2d.legend(loc='upper right', fontsize=10)
        
        # Animation function
        def animate(frame):
            if path is None or len(path) == 0:
                return path_line, path_points, path_line_3d
            
            # Animate path drawing
            progress = min(frame / 30, 1.0)  # 30 frames to complete
            idx = int(progress * len(path))
            
            if idx > 0:
                path_segment = path[:idx]
                path_arr = np.array(path_segment)
                path_line.set_data(path_arr[:, 0], path_arr[:, 1])
                path_points.set_data(path_arr[:, 0], path_arr[:, 1])
                
                # 3D path
                path_line_3d.set_data(path_arr[:, 0], path_arr[:, 1])
                path_line_3d.set_3d_properties(z_height[:idx])
            
            return path_line, path_points, path_line_3d
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=60, interval=50, 
                           blit=True, repeat=True)
        
        plt.tight_layout()
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=20)
            print("Animation saved!")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, anim
    
    def plot_static(self, paths_dict, start_world, goal_world, save_path=None, show=True):
        """
        Plot multiple paths for comparison
        
        Args:
            paths_dict: Dict of {algorithm_name: (path, stats)}
            start_world, goal_world: Start and goal positions
            save_path: If set, save figure to this path
            show: If False, do not call plt.show() (for headless)
        """
        fig, axes = plt.subplots(1, len(paths_dict), figsize=(6*len(paths_dict), 5))
        if len(paths_dict) == 1:
            axes = [axes]
        
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        for idx, (algo_name, (path, stats)) in enumerate(paths_dict.items()):
            ax = axes[idx]
            ax.set_title(f'{algo_name} Path', fontsize=12, weight='bold')
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Grid
            extent = [self.grid.origin[0], 
                      self.grid.origin[0] + self.grid.grid_size[0] * self.grid.resolution,
                      self.grid.origin[1], 
                      self.grid.origin[1] + self.grid.grid_size[1] * self.grid.resolution]
            
            ax.imshow(self.grid.grid.T, origin='lower', extent=extent, 
                     cmap='RdYlGn_r', alpha=0.5, interpolation='nearest')
            
            # Start/Goal
            ax.plot(start_world[0], start_world[1], 'go', markersize=12, label='Start')
            ax.plot(goal_world[0], goal_world[1], 'r*', markersize=15, label='Goal')
            
            # Path
            if path:
                path_arr = np.array(path)
                ax.plot(path_arr[:, 0], path_arr[:, 1], 
                       color=colors[idx % len(colors)], linewidth=2, 
                       marker='o', markersize=4, alpha=0.7, label='Path')
                
                # Stats
                path_length = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) 
                                 for i in range(len(path)-1))
                stats_str = f"Waypoints: {len(path)}\nDistance: {path_length:.2f}m"
                ax.text(0.02, 0.98, stats_str, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.7), fontsize=9)
            
            ax.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig


def _load_point_cloud_from_step2(step2_dir: str) -> np.ndarray:
    """Load colored point cloud from Step 2 output and return Nx3 points (world coords)."""
    ply_path = os.path.join(step2_dir, "pointcloud", "colored_cloud.ply")
    if not os.path.isfile(ply_path):
        ply_path = os.path.join(step2_dir, "pointcloud", "filtered_cloud.ply")
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"No point cloud found in {step2_dir}/pointcloud/")
    pcd = o3d.io.read_point_cloud(ply_path)
    return np.asarray(pcd.points)


def _get_start_goal_from_transforms(step2_dir: str, z_height: float = 1.0):
    """Get start (first) and goal (last) camera XY from transforms.json, at z=z_height."""
    path = os.path.join(step2_dir, "transforms.json")
    with open(path, "r") as f:
        data = json.load(f)
    frames = data["frames"]
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames for start/goal")
    c2w_first = np.array(frames[0]["transform_matrix"])
    c2w_last = np.array(frames[-1]["transform_matrix"])
    start_xy = c2w_first[:3, 3][:2].copy()
    goal_xy = c2w_last[:3, 3][:2].copy()
    return np.array(start_xy), np.array(goal_xy)


def _overlay_path_on_blueprint(step2_dir: str, path, start_world, goal_world, save_path: str) -> bool:
    """Overlay path on Step 2 floorplan image using blueprint_meta.json; save to save_path. Returns True if saved."""
    meta_path = os.path.join(step2_dir, "blueprint", "blueprint_meta.json")
    img_path = os.path.join(step2_dir, "blueprint", "floorplan_2d.png")
    if not os.path.isfile(meta_path) or not os.path.isfile(img_path):
        return False
    with open(meta_path, "r") as f:
        meta = json.load(f)
    res = meta["resolution"]
    x_min, x_max = meta["x_min"], meta["x_max"]
    y_min, y_max = meta["y_min"], meta["y_max"]
    img = cv2.imread(img_path)
    if img is None:
        return False

    def world_to_px(x, y):
        px = int((x - x_min) * res)
        py = int((y_max - y) * res)
        return (px, py)

    pts = [world_to_px(p[0], p[1]) for p in path] if path else []
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], (0, 128, 255), 3)
    if pts:
        for p in pts:
            cv2.circle(img, p, 4, (0, 128, 255), -1)
    sx, sy = world_to_px(start_world[0], start_world[1])
    gx, gy = world_to_px(goal_world[0], goal_world[1])
    cv2.circle(img, (sx, sy), 12, (0, 255, 0), 2)
    cv2.circle(img, (gx, gy), 12, (0, 0, 255), 2)
    cv2.imwrite(save_path, img)
    return True


def _convert_to_native(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_to_native(i) for i in obj]
    return obj


def run_step3(step2_output_dir: str, save_dir: Optional[str] = None, show: bool = False) -> dict:
    """
    Run path planning pipeline using Step 2 outputs (point cloud, blueprint).
    Loads point cloud, builds occupancy grid, plans A* path, generates visualizations.
    Saves: occupancy_grid.png, path_animation.gif, blueprint_with_path.png, path_output.json.
    
    Args:
        step2_output_dir: Directory containing pointcloud/, blueprint/, transforms.json.
        save_dir: Where to write outputs. Defaults to step2_output_dir.
        show: If True, call plt.show() for animations (use False for headless/API).
    
    Returns:
        Dict with keys: success, path, stats, output_paths (paths to saved files).
    """
    save_dir = save_dir or step2_output_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out_paths = {}

    points = _load_point_cloud_from_step2(step2_output_dir)
    start_world, goal_world = _get_start_goal_from_transforms(step2_output_dir, z_height=1.0)

    grid = OccupancyGrid(resolution=0.1, z_slice_height=1.0, z_tolerance=0.3)
    occ = grid.from_point_cloud(points, safety_margin=2)

    astar = AStarPlanner(grid)
    path, stats = astar.plan(start_world, goal_world)
    if not path:
        stats["success"] = False
        return {"success": False, "path": None, "stats": stats, "output_paths": {}}

    stats["success"] = True
    vis = PathVisualizer(grid)

    # Static occupancy grid + path
    grid_png = os.path.join(save_dir, "occupancy_grid.png")
    vis.plot_static({"A*": (path, stats)}, start_world, goal_world, save_path=grid_png, show=show)
    out_paths["occupancy_grid"] = grid_png

    # Path animation GIF
    gif_path = os.path.join(save_dir, "path_animation.gif")
    vis.animate_planning(path, stats, start_world, goal_world, save_path=gif_path, show=show)
    out_paths["path_animation"] = gif_path

    # Blueprint overlay
    bp_path = os.path.join(save_dir, "blueprint_with_path.png")
    if _overlay_path_on_blueprint(step2_output_dir, path, start_world, goal_world, bp_path):
        out_paths["blueprint_with_path"] = bp_path

    # Path JSON
    path_len_m = sum(
        np.linalg.norm(np.array(path[i + 1]) - np.array(path[i])) for i in range(len(path) - 1)
    )
    payload = {
        "algorithm": "A*",
        "start": _convert_to_native(start_world),
        "goal": _convert_to_native(goal_world),
        "path": [_convert_to_native(p) for p in path],
        "statistics": _convert_to_native({**stats, "path_length_meters": path_len_m}),
        "grid_info": {
            "resolution": float(grid.resolution),
            "origin": _convert_to_native(grid.origin),
            "size": [int(grid.grid_size[0]), int(grid.grid_size[1])],
        },
    }
    json_path = os.path.join(save_dir, "path_output.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    out_paths["path_json"] = json_path

    return {"success": True, "path": path, "stats": stats, "output_paths": out_paths}


def demo_pipeline(save_directory=None):
    """Demo the complete path planning pipeline (synthetic data).
    
    Args:
        save_directory: Directory to save outputs (default: current directory)
    """
    if save_directory is None:
        save_directory = os.path.expanduser("~")
    else:
        Path(save_directory).mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("INDOOR PATH PLANNING VISUALIZATION DEMO")
    print("=" * 60)
    print(f"\nSave directory: {save_directory}")
    
    # Simulate point cloud from DA3/PlanarGS output
    print("\n[1/5] Generating synthetic indoor point cloud...")
    np.random.seed(42)
    
    # Create simple room layout
    room_points = []
    # Floor
    for x in np.linspace(0, 10, 200):
        for y in np.linspace(0, 8, 160):
            room_points.append([x, y, 0.0 + np.random.normal(0, 0.05)])
    
    # Walls (obstacles)
    for z in np.linspace(0, 2.5, 30):
        for x in np.linspace(0, 10, 100):
            room_points.append([x, 0, z])  # South wall
            room_points.append([x, 8, z])  # North wall
        for y in np.linspace(0, 8, 80):
            room_points.append([0, y, z])  # West wall
            room_points.append([10, y, z])  # East wall
    
    # Interior obstacles (furniture)
    for x in np.linspace(3, 4, 20):
        for y in np.linspace(3, 5, 40):
            for z in np.linspace(0, 1.5, 20):
                room_points.append([x, y, z])
    
    for x in np.linspace(7, 8, 20):
        for y in np.linspace(1, 2, 20):
            for z in np.linspace(0, 1.5, 20):
                room_points.append([x, y, z])
    
    points = np.array(room_points)
    print(f"Generated {len(points)} points")
    
    # Create occupancy grid
    print("\n[2/5] Creating occupancy grid from point cloud...")
    grid = OccupancyGrid(resolution=0.1, z_slice_height=1.0, z_tolerance=0.3)
    occupancy = grid.from_point_cloud(points, safety_margin=2)
    print(f"Grid size: {grid.grid_size} (shape: {occupancy.shape})")
    print(f"Occupied cells: {np.sum(occupancy)} / {occupancy.size} ({100*np.sum(occupancy)/occupancy.size:.1f}%)")
    
    # Define start and goal
    start = np.array([1.0, 1.0])
    goal = np.array([9.0, 7.0])
    print(f"\nStart: {start} m")
    print(f"Goal: {goal} m")
    
    # Run A* planner
    print("\n[3/5] Running A* path planner...")
    astar = AStarPlanner(grid)
    path_astar, stats_astar = astar.plan(start, goal)
    
    if path_astar:
        print(f"✓ A* succeeded!")
        print(f"  Nodes explored: {stats_astar['nodes_explored']}")
        print(f"  Path length: {len(path_astar)} waypoints")
        print(f"  Path cost: {stats_astar['path_cost']:.2f}")
    else:
        print("✗ A* failed to find path")
    
    # Run RRT planner
    print("\n[4/5] Running RRT path planner...")
    rrt = RRTPlanner(grid, max_iterations=3000, step_size=0.3)
    path_rrt, stats_rrt = rrt.plan(start, goal)
    
    if path_rrt:
        print(f"✓ RRT succeeded!")
        print(f"  Iterations: {stats_rrt['iterations']}")
        print(f"  Tree nodes: {stats_rrt['nodes_in_tree']}")
        print(f"  Path length: {len(path_rrt)} waypoints")
    else:
        print("✗ RRT failed to find path")
    
    # Visualize
    print("\n[5/5] Creating visualizations...")
    visualizer = PathVisualizer(grid)
    
    # Animated visualization (A*)
    if path_astar:
        print("\nGenerating A* animated visualization...")
        anim_path = str(Path(save_directory) / 'astar_animation.gif')
        visualizer.animate_planning(path_astar, stats_astar, start, goal,
                                   save_path=anim_path)
    
    # Static comparison
    print("\nGenerating comparison plot...")
    paths_dict = {}
    if path_astar:
        paths_dict['A*'] = (path_astar, stats_astar)
    if path_rrt:
        paths_dict['RRT'] = (path_rrt, stats_rrt)
    
    if paths_dict:
        visualizer.plot_static(paths_dict, start, goal)
    
    # Export path to JSON
    if path_astar:
        # Convert numpy types to native Python types
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            return obj
        
        output_data = {
            'algorithm': 'A*',
            'start': start.tolist(),
            'goal': goal.tolist(),
            'path': [p.tolist() if isinstance(p, np.ndarray) else list(p) for p in path_astar],
            'statistics': convert_to_native(stats_astar),
            'grid_info': {
                'resolution': float(grid.resolution),
                'origin': grid.origin.tolist(),
                'size': [int(grid.grid_size[0]), int(grid.grid_size[1])]
            }
        }
        
        json_path = str(Path(save_directory) / 'path_output.json')
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Path exported to {json_path}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    # Check if custom save path provided
    if len(sys.argv) > 1:
        save_path = sys.argv[1]
    else:
        # Default Windows desktop path as specified by user
        save_path = r"C:\Users\deoat\Desktop\Construct\output"
    
    demo_pipeline(save_directory=save_path)