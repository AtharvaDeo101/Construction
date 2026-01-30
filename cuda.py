# import torch
# print("PyTorch:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("GPU:", torch.cuda.get_device_name(0))
#     print("CUDA version:", torch.version.cuda)



import open3d as o3d
print(o3d.core.cuda.is_available())   

geo = r"C:\Users\deoat\Desktop\Construct\output\pointcloud\raw_cloud.ply"

o3d.visualization.draw_geometries([geo])