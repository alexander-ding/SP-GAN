import numpy as np
import open3d as o3d
import torch


def tensor_to_pcd(x, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(x))
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd


x = torch.load("temp.pt", map_location=torch.device("cpu"))
blue = np.array([[0], [0], [1]])
green = np.array([[0], [1], [0]])
red = np.array([[1], [0], [0]])
pose = tensor_to_pcd(x["pose"], blue)
original_chair = tensor_to_pcd(x["original_chair"], color=red)
new_chair = tensor_to_pcd(x["new_chair"], color=green)
o3d.visualization.draw_geometries([pose, original_chair])
o3d.visualization.draw_geometries([pose, new_chair])
