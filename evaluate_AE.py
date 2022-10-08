import open3d
import numpy as np
import torch
import pickle
import trimesh
import torch.nn.functional as F

from architecture import AutoEncoder
device = torch.device("cuda")
model = AutoEncoder(normal_channel=False).to(device)
model.load_state_dict(torch.load("/home/baothach/shape_servo_data/teleoperation/sanity_check_examples/ex_2/autoencoder/weights_simple_architecture_success/epoch 200"))
model.eval()

base_pos = np.array([0.0, -0.44, 0.16+0.015])
delta_x = np.random.uniform(low = -0.1 , high = 0.1)
delta_y = np.random.uniform(low = -0.03 , high = 0.02)
delta_z = np.random.uniform(low = -0.1 , high = 0.1)    

random_pos = base_pos + np.array([delta_x, delta_y, delta_z])

mesh = trimesh.creation.icosphere(radius=0.02)#0.02)
# mesh = trimesh.intersections.slice_mesh_plane(mesh=mesh, plane_normal=[0,0,1], plane_origin=[0,0,0.0], cap=True)
# mesh = trimesh.creation.box((0.1, 0.1, 0.04))
T = trimesh.transformations.translation_matrix(random_pos)
mesh.apply_transform(T)

points = trimesh.sample.sample_surface_even(mesh, count=256)[0]


# reconstructed_points = data[0]

pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(np.array(points))  
pcd.paint_uniform_color([1, 0.706, 0])

# pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=8))   
# normals = np.asarray(pcd.normals)
# processed_points = np.concatenate((points, normals), axis = 1)
# processed_points = points
# print(processed_points.shape)

points_tensor = torch.from_numpy(points.transpose(1,0)).unsqueeze(0).float().to(device)
reconstructed_points = model(points_tensor)
loss = F.mse_loss(points_tensor.permute(0,2,1), reconstructed_points.permute(0,2,1))
print("loss:", loss)
# print(reconstructed_points.shape)
reconstructed_points = np.swapaxes(reconstructed_points.squeeze().cpu().detach().numpy(),0,1)
reconstructed_points = reconstructed_points[:,:3]
print(reconstructed_points.shape)
# print(reconstructed_points)
pcd2 = open3d.geometry.PointCloud()
pcd2.points = open3d.utility.Vector3dVector(np.array(reconstructed_points)) 
pcd2.paint_uniform_color([1, 0, 0])
open3d.visualization.draw_geometries([pcd, pcd2])  