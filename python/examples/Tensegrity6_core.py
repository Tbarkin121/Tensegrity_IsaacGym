"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


DOF control methods example
---------------------------
An example that demonstrates various DOF control methods:
- Load cartpole asset from an urdf
- Get/set DOF properties
- Set DOF position and velocity targets
- Get DOF positions
- Apply DOF efforts
"""

import math
from matplotlib.pyplot import spring

from numpy.core.getlimits import _fr1
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
import numpy as np
import time
import yaml
# load configuration data
with open("../../training/cfg/task/TenseBot6.yaml", "r") as cfg:
    try:
        cfg = yaml.safe_load(cfg)
    except yaml.YAMLError as exc:
        print(exc)

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Joint control Methods Example")

# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 500.0

# sim_params.flex.shape_collision_margin = 0.25
# sim_params.flex.num_outer_iterations = 4
# sim_params.flex.num_inner_iterations = 10
# sim_params.flex.solver_type = 2
# sim_params.flex.deterministic_mode = 1

sim_params.physx.solver_type = 0
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False

sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
# sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

device = 'cpu'
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
# sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_FLEX, sim_params)



if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# set up the env grid
num_envs = 1
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)

collision_group = 0
collision_filter = 0
# add cartpole urdf asset
asset_root = "../../assets"
support_asset_file = "urdf/RodAssembly/urdf/RodAssembly.urdf"
core_asset_file = "urdf/Core.urdf"

# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.angular_damping = 1
asset_options.max_angular_velocity = 100
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
print("Loading asset '%s' from '%s'" % (support_asset_file, asset_root))
asset_options.fix_base_link = False
support_asset = gym.load_asset(sim, asset_root, support_asset_file, asset_options)
core_asset = gym.load_asset(sim, asset_root, core_asset_file, asset_options)
# initial root pose for cartpole actors
initial_pose = gymapi.Transform()

# Create environment 0
# Cart held steady using position target mode.
# Pole held at a 45 degree angle using position target mode.
env0 = gym.create_env(sim, env_lower, env_upper, 2)



# time.sleep(100)
# initial root pose for tensegrity support actors
initial_pose = gymapi.Transform()
Tensegrity=[]

# create force sensors attached to the "feet"
num_bodies = gym.get_asset_rigid_body_count(support_asset)
body_names = [gym.get_asset_rigid_body_name(support_asset, i) for i in range(num_bodies)]
extremity_names = [s for s in body_names if "endpoint" in s]
extremities_index = torch.zeros(len(extremity_names), dtype=torch.long, device=device)
extremity_indices = [gym.find_asset_rigid_body_index(support_asset, name) for name in extremity_names]

sensor_pose = gymapi.Transform()
for body_idx in extremity_indices:
    gym.create_asset_force_sensor(support_asset, body_idx, sensor_pose)

for name, pos, ori in zip(cfg["tensegrityParams"]["supportNames"],
                          cfg["tensegrityParams"]["positions"],
                          cfg["tensegrityParams"]["orientations"]):
    # Spawn the Support Asset
    scaled_pos = np.array(pos)*cfg["tensegrityParams"]["seperationDist"]
    initial_pose.p = gymapi.Vec3(scaled_pos[0],scaled_pos[1],scaled_pos[2]+cfg["tensegrityParams"]["spawnHeight"])
    initial_pose.r = gymapi.Quat.from_euler_zyx(ori[0],ori[1],ori[2]) 
    SupportHandle = gym.create_actor(env0, support_asset, initial_pose, name, collision_group, collision_filter)    
    # Set the coloring (Fun)
    gym.set_rigid_body_color(env0, SupportHandle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))
    gym.set_rigid_body_color(env0, SupportHandle, 1, gymapi.MESH_VISUAL, gymapi.Vec3(0.06, 0.97, 0.38))
    gym.set_rigid_body_color(env0, SupportHandle, 2, gymapi.MESH_VISUAL, gymapi.Vec3(0.38, 0.06, 0.97))
    # Add to Tensegrity list for access later
    Tensegrity.append(SupportHandle)
# Create the armored core! 
initial_pose.p = gymapi.Vec3(0, 0, cfg["tensegrityParams"]["spawnHeight"])
initial_pose.r = gymapi.Quat.from_euler_zyx(0,0,0) 
CoreHandle = gym.create_actor(env0, core_asset, initial_pose, 'Core', collision_group, collision_filter)   

# Look at the first env
cam_pos = gymapi.Vec3(0.25, 0.25, 0.1)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

num_actors = gym.get_actor_count(env0)
num_bodies = gym.get_env_rigid_body_count(env0)

connection_matrix = torch.zeros((num_bodies, len(cfg["tensegrityParams"]["connections"])+6))
connection_list = []
idx = 0
for connection in cfg["tensegrityParams"]["connections"]:
    support1 = cfg["tensegrityParams"]["supportNames"].index(connection[0])      #converts to the body index number
    point1 = 1 if connection[1] == "B" else 0 #0 for top and 1 for bottom
    rb_num1 = support1*3 + point1 + 1 #traverse the list (1,2),(4,5),(7,8),(10,11),(13,14),(16,17). the end point rigid body index

    support2 = cfg["tensegrityParams"]["supportNames"].index(connection[2])
    point2 = 1 if connection[3] == "B" else 0 #0 for top and 1 for bottom
    rb_num2 = support2*3 + point2 + 1 

    spring_length = connection[4]
    connection_list.append([rb_num1, rb_num2, spring_length])

    connection_matrix[rb_num1, idx] = 1
    connection_matrix[rb_num2, idx] = -1
    idx += 1

# Add Core Connections
tmp=1
for support in cfg["tensegrityParams"]["supportNames"]:
    support = cfg["tensegrityParams"]["supportNames"].index(support)
    rb_num1 = support*3 #This will be the rb of the center of the support
    core = 6
    rb_num2 = core*3+tmp
    tmp+=1
    spring_length=0.0
    connection_list.append([rb_num1, rb_num2, spring_length])

    connection_matrix[rb_num1, idx] = 1
    connection_matrix[rb_num2, idx] = -1
    idx += 1
    # time.sleep(2)


connection_list = torch.tensor(connection_list, device=device)

# print(connection_list)
# print(connection_matrix)
# time.sleep(5)    


# Get state tensors
rb_state_tensor = gym.acquire_rigid_body_state_tensor(sim)
rb_state = gymtorch.wrap_tensor(rb_state_tensor)
print(rb_state.shape)
rb_pos = rb_state.view(num_bodies, 13)[:,0:3] #(num_envs, num_rigid_bodies, 13)[pos,ori,Lin-vel,Ang-vel]
rb_ori = rb_state.view(num_bodies, 13)[:,3:7] #(num_envs, num_rigid_bodies, 13)[pos,ori,Lin-vel,Ang-vel]
gym.refresh_rigid_body_state_tensor(sim)
rb_initial_ori = rb_ori.detach().clone()
rb_lin_vel = rb_state.view(num_bodies, 13)[:,7:10] #(num_envs, num_rigid_bodies, 13)[pos,ori,Lin-vel,Ang-vel]
rb_ang_vel = rb_state.view(num_bodies, 13)[:,10:13] #(num_envs, num_rigid_bodies, 13)[pos,ori,Lin-vel,Ang-vel]

print(rb_ori)
print(rb_initial_ori)
# time.sleep(5)

actor_root_state = gym.acquire_actor_root_state_tensor(sim)
root_states = gymtorch.wrap_tensor(actor_root_state)
tensebot_pos = root_states.view(num_envs, num_actors, 13)[..., 0:6, 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
tensebot_ori = root_states.view(num_envs, num_actors, 13)[..., 0:6, 3:7] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
tensebot_linvel = root_states.view(num_envs, num_actors, 13)[..., 0:6, 7:10] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
tensebot_angvel = root_states.view(num_envs, num_actors, 13)[..., 0:6, 10:13] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)

gym.refresh_actor_root_state_tensor(sim)

tensebot_init_pos = tensebot_pos.clone()
tensebot_init_ori = tensebot_ori.clone()

contact_tensor = gym.acquire_net_contact_force_tensor(sim)
vec_contact_tensor = gymtorch.wrap_tensor(contact_tensor)

# gym.refresh_dof_state_tensor(sim)
# gym.refresh_actor_root_state_tensor(sim)
gym.refresh_rigid_body_state_tensor(sim)

print('rb_pos')
print(rb_pos)

print(body_names)
print(extremity_names)
print(extremity_indices)


# Simulate
spring_coff = cfg["tensegrityParams"]["spring_coff"]
damping_coff = cfg["tensegrityParams"]["damping_coff"]
frame_count = 1
flip_flip = 1
force_counter = 0

centerleftright = 1
counter = torch.tensor(0)
while not gym.query_viewer_has_closed(viewer):
    spring_length_multiplier = torch.cos(counter/100)*cfg["tensegrityParams"]["spring_length_change_factor"]+ 1 #Modifies the length from 0.1 to 1.9 the specified length
    counter += 1

    gym.refresh_rigid_body_state_tensor(sim)
    forces = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float)
    force_positions = rb_pos.clone()

    num_lines = len(connection_list)
    line_vertices = torch.zeros((num_lines*2,3), device=device, dtype=torch.float)
    line_colors = torch.zeros((num_lines,3), device=device, dtype=torch.float)
    i = 0
    # print("connection_list")
    # print(connection_list)
    # print(connection_list.shape)
    indicies_0 = connection_list[:, 0].to(dtype=torch.long)
    indicies_1 = connection_list[:, 1].to(dtype=torch.long)
    P1s = force_positions[indicies_0, :] #Positions
    P2s = force_positions[indicies_1, :]
    V1s = rb_lin_vel[indicies_0, :] #Velocities
    V2s = rb_lin_vel[indicies_1, :]

    spring_constant = spring_coff
    spring_lengths = connection_list[:, 2]*spring_length_multiplier
    # print('P1s')
    # print(P1s)
    # print('P2s')
    # print(P2s)
    endpoint_distances = torch.norm(P1s-P2s, dim=-1)
    # print('endpoint_distances')
    # print(torch.unsqueeze(endpoint_distances,1).repeat(1,3))
    # print(torch.unsqueeze(endpoint_distances,1).repeat(1,3).shape)

    endpoint_normalized_vectors = torch.div((P1s - P2s), torch.unsqueeze(endpoint_distances,1).repeat(1,3))
    # print('endpoint_normalized_vectors')
    # print(endpoint_normalized_vectors)
    # print(endpoint_normalized_vectors.shape)

    spring_forces = spring_constant*(endpoint_distances-spring_lengths)
    
    # Set springs to exert force for tension and not compression (Helps with stability and is how cables would work)
    spring_forces = torch.max(spring_forces, torch.zeros_like(spring_forces))
    applied_forces = torch.mul(endpoint_normalized_vectors, torch.unsqueeze(spring_forces,1).repeat(1,3))
    # print('spring_forces')
    # print(spring_forces)
    # print(spring_forces.shape)
    # print('applied_forces')
    # print(applied_forces)
    # print(applied_forces.shape)


    endpoint_velocities = torch.zeros((len(connection_list), 2, 3), device=device, dtype=torch.float)
    endpoint_velocities[:, 0, :] = V1s
    endpoint_velocities[:, 1, :] = V2s
    

    diff_matrix = torch.tensor([[-1, 1]], device=device, dtype=torch.float)
    diff_matrix = diff_matrix.repeat(len(connection_list), 1)
    diff_matrix = torch.unsqueeze(diff_matrix, dim=1)
    endpoint_velocity_components = torch.matmul(endpoint_velocities, torch.unsqueeze(endpoint_normalized_vectors, dim = -1)) # Velocity components along force line
    endpoint_velocity_diffs = torch.matmul(diff_matrix, endpoint_velocity_components ) # Difference in velocity components of each endpoint
    damping_forces = endpoint_normalized_vectors*torch.squeeze(endpoint_velocity_diffs, dim=-1)*damping_coff
    # print('damping_force')
    # print(damping_force)
    # print(damping_force.shape)
    # time.sleep(1)

    line_vertices = torch.zeros(len(connection_list)*2, 3)
    line_vertices[0::2, :] = P1s
    line_vertices[1::2, :] = P2s
    line_colors = torch.tensor([[1.0, 0.0, 0.0]]).repeat((len(connection_list), 1))
    i += 1

    applied_forces[-6:] /= 10
    damping_forces[-6:] /= 10
    forces[0,:] = torch.matmul(connection_matrix, -applied_forces + damping_forces)
    # force_position_offsets = torch.matmul(connection_matrix, endpoint_normalized_vectors*0.01)
    # force_positions += force_position_offsets

    # print(forces)
    # time.sleep(10)

    # print('forces')
    # print(forces)
    # print('force_positions')
    # print(force_positions)
    print(force_positions)
    # time.sleep(5)
    if((frame_count % 1000) == 0):
        if(flip_flip) == 1:
            force_counter += 1
        else:
            force_counter -= 1
        if(force_counter>50):
            flip_flip = -1
        if(force_counter<-50):
            force_counter = 1
        # forces[0, 0, :] = torch.tensor([0.0, 0.0, 10.0])*flip_flip
        # forces[0, 6, :] = torch.tensor([10.0, 0.0, 0.0])*flip_flip
        # forces[0, 12, :] = torch.tensor([0.0, 10.0, 0.0])*flip_flip        
        # forces[0, 3, :] = torch.tensor([0.0, 0.0, -10.0])*flip_flip
        # forces[0, 6, :] = torch.tensor([0.1, 0.0, 0.0])*flip_flip
        # forces[0, 9, :] = torch.tensor([-0.1, 0.0, 0.0])*flip_flip
        # forces[0, 12, :] += torch.tensor([0.0, 0.1, 0.0])*flip_flip
        # forces[0, 15, :] += torch.tensor([0.0, -0.1, 0.0])*flip_flip
    frame_count += 1

    gym.apply_rigid_body_force_at_pos_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(force_positions), gymapi.ENV_SPACE)
    

    # Draw Lines
    # print('line_verts')
    # print(line_vertices)
    gym.clear_lines(viewer)
    gym.add_lines(viewer, env0, num_lines, line_vertices, line_colors)

    color = gymapi.Vec3(1,0.5,0.9)
    # gym.draw_env_rigid_contacts(viewer, env0, color, -1.0, True)


    gym.refresh_actor_root_state_tensor(sim)
    avg_pos = torch.mean(tensebot_pos[0], dim=0)
    avg_linvel = torch.mean(tensebot_linvel[0], dim=0)
    avg_angvel = torch.mean(tensebot_angvel[0], dim=0)

    #Quat Average
    M = torch.zeros((4,4))
    for i in range(num_actors-1):
        q1 = gymapi.Quat()
        print(tensebot_ori[0, i, 0])
        
        q1.x = tensebot_ori[0, i, 0]
        q1.y = tensebot_ori[0, i, 1]
        q1.z = tensebot_ori[0, i, 2]
        q1.w = tensebot_ori[0, i, 3]
        q1.normalize()

        q2 = gymapi.Quat()
        q2.x = tensebot_init_ori[0, i, 0]
        q2.y = tensebot_init_ori[0, i, 1]
        q2.z = tensebot_init_ori[0, i, 2]
        q2.w = tensebot_init_ori[0, i, 3]
        q2.normalize()

        q = q1*q2.inverse()
        print(q1)
        print(q2)
        print(q)
        print()
        # time.sleep(10)
        
        # tmp_euler = list(q.to_euler_zyx())
        # q = gymapi.Quat.from_euler_zyx(tmp_euler[2],tmp_euler[1],tmp_euler[0])
        qtmp = torch.tensor([[q.x, q.y, q.z, q.w]])
        M += torch.matmul(torch.transpose(qtmp, dim0=0, dim1=1), qtmp)

        # print(q.z)        
        # print(M)
        # time.sleep(1)

    EigVal, EigVec = torch.linalg.eig(M)
    max_idx = torch.argmax(torch.real(EigVal))
    max_q = torch.real(EigVec[:, max_idx])
    
    white = torch.tensor([[1.0, 1.0, 1.0]]).repeat((3, 1))
    magenta = torch.tensor([[1.0, 0.0, 1.0]]).repeat((3, 1)) 
    black = torch.tensor([[0.0, 0.0, 0.0]]).repeat((3, 1)) 
    green = torch.tensor([[0.0, 1.0, 0.0]]).repeat((3, 1)) 

    axis_unit_vectors = [gymapi.Vec3(1.0, 0.0, 0.0), gymapi.Vec3(0.0, 1.0, 0.0), gymapi.Vec3(0.0, 0.0, 1.0)]
    print(axis_unit_vectors)
    num_lines = 3
    center_verts = torch.zeros((6,3))
    center_verts[0,:] = avg_pos
    center_verts[1,:] = avg_pos + torch.tensor([axis_unit_vectors[0].x, axis_unit_vectors[0].y, axis_unit_vectors[0].z])
    center_verts[2,:] = avg_pos
    center_verts[3,:] = avg_pos + torch.tensor([axis_unit_vectors[1].x, axis_unit_vectors[1].y, axis_unit_vectors[1].z])
    center_verts[4,:] = avg_pos
    center_verts[5,:] = avg_pos + torch.tensor([axis_unit_vectors[2].x, axis_unit_vectors[2].y, axis_unit_vectors[2].z])
    linvel_verts = torch.zeros(2,3)
    linvel_verts[0,:] = avg_pos
    linvel_verts[1,:] = avg_pos + avg_linvel
    angvel_verts = center_verts.clone()
    angvel_verts[1,:] = avg_pos + torch.tensor([avg_angvel[0], 0.0, 0.0])
    angvel_verts[3,:] = avg_pos + torch.tensor([0.0, avg_angvel[1], 0.0])
    angvel_verts[5,:] = avg_pos + torch.tensor([0.0, 0.0, avg_angvel[2]])
    gym.add_lines(viewer, env0, num_lines, center_verts, white)
    gym.add_lines(viewer, env0, 1, linvel_verts, magenta)
    gym.add_lines(viewer, env0, 3, angvel_verts, black)

    body_rotation =  gymapi.Transform()
    body_rotation.r.x = max_q[0]
    body_rotation.r.y = max_q[1]
    body_rotation.r.z = max_q[2]
    body_rotation.r.w = max_q[3]
    print(body_rotation.r)
    RX = body_rotation.transform_vector(axis_unit_vectors[0])
    RY = body_rotation.transform_vector(axis_unit_vectors[1])
    RZ = body_rotation.transform_vector(axis_unit_vectors[2])
    rot_center_verts = torch.zeros((6,3))
    rot_center_verts[0,:] = avg_pos
    rot_center_verts[1,:] = avg_pos + torch.tensor([RX.x, RX.y, RX.z])
    rot_center_verts[2,:] = avg_pos
    rot_center_verts[3,:] = avg_pos + torch.tensor([RY.x, RY.y, RY.z])
    rot_center_verts[4,:] = avg_pos
    rot_center_verts[5,:] = avg_pos + torch.tensor([RZ.x, RZ.y, RZ.z])
    gym.add_lines(viewer, env0, num_lines, rot_center_verts, green)

    
    gym.refresh_force_sensor_tensor(sim)
    # print(foot_forces)
    # print(torch.mean(foot_forces, dim=-1))
    # time.sleep(1)
    # step the physics

    gym.refresh_net_contact_force_tensor(sim)
    contact = torch.where(torch.sum(vec_contact_tensor, dim=-1) != 0, 1, 0)
    print(contact)
    print(contact.shape)
    # time.sleep(1)
    

    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    

 
    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)