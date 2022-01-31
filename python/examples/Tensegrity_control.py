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

from numpy.core.getlimits import _fr1
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
import time

def QUEST_Algo():
    # Average of quaternions.
    pass


# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Joint control Methods Example")

# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 1000.0

# sim_params.flex.shape_collision_margin = 0.25
# sim_params.flex.num_outer_iterations = 4
# sim_params.flex.num_inner_iterations = 10
# sim_params.flex.solver_type = 2
# sim_params.flex.deterministic_mode = 1

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False

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
plane_params.static_friction = 0.0
plane_params.dynamic_friction = 0.0
gym.add_ground(sim, gymapi.PlaneParams())

# set up the env grid
num_envs = 1
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)

collision_group = 0
collision_filter = 0
# add cartpole urdf asset
asset_root = "../../assets"
asset_file = "urdf/RodAssembly/urdf/RodAssembly.urdf"

# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.angular_damping = 1
asset_options.max_angular_velocity = 100
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
post_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
asset_options.fix_base_link = False
sling_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
# initial root pose for cartpole actors
initial_pose = gymapi.Transform()

# Create environment 0
# Cart held steady using position target mode.
# Pole held at a 45 degree angle using position target mode.
env0 = gym.create_env(sim, env_lower, env_upper, 2)

radius = 0.05

theta = torch.tensor(0*3.1415/180)
initial_pose.p = gymapi.Vec3(radius*torch.cos(theta), 0.25, radius*torch.sin(theta))
initial_pose.r = gymapi.Quat.from_euler_zyx(-3.1415/4, 0, 0) 
Sling = gym.create_actor(env0, sling_asset, initial_pose, 'Sling', collision_group, collision_filter)

theta = torch.tensor(120*3.1415/180)
initial_pose.p = gymapi.Vec3(radius*torch.cos(theta), 0.25, radius*torch.sin(theta))
initial_pose.r = gymapi.Quat.from_euler_zyx(-3.1415/4, 3.1415*2/3, 0) 
LeftPost = gym.create_actor(env0, post_asset, initial_pose, 'LeftPost', collision_group, collision_filter)

theta = torch.tensor(240*3.1415/180)
initial_pose.p = gymapi.Vec3(radius*torch.cos(theta), 0.25, radius*torch.sin(theta))
initial_pose.r = gymapi.Quat.from_euler_zyx(-3.1415/4, 3.1415*4/3, 0) 
RightPost = gym.create_actor(env0, post_asset, initial_pose, 'RightPost', collision_group, collision_filter)

gym.set_rigid_body_color(env0, Sling, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))
gym.set_rigid_body_color(env0, Sling, 1, gymapi.MESH_VISUAL, gymapi.Vec3(0.06, 0.97, 0.38))
gym.set_rigid_body_color(env0, Sling, 2, gymapi.MESH_VISUAL, gymapi.Vec3(0.38, 0.06, 0.97))
gym.set_rigid_body_color(env0, LeftPost, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))
gym.set_rigid_body_color(env0, LeftPost, 1, gymapi.MESH_VISUAL, gymapi.Vec3(0.06, 0.97, 0.38))
gym.set_rigid_body_color(env0, LeftPost, 2, gymapi.MESH_VISUAL, gymapi.Vec3(0.38, 0.06, 0.97))
gym.set_rigid_body_color(env0, RightPost, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))
gym.set_rigid_body_color(env0, RightPost, 1, gymapi.MESH_VISUAL, gymapi.Vec3(0.06, 0.97, 0.38))
gym.set_rigid_body_color(env0, RightPost, 2, gymapi.MESH_VISUAL, gymapi.Vec3(0.38, 0.06, 0.97))
                    

# Look at the first env
cam_pos = gymapi.Vec3(0.5, 0.5, 0)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

num_actors = gym.get_actor_count(env0)
num_bodies = gym.get_env_rigid_body_count(env0)

# Get state tensors
rb_state_tensor = gym.acquire_rigid_body_state_tensor(sim)
rb_state = gymtorch.wrap_tensor(rb_state_tensor)
print(rb_state.shape)
rb_pos = rb_state.view(num_bodies, 13)[:,0:3] #(num_envs, num_rigid_bodies, 13)[pos,ori,Lin-vel,Ang-vel]
rb_ori = rb_state.view(num_bodies, 13)[:,3:7] #(num_envs, num_rigid_bodies, 13)[pos,ori,Lin-vel,Ang-vel]
rb_lin_vel = rb_state.view(num_bodies, 13)[:,7:10] #(num_envs, num_rigid_bodies, 13)[pos,ori,Lin-vel,Ang-vel]
rb_ang_vel = rb_state.view(num_bodies, 13)[:,10:13] #(num_envs, num_rigid_bodies, 13)[pos,ori,Lin-vel,Ang-vel]

# gym.refresh_dof_state_tensor(sim)
# gym.refresh_actor_root_state_tensor(sim)
gym.refresh_rigid_body_state_tensor(sim)


print('rb_pos')
print(rb_pos)

body_names = [gym.get_asset_rigid_body_name(post_asset, i) for i in range(gym.get_asset_rigid_body_count(post_asset))]
extremity_names = [s for s in body_names if "endpoint" in s]
extremity_indices = [gym.find_asset_rigid_body_index(post_asset, name) for name in extremity_names]

print(body_names)
print(extremity_names)
print(extremity_indices)


# Simulate
spring_coff = 50
damping_coff = 0.999
spring_length = 0.0
frame_count = 0
connection_list = []

# (1,2),(4,5),(7,8)
# Connect All Bottoms
connection_list.append((1, 4, 0.1))
connection_list.append((1, 7, 0.1)) 
connection_list.append((4, 7, 0.1)) 
#Connect All Tops
connection_list.append((2, 5, 0.1))
connection_list.append((2, 8, 0.1)) 
connection_list.append((5, 8, 0.1)) 

#Top1 to Bottom2
connection_list.append((2, 4, 0.1)) #Body0 top is connected to Body1 bottom
#Top2 to Bottom3
connection_list.append((5, 7, 0.1)) #Body0 top is connected to Body1 bottom
#Top3 to Bottom1    
connection_list.append((8, 1, 0.1)) #Body0 top is connected to Body1 bottom


centerleftright = 1
counter = torch.tensor(0)
while not gym.query_viewer_has_closed(viewer):
    # time.sleep(2)
    spring_length_multiplier = torch.cos(counter/100)*0.8 + 1 #Modifies the length from 0.2 to 1.8 the specified length
    counter += 1

    gym.refresh_rigid_body_state_tensor(sim)
    forces = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float)
    force_positions = rb_pos.clone()

    num_lines = len(connection_list)
    line_vertices = torch.zeros((num_lines*2,3), device=device, dtype=torch.float)
    line_colors = torch.zeros((num_lines,3), device=device, dtype=torch.float)
    i = 0
    for connection in connection_list:
        # print(connection)
        P1 = force_positions[connection[0],:]
        P2 = force_positions[connection[1],:]
        spring_constant = spring_coff
        spring_length = connection[2]*spring_length_multiplier

        endpoint_distance = torch.norm(P1-P2)
        endpoint_normalized_vector = (P1-P2)/endpoint_distance
        spring_force = spring_constant*(endpoint_distance-spring_length)
        # Set springs to only work for tension and not compression
        spring_force = torch.max(torch.tensor(spring_force), torch.zeros_like(spring_force))
        appled_force = endpoint_normalized_vector*spring_force
        
        # R2 = (P2-P1)/N
        # F1 = torch.max(torch.tensor(spring_constant*R1*(N-spring_length)), torch.zeros_like(N))
        # F1 = torch.min(torch.tensor(spring_constant*R1*(N-spring_length)), torch.tensor(0))
        print('Spring {} Tension = {}'.format(i, spring_force))
        forces[0, connection[0], :] -= appled_force
        forces[0, connection[1], :] += appled_force
    
        test = torch.zeros((2,3), device=device, dtype=torch.float)
        test[0, :] = rb_lin_vel[connection[0], :]
        test[1, :] = rb_lin_vel[connection[1], :]
        # print(test.size())
        R1T = torch.unsqueeze(endpoint_normalized_vector, 1)
        print(test.shape)
        print(R1T.shape)
        # time.sleep(5)
        diffthinggy = torch.tensor([[-1, 1]], device=device, dtype=torch.float)
        # print(diffthinggy)
        test2 = torch.matmul(diffthinggy, torch.matmul(test, R1T))
        # print(R1*test2*damping_coff)
        # print(R1)
        forces[0, connection[0], :] += torch.squeeze(endpoint_normalized_vector*test2*damping_coff)
        forces[0, connection[1], :] -= torch.squeeze(endpoint_normalized_vector*test2*damping_coff)
        # print(test2)
        
        line_vertices[i*2,:] = force_positions[connection[0],:]
        line_vertices[i*2+1,:] = force_positions[connection[1],:]
        line_colors[i,:] = torch.tensor([1.0, 0.0, 0.0])
        i += 1


    # print('forces')
    # print(forces)
    # print('force_positions')
    # print(force_positions)

    # if((frame_count % 1000) == 0):
    #     forces[0, 0, :] += torch.tensor([0.0, 0.0, 100.0])

    gym.apply_rigid_body_force_at_pos_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(force_positions), gymapi.ENV_SPACE)


    # Draw Lines
    # print('line_verts')
    # print(line_vertices)
    gym.clear_lines(viewer)
    gym.add_lines(viewer, env0, num_lines, line_vertices, line_colors)
    

    frame_count += 1
    # step the physics
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
