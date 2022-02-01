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


class Tensegrity6_testing():
    def __init__(self):
        with open("../../training/cfg/task/TenseBot6.yaml", "r") as cfg:
            try:
                self.cfg = yaml.safe_load(cfg)
            except yaml.YAMLError as exc:
                print(exc)
        self.num_envs = 4
        self.create_sim()
        self.create_envs(4, 1, 2)
        self.get_state_tensors()
        # Look at the first env
        cam_pos = gymapi.Vec3(1, -2, 1)
        cam_target = gymapi.Vec3(1, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.simulation_loop()

    def create_sim(self):
        # initialize gym
        self.gym = gymapi.acquire_gym()
        # parse arguments
        args = gymutil.parse_arguments(description="Playing with the Tensegrity Code")

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

        self.device = 'cpu'
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
        # sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_FLEX, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # create viewer using the default camera properties
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise ValueError('*** Failed to create viewer')

    def create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        collision_group = 0
        collision_filter = 0

        asset_root = "../../assets"
        support_asset_file = "urdf/RodAssembly/urdf/RodAssembly.urdf"
        core_asset_file = "urdf/Core.urdf"

        # Load asset with default control type of position for all joints
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = 1
        asset_options.max_angular_velocity = 100
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.fix_base_link = False
        support_asset = self.gym.load_asset(self.sim, asset_root, support_asset_file, asset_options)
        core_asset =  self.gym.load_asset(self.sim, asset_root, core_asset_file, asset_options)

        

        # Create environmentself.tensebot_handles = []
        self.tensebot_handles = []
        self.envs = []        
        self.create_force_sensors(support_asset)
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            # Creates a tensegrity bot for an environment
            # Returns a list of handles for the support actors
            
            self.tensebot_handles.append(self.create_tensegrity(env_ptr, support_asset, core_asset, collision_group=i, collision_filter=0))
            self.envs.append(env_ptr)
            
        self.num_actors = self.gym.get_actor_count(self.envs[0])
        self.num_bodies = self.gym.get_env_rigid_body_count(self.envs[0])

        print('num_actors:{}, num_bodies:{}'.format(self.num_actors, self.num_bodies))
        self.body_dict = self.gym.get_actor_rigid_body_dict(env_ptr, self.tensebot_handles[0][0])
        self.joint_dict = self.gym.get_actor_joint_dict(env_ptr, self.tensebot_handles[0][0])

        print('body_dict:')
        print(self.body_dict)
        for b in self.body_dict:
                print(b)
        print('joint_dict:')
        for j in self.joint_dict:
            print(j)    

        self.create_tensegrity_connections()    

    def create_tensegrity(self, env, support_asset, core_asset, collision_group, collision_filter):
        # Create the support objects for the env
        assembly_handles = []
        initial_pose = gymapi.Transform()
        self.SupportHandles=[]
        for name, pos, ori in zip(self.cfg["tensegrityParams"]["supportNames"],
                                self.cfg["tensegrityParams"]["positions"],
                                self.cfg["tensegrityParams"]["orientations"]):
            # Spawn the Support Asset
            scaled_pos = np.array(pos)*self.cfg["tensegrityParams"]["seperationDist"]
            initial_pose.p = gymapi.Vec3(scaled_pos[0],scaled_pos[1],scaled_pos[2]+self.cfg["tensegrityParams"]["spawnHeight"])
            initial_pose.r = gymapi.Quat.from_euler_zyx(ori[0],ori[1],ori[2])
            SupportHandle = self.gym.create_actor(env, support_asset, initial_pose, name, collision_group, collision_filter, 0)    
            # Set the coloring (Fun)
            # rand_colors = torch.rand((3), device=self.device)
            self.gym.set_rigid_body_color(env, SupportHandle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))
            self.gym.set_rigid_body_color(env, SupportHandle, 1, gymapi.MESH_VISUAL, gymapi.Vec3(0.06, 0.97, 0.38))
            self.gym.set_rigid_body_color(env, SupportHandle, 2, gymapi.MESH_VISUAL, gymapi.Vec3(0.38, 0.06, 0.97))
            # Add to Tensegrity list for access later
            assembly_handles.append(SupportHandle)
        # Create the armored core! 
        initial_pose.p = gymapi.Vec3(0, 0, self.cfg["tensegrityParams"]["spawnHeight"])
        initial_pose.r = gymapi.Quat.from_euler_zyx(0,0,0) 
        CoreHandle = self.gym.create_actor(env, core_asset, initial_pose, 'Core', collision_group, collision_filter)       
        assembly_handles.append(CoreHandle)            
        return assembly_handles

    def create_tensegrity_connections(self):
        # Defining the spring connection List
        self.connection_list = []
        num_connections = len(self.cfg["tensegrityParams"]["connections"])
        
        self.connection_matrix = torch.zeros(self.num_bodies, num_connections, device=self.device)
        # The +6 is for the 6 connections to the core
        for connection, idx in zip(self.cfg["tensegrityParams"]["connections"], range(num_connections)):
            print(connection)
            if(connection[0] in self.cfg["tensegrityParams"]["supportNames"] and connection[2] in self.cfg["tensegrityParams"]["supportNames"]):
                support1 = self.cfg["tensegrityParams"]["supportNames"].index(connection[0])      #converts to the body index number
                point1 = 1 if connection[1] == "B" else 0 #0 for top and 1 for bottom
                rb_num1 = support1*3 + point1 + 1 #traverse the list (1,2),(4,5),(7,8),(10,11),(13,14),(16,17). the end point rigid body index

                support2 = self.cfg["tensegrityParams"]["supportNames"].index(connection[2])
                point2 = 1 if connection[3] == "B" else 0 #0 for top and 1 for bottom
                rb_num2 = support2*3 + point2 + 1 

                spring_length = connection[4]
                self.connection_list.append([rb_num1, rb_num2, spring_length])

                self.connection_matrix[rb_num1, idx] = 1
                self.connection_matrix[rb_num2, idx] = -1
            else: #Else Core... for now
                support1 = len(self.cfg["tensegrityParams"]["supportNames"])*3
                if(connection[1] == "C1"):
                    point1 = 1
                elif(connection[1] == "C2"):
                    point1 = 2
                elif(connection[1] == "C3"):
                    point1 = 3
                elif(connection[1] == "C4"):
                    point1 = 4
                elif(connection[1] == "C5"):
                    point1 = 5
                elif(connection[1] == "C6"):
                    point1 = 6
                rb_num1 = support1 + point1

                support2 = self.cfg["tensegrityParams"]["supportNames"].index(connection[2])
                rb_num2 = support2*3
                spring_length = 0.0
                self.connection_list.append([rb_num1, rb_num2, spring_length])
                self.connection_matrix[rb_num1, idx] = 1
                self.connection_matrix[rb_num2, idx] = -1

        self.connection_list = torch.tensor(self.connection_list, device=self.device)
        # print(self.connection_list[0,...])
        # print(self.connection_matrix[0,...])
        # time.sleep(10)
    
    def get_state_tensors(self):
        # Getting root state tensors 
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.pos = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.ori = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 3:7] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.linvel = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 7:10] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.angvel = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 10:13] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.tensebot_init_pos = self.pos.clone()
        self.tensebot_init_ori = self.ori.clone()

        # Get Rigid Body state tensors
        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state = gymtorch.wrap_tensor(rb_state_tensor)
        print(self.rb_state.shape)
        self.rb_pos = self.rb_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3] #(num_envs, num_rigid_bodies, 13)[pos,ori,Lin-vel,Ang-vel]
        self.rb_ori = self.rb_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7] #(num_envs, num_rigid_bodies, 13)[pos,ori,Lin-vel,Ang-vel]
        self.rb_linvel = self.rb_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10] #(num_envs, num_rigid_bodies, 13)[pos,ori,Lin-vel,Ang-vel]
        self.rb_angvel = self.rb_state.view(self.num_envs, self.num_bodies, 13)[..., 10:13] #(num_envs, num_rigid_bodies, 13)[pos,ori,Lin-vel,Ang-vel]

        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        sensors_per_env = 12
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor)
        # tensebot_pos = root_states.view(num_envs, num_actors, 13)[..., 0:6, 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.foot_forces = self.vec_sensor_tensor.view(self.num_envs, sensors_per_env, 6)[..., 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.foot_torques = self.vec_sensor_tensor.view(self.num_envs, sensors_per_env, 6)[..., 3:6] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)

        self.contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.vec_contact_tensor = gymtorch.wrap_tensor(self.contact_tensor)
        self.body_contact_force = self.vec_contact_tensor.view(self.num_envs, self.num_bodies, 3)[:, 0:18, :]
        # # gym.refresh_dof_state_tensor(sim)
        # # gym.refresh_actor_root_state_tensor(sim)
        # gym.refresh_rigid_body_state_tensor(sim)

        # print('rb_pos')
        # print(rb_pos)

        # print(body_names)
        # print(extremity_names)
        # print(extremity_indices)

    def create_force_sensors(self, asset):
        # create force sensors attached to the "feet"
        num_bodies = self.gym.get_asset_rigid_body_count(asset)
        body_names = [self.gym.get_asset_rigid_body_name(asset, i) for i in range(num_bodies)]
        extremity_names = [s for s in body_names if "endpoint" in s]
        extremities_index = torch.zeros(len(extremity_names), dtype=torch.long, device=self.device)
        extremity_indices = [self.gym.find_asset_rigid_body_index(asset, name) for name in extremity_names]

        sensor_pose = gymapi.Transform()
        for body_idx in extremity_indices:
            print(body_idx)
            time.sleep(1)
            self.gym.create_asset_force_sensor(asset, body_idx, sensor_pose)

    def create_contact_sensors(self):
        contact_state_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_tensor = gymtorch.wrap_tensor(contact_state_tensor)   
        self.body_contact_force = self.contact_tensor.view(self.num_envs, self.num_bodies, 3)[:, 0:18, :]

    def calculate_tensegrity_forces(self, actions):
        # This might need a low pass filter
        spring_length_multiplier = actions*self.cfg["tensegrityParams"]["spring_length_change_factor"] + 1 
        # spring_length_multiplier = torch.ones((self.num_envs, len(self.cfg["tensegrityParams"]["connections"])), device=self.device)

        # forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)        
        forces = torch.zeros_like(self.rb_pos, device=self.device, dtype=torch.float)
        force_positions = self.rb_pos.clone()

        num_lines = len(self.connection_list)
        # print(self.connection_list)
        # print(num_lines)
        # print(self.connection_matrix.shape)
        # time.sleep(10)
        line_vertices = torch.zeros((num_lines*2,3), device=self.device, dtype=torch.float)
        line_colors = torch.zeros((num_lines,3), device=self.device, dtype=torch.float)

        indicies_0 = self.connection_list[:, 0].to(dtype=torch.long)
        indicies_1 = self.connection_list[:, 1].to(dtype=torch.long)
        P1s = force_positions[:, indicies_0, :] #Positions
        P2s = force_positions[:, indicies_1, :]
        V1s = self.rb_linvel[:, indicies_0, :] #Velocities
        V2s = self.rb_linvel[:, indicies_1, :]

        spring_constant = self.cfg["tensegrityParams"]["spring_coff"]
        damping_constant = self.cfg["tensegrityParams"]["damping_coff"]
        spring_lengths = torch.unsqueeze(self.connection_list[:, 2], dim=0).repeat((self.num_envs, 1))
        # print(spring_lengths)
        # print(spring_lengths.shape)
        # print(spring_length_multiplier)
        # print(spring_length_multiplier.shape)

        spring_lengths[:, 0:24] *= spring_length_multiplier
        endpoint_distances = torch.norm(P1s-P2s, dim=-1)
        # print((P1s - P2s).shape)
        # print(endpoint_distances.shape)
        # print(torch.unsqueeze(endpoint_distances, dim=-1).repeat(1,1,3).shape)
        # time.sleep(1)
        endpoint_normalized_vectors = torch.div((P1s - P2s), torch.unsqueeze(endpoint_distances,-1).repeat(1,1,3))
        spring_forces = spring_constant*(endpoint_distances-spring_lengths)
        # Set springs to exert force for tension and not compression (Helps with stability and is how cables would work)
        spring_forces = torch.max(spring_forces, torch.zeros_like(spring_forces))
        applied_forces = torch.mul(endpoint_normalized_vectors, torch.unsqueeze(spring_forces,-1).repeat(1,1,3))
        
        endpoint_velocities = torch.zeros( (self.num_envs, len(self.connection_list), 2, 3), device=self.device, dtype=torch.float)
        endpoint_velocities[:, :, 0, :] = V1s
        endpoint_velocities[:, :, 1, :] = V2s
        

        diff_matrix = torch.tensor([[-1, 1]], device=self.device, dtype=torch.float)
        diff_matrix = diff_matrix.repeat(len(self.connection_list), 1)
        diff_matrix = torch.unsqueeze(diff_matrix, dim=1)
        endpoint_velocity_components = torch.matmul(endpoint_velocities, torch.unsqueeze(endpoint_normalized_vectors, dim = -1)) # Velocity components along force line
        endpoint_velocity_diffs = torch.matmul(diff_matrix, endpoint_velocity_components ) # Difference in velocity components of each endpoint
        damping_forces = endpoint_normalized_vectors*torch.squeeze(endpoint_velocity_diffs, dim=-1)*damping_constant

        line_vertices = torch.zeros( (len(self.connection_list)*2, 3) )
        line_vertices[0::2, :] = P1s[0,:]
        line_vertices[1::2, :] = P2s[0,:]
        line_colors = torch.tensor([[1.0, 1.0, 1.0]]).repeat((len(self.connection_list), 1))


        # time.sleep(1)
        # print(self.connection_matrix.shape)
        # print(torch.unsqueeze(self.connection_matrix, 0).shape)
        # print(torch.unsqueeze(self.connection_matrix, 0).repeat(self.num_envs,1,1).shape)
        # print( (-applied_forces + damping_forces).shape )
        # print(torch.matmul(torch.unsqueeze(self.connection_matrix, 0).repeat(self.num_envs,1,1), -applied_forces + damping_forces).shape)
        # print(applied_forces.shape)
        applied_forces[:, -6:, :] /= 10
        damping_forces[:, -6:, :] /= 10
        forces = torch.matmul(torch.unsqueeze(self.connection_matrix, 0).repeat(self.num_envs,1,1), -applied_forces + damping_forces)
        forces = torch.nan_to_num(forces, nan=0.0)
        # print(forces)
        # time.sleep(1)
        
        self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(force_positions), gymapi.ENV_SPACE)
        self.gym.clear_lines(self.viewer)
        self.gym.add_lines(self.viewer, self.envs[0], num_lines, line_vertices.cpu().detach(), line_colors.cpu().detach())

    def simulation_loop(self):
        while not self.gym.query_viewer_has_closed(self.viewer):
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)

            actions = torch.zeros((self.num_envs, 24))
            self.calculate_tensegrity_forces(actions)
                
            contact = torch.where(torch.sum(self.body_contact_force, dim=-1) != 0, 1, 0)
            print(contact)

            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)

Test = Tensegrity6_testing()

# # Simulate
# spring_coff = cfg["tensegrityParams"]["spring_coff"]
# damping_coff = cfg["tensegrityParams"]["damping_coff"]
# frame_count = 1
# flip_flip = 1
# force_counter = 0

# centerleftright = 1
# counter = torch.tensor(0)
# while not gym.query_viewer_has_closed(viewer):

#     if((frame_count % 1000) == 0):
#         if(flip_flip) == 1:
#             force_counter += 1
#         else:
#             force_counter -= 1
#         if(force_counter>50):
#             flip_flip = -1
#         if(force_counter<-50):
#             force_counter = 1
#         # forces[0, 0, :] = torch.tensor([0.0, 0.0, 10.0])*flip_flip
#         # forces[0, 6, :] = torch.tensor([10.0, 0.0, 0.0])*flip_flip
#         # forces[0, 12, :] = torch.tensor([0.0, 10.0, 0.0])*flip_flip        
#         # forces[0, 3, :] = torch.tensor([0.0, 0.0, -10.0])*flip_flip
#         # forces[0, 6, :] = torch.tensor([0.1, 0.0, 0.0])*flip_flip
#         # forces[0, 9, :] = torch.tensor([-0.1, 0.0, 0.0])*flip_flip
#         # forces[0, 12, :] += torch.tensor([0.0, 0.1, 0.0])*flip_flip
#         # forces[0, 15, :] += torch.tensor([0.0, -0.1, 0.0])*flip_flip
#     frame_count += 1

#     gym.apply_rigid_body_force_at_pos_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(force_positions), gymapi.ENV_SPACE)
    

#     color = gymapi.Vec3(1,0.5,0.9)
#     # gym.draw_env_rigid_contacts(viewer, env0, color, -1.0, True)


#     gym.refresh_actor_root_state_tensor(sim)
#     avg_pos = torch.mean(tensebot_pos[0], dim=0)
#     avg_linvel = torch.mean(tensebot_linvel[0], dim=0)
#     avg_angvel = torch.mean(tensebot_angvel[0], dim=0)

#     #Quat Average
#     # torques to keep supports from yawing all over the place.... 
#     torques = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float32)
#     print(torques.shape)
#     print(num_bodies)
#     print('!!!!')
#     M = torch.zeros((4,4))
#     for i in range(num_actors):
#         q1 = gymapi.Quat()
#         print(tensebot_ori[0, i, 0])
        
#         q1.x = tensebot_ori[0, i, 0]
#         q1.y = tensebot_ori[0, i, 1]
#         q1.z = tensebot_ori[0, i, 2]
#         q1.w = tensebot_ori[0, i, 3]
#         q1.normalize()

#         q2 = gymapi.Quat()
#         q2.x = tensebot_init_ori[0, i, 0]
#         q2.y = tensebot_init_ori[0, i, 1]
#         q2.z = tensebot_init_ori[0, i, 2]
#         q2.w = tensebot_init_ori[0, i, 3]
#         q2.normalize()

#         q = q1*q2.inverse()
#         print(q1)
#         print(q2)
#         print(q)
#         print()
#         # time.sleep(10)
        
#         # tmp_euler = list(q.to_euler_zyx())
#         # q = gymapi.Quat.from_euler_zyx(tmp_euler[2],tmp_euler[1],tmp_euler[0])
#         qtmp = torch.tensor([[q.x, q.y, q.z, q.w]])
#         M += torch.matmul(torch.transpose(qtmp, dim0=0, dim1=1), qtmp)

#         # print(q.z)
#         torques[0,i*3,2] = -q.z*0.5
        
#         # print(M)
#         # time.sleep(1)
#     gym.apply_rigid_body_force_tensors(sim, None, gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)
#     EigVal, EigVec = torch.linalg.eig(M)
#     max_idx = torch.argmax(torch.real(EigVal))
#     max_q = torch.real(EigVec[:, max_idx])
    
#     white = torch.tensor([[1.0, 1.0, 1.0]]).repeat((3, 1))
#     magenta = torch.tensor([[1.0, 0.0, 1.0]]).repeat((3, 1)) 
#     black = torch.tensor([[0.0, 0.0, 0.0]]).repeat((3, 1)) 
#     green = torch.tensor([[0.0, 1.0, 0.0]]).repeat((3, 1)) 

#     axis_unit_vectors = [gymapi.Vec3(1.0, 0.0, 0.0), gymapi.Vec3(0.0, 1.0, 0.0), gymapi.Vec3(0.0, 0.0, 1.0)]
#     print(axis_unit_vectors)
#     num_lines = 3
#     center_verts = torch.zeros((6,3))
#     center_verts[0,:] = avg_pos
#     center_verts[1,:] = avg_pos + torch.tensor([axis_unit_vectors[0].x, axis_unit_vectors[0].y, axis_unit_vectors[0].z])
#     center_verts[2,:] = avg_pos
#     center_verts[3,:] = avg_pos + torch.tensor([axis_unit_vectors[1].x, axis_unit_vectors[1].y, axis_unit_vectors[1].z])
#     center_verts[4,:] = avg_pos
#     center_verts[5,:] = avg_pos + torch.tensor([axis_unit_vectors[2].x, axis_unit_vectors[2].y, axis_unit_vectors[2].z])
#     linvel_verts = torch.zeros(2,3)
#     linvel_verts[0,:] = avg_pos
#     linvel_verts[1,:] = avg_pos + avg_linvel
#     angvel_verts = center_verts.clone()
#     angvel_verts[1,:] = avg_pos + torch.tensor([avg_angvel[0], 0.0, 0.0])
#     angvel_verts[3,:] = avg_pos + torch.tensor([0.0, avg_angvel[1], 0.0])
#     angvel_verts[5,:] = avg_pos + torch.tensor([0.0, 0.0, avg_angvel[2]])
#     gym.add_lines(viewer, env0, num_lines, center_verts, white)
#     gym.add_lines(viewer, env0, 1, linvel_verts, magenta)
#     gym.add_lines(viewer, env0, 3, angvel_verts, black)

#     body_rotation =  gymapi.Transform()
#     body_rotation.r.x = max_q[0]
#     body_rotation.r.y = max_q[1]
#     body_rotation.r.z = max_q[2]
#     body_rotation.r.w = max_q[3]
#     print(body_rotation.r)
#     RX = body_rotation.transform_vector(axis_unit_vectors[0])
#     RY = body_rotation.transform_vector(axis_unit_vectors[1])
#     RZ = body_rotation.transform_vector(axis_unit_vectors[2])
#     rot_center_verts = torch.zeros((6,3))
#     rot_center_verts[0,:] = avg_pos
#     rot_center_verts[1,:] = avg_pos + torch.tensor([RX.x, RX.y, RX.z])
#     rot_center_verts[2,:] = avg_pos
#     rot_center_verts[3,:] = avg_pos + torch.tensor([RY.x, RY.y, RY.z])
#     rot_center_verts[4,:] = avg_pos
#     rot_center_verts[5,:] = avg_pos + torch.tensor([RZ.x, RZ.y, RZ.z])
#     gym.add_lines(viewer, env0, num_lines, rot_center_verts, green)

    
#     gym.refresh_force_sensor_tensor(sim)
#     # print(foot_forces)
#     # print(torch.mean(foot_forces, dim=-1))
#     # time.sleep(1)
#     # step the physics

#     gym.refresh_net_contact_force_tensor(sim)
#     contact = torch.where(torch.sum(vec_contact_tensor, dim=-1) != 0, 1, 0)
#     print(contact)
#     print(contact.shape)
#     # time.sleep(1)
    

#     gym.simulate(sim)
#     gym.fetch_results(sim, True)

#     # update the viewer
#     gym.step_graphics(sim)
#     gym.draw_viewer(viewer, sim, True)
    

 
#     # Wait for dt to elapse in real time.
#     # This synchronizes the physics simulation with the rendering rate.
#     gym.sync_frame_time(sim)

# print('Done')

# gym.destroy_viewer(viewer)
# gym.destroy_sim(sim)