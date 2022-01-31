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


class Quat_Avg():
    def __init__(self):
        with open("../../training/cfg/task/TenseBot6.yaml", "r") as cfg:
            try:
                self.cfg = yaml.safe_load(cfg)
            except yaml.YAMLError as exc:
                print(exc)

        self.create_sim()
        self.add_objects()
        

    def create_sim(self):
        # initialize gym
        self.gym = gymapi.acquire_gym()
        # parse arguments
        args = gymutil.parse_arguments(description="Quat Averaging Example")

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
        # sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        if args.use_gpu_pipeline:
            print("WARNING: Forcing CPU pipeline.")

        device = 'cpu'
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
        # sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_FLEX, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # create viewer using the default camera properties
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise ValueError('*** Failed to create viewer')

    def add_objects(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

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
        asset_options.fix_base_link = False
        support_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        # initial root pose for cartpole actors
        

        # Create environment 0
        # Cart held steady using position target mode.
        # Pole held at a 45 degree angle using position target mode.
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, 2)
        initial_pose = gymapi.Transform()
        self.actor_handles = []
        env_ids = []        
        for i in range(4):
            ori_start = torch.rand((3))
            ori_start = torch.tensor([1,1,0])*i*3.1415/4
            initial_pose.p = gymapi.Vec3(i, 0, 1)
            initial_pose.r = gymapi.Quat.from_euler_zyx(ori_start[0], ori_start[1], ori_start[2]) 
            Actor = self.gym.create_actor(self.env, support_asset, initial_pose, 'A{}'.format(i), collision_group, collision_filter)    
            self.actor_handles.append(Actor)
            env_ids.append(i)
        self.env_ids_int32 = torch.tensor(env_ids, dtype=torch.int32)
        for actor in self.actor_handles:
            self.gym.set_rigid_body_color(self.env, actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))
            self.gym.set_rigid_body_color(self.env, actor, 1, gymapi.MESH_VISUAL, gymapi.Vec3(0.06, 0.97, 0.38))
            self.gym.set_rigid_body_color(self.env, actor, 2, gymapi.MESH_VISUAL, gymapi.Vec3(0.38, 0.06, 0.97))
        # Look at the first env
        cam_pos = gymapi.Vec3(1, -2, 1)
        cam_target = gymapi.Vec3(1, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        num_actors = self.gym.get_actor_count(self.env)
        num_bodies = self.gym.get_env_rigid_body_count(self.env)

        # Getting root state tensors 
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.pos = self.root_states.view(num_envs, num_actors, 13)[..., 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.ori = self.root_states.view(num_envs, num_actors, 13)[..., 3:7] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.linvel = self.root_states.view(num_envs, num_actors, 13)[..., 7:10] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.angvel = self.root_states.view(num_envs, num_actors, 13)[..., 10:13] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)

    def simulation_loop(self):

        first_loop = 0
        loop_count=1
        while not self.gym.query_viewer_has_closed(self.viewer):
            # forces = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float)
            # torques = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float)    
            # gym.apply_rigid_body_force_at_pos_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(force_positions), gymapi.ENV_SPACE)

            self.gym.refresh_actor_root_state_tensor(self.sim)

            # Calculate the average quaternion from 1 and 2 and apply to 3 
            # print(self.ori)

            # self.ori[:, 2, :] = self.compute_avg_of_2_quat(self.ori[:,0, :], self.ori[:,1, :])
            self.ori[:, 3, :] = self.quat_avg_many([self.ori[:,0, :], self.ori[:,1, :], self.ori[:, 2, :]])
            # if(first_loop == 0):
            #     self.angvel[:,0,0] = 2
            #     self.angvel[:,1,1] = 2
            #     self.angvel[:,2,2] = 2
            #     first_loop = 1
            if(loop_count % 1000 == 0):
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                self.ori[:,:, 2] = 0
                time.sleep(5)
            loop_count+=1
            print(self.ori)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.root_states),
                                                        gymtorch.unwrap_tensor(self.env_ids_int32), len(self.env_ids_int32))

            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            

        
            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)

        print('Done')

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        
    def compute_avg_of_2_quat(self, q1, q2):
        w1 = 1 # Weights
        w2 = 1
        # print('q1')
        # print(q1)
        # print('q2')
        # print(q2)
        q1q2 = torch.matmul( q1, torch.transpose(q2, dim0=0, dim1=1))
        
        z = torch.sqrt( (w1-w2)**2 + 4*w1*w2*q1q2**2 )
        eigenvalue1 = 0.5 * (w1 + w2 + z)
        eigenvalue0 = -0.5 * (w1 + w2 + z)
        # print('eigenvalue0')
        # print(eigenvalue0)
        # print('eigenvalue1')
        # print(eigenvalue1)

        numerator1 = w1*(w1-w2+z)
        denominator1 = z*(w1+w2+z)
        numerator2 = w2*(w2-w1+z)
        denominator2 = z*(w1+w2+z)
        # print('q1q2')
        # print(q1q2)
        q_avg = torch.sqrt(numerator1/denominator1) * q1 + torch.sign(q1q2)*torch.sqrt(numerator2/denominator2)*q2
        # print('q_avg')
        # print(q_avg)
        # time.sleep(0.5)
        if(q1q2 != 0):
            return q_avg
        else: # Returns current position, i think we get nans if q1q2 = 0
            return self.ori[:,2,:]

    def quat_avg_many(self, q_list):
        M = torch.zeros((4,4))

        
        for q in q_list:
            # print(q)
            tmp_quat = gymapi.Quat(q[0, 0], q[0, 1], q[0, 2], q[0, 3])
            tmp_euler = list(tmp_quat.to_euler_zyx())
            # print(tmp_euler)
            tmp_euler[0] = 0
            tmp_quat = gymapi.Quat.from_euler_zyx(tmp_euler[0],tmp_euler[1],tmp_euler[2])
            # print(tmp_euler)
            # print('z{}:y{}:x{}'.format(tmp_euler[0], tmp_euler[1], tmp_euler[2]))
            # print(tmp_quat)
            # time.sleep(2)
            # print(q)
            # print(q.shape)
            q[0,0] = tmp_quat.x
            q[0,1] = tmp_quat.y
            q[0,2] = tmp_quat.z
            q[0,3] = tmp_quat.w
            # print(q)
            # time.sleep(5)

            M += torch.matmul(torch.transpose(q, dim0=0, dim1=1), q)
            # print(M)

            # #12-57 http://www.malcolmdshuster.com/FC_Lerner-SADC-ThreeAxis_MDSscan.pdf
            # q123 = q[0:3]
            # q4 = q[3]
            # I3 = torch.eye(3)
            # qqt = torch.matmul(q123, torch.transpose(q123, dim0=0, dim1=1))
            # Q = torch.tensor([[0, -q123[2], q123[1]],
            #                     [q123[2], 0, -q123[0]],
            #                     [-q123[1], q123[0], 0]])

            # A = q4**2 - torch.dot(q123,q123)*I3 + 2*qqt - 2*q4*Q
            # print(A)

            # #12-60
            # B = 

        EigVal, EigVec = torch.linalg.eig(M)
        max_idx = torch.argmax(torch.real(EigVal))
        max_q = torch.real(EigVec[:, max_idx])
        # print('!!!!!!!')
        # print(L)
        # print(V)
        # print(max_idx)
        # print(max_q)
        # print(torch.norm(max_q))
        # time.sleep(5)
        return(max_q)


Quat_Avg_Example = Quat_Avg()
Quat_Avg_Example.simulation_loop()


