import numpy as np
import os
import torch
import time
import matplotlib.pyplot as plt

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from isaacgym.gymtorch import *

from isaacgymenvs.utils.torch_jit_utils import *
from tasks.base.vec_task import VecTask


class TenseBot(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        self.dt = self.cfg["sim"]["dt"]

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        # self.randomization_params = self.cfg["task"]["randomization_params"]
        # self.randomize = self.cfg["task"]["randomize"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        # self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        # self.power_scale = self.cfg["env"]["powerScale"]
        self.heading_weight = self.cfg["env"]["headingWeight"]
        self.up_weight = self.cfg["env"]["upWeight"]
        # self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        # self.energy_cost_scale = self.cfg["env"]["energyCost"]
        # self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]

        # self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.drive_mode = self.cfg["env"]["actuatorParams"]["driveMode"]
        self.stiffness = self.cfg["env"]["actuatorParams"]["stiffness"] * self.drive_mode
        self.damping = self.cfg["env"]["actuatorParams"]["damping"] * self.drive_mode
        self.maxPosition = self.cfg["env"]["actuatorParams"]["maxPosition"]
        self.maxSpeed = self.cfg["env"]["actuatorParams"]["maxSpeed"]
        self.maxTorque = self.cfg["env"]["actuatorParams"]["maxTorque"]
        self.friction = self.cfg["env"]["actuatorParams"]["friction"]
        self.torqueDecay = self.cfg["env"]["actuatorParams"]["torqueDecay"]

        self.angularDamping = self.cfg["env"]["assetParams"]["angularDamping"]
        self.angularVelocity = self.cfg["env"]["assetParams"]["angularVelocity"]

        self.goal_dist = self.cfg["env"]["goalDist"]
        self.goal_threshold = self.cfg["env"]["goalThreshold"]

        # obs_buf shapes: (53)
        # obs_buf[0:39] = Rod State x 3 : Pos(3), Ori(4), LinVel(3), AngVel(3)
        # obs_buf[39:42] = Goal Pos : Pos(3)
        # obs_buf[42:45] = vector to goal (3) 
        # obs_buf[45:53] = actions : Spring Length Multipliers (9)
        self.cfg["env"]["numObservations"] = 54
        # Spring Length Mulitpliers (9)
        self.cfg["env"]["numActions"] = 9

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        # set init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang
        self.base_init_state = torch.tensor(state, device=self.device)
        self.start_rotation = torch.tensor(rot, device=self.device)
        
        # get gym GPU root state tensor
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        print('root_state')
        print(self.root_states.cpu().detach().numpy())
        print(self.root_states.shape)
        print('num_envs {}, num_actors {}'.format(self.num_envs, self.num_actors))

        self.tensebot_pos = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0:3, 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.tensebot_ori = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0:3, 3:7] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.tensebot_linvel = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0:3, 7:10] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.tensebot_angvel = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0:3, 10:13] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.goal_pos = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 3, 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.tensebot_root_state = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0:3, :]
        self.tensebot_initial_root_states = self.tensebot_root_state.clone()
        # self.tensebot_initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)

        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state = gymtorch.wrap_tensor(rb_state_tensor)
        print('rigid_body_state')
        print(self.rb_state.cpu().detach().numpy())
        print(self.rb_state.shape)
        print('num_envs {}, num_bodies {}'.format(self.num_envs, self.num_bodies))

        self.rb_pos = self.rb_state.view(self.num_envs, self.num_bodies, 13)[:, :, 0:3] #num_envs, num_rigid_bodies, 13 (pos,ori,Lvel,Avel)
        self.rb_ori = self.rb_state.view(self.num_envs, self.num_bodies, 13)[:, :, 3:7] #num_envs, num_rigid_bodies, 13 (pos,ori,Lvel,Avel)
        self.rb_linvel = self.rb_state.view(self.num_envs, self.num_bodies, 13)[:, :, 7:10] #num_envs, num_rigid_bodies, 13 (pos,ori,Lvel,Avel)
        self.rb_angvel = self.rb_state.view(self.num_envs, self.num_bodies, 13)[:, :, 10:13] #num_envs, num_rigid_bodies, 13 (pos,ori,Lvel,Avel)
        # Used for rewarding moving towards a target
        

        # tensebot_avg_pos = torch.mean(self.tensebot_pos, dim=1)
        tensebot_avg_pos = self.tensebot_pos[:,0,:]
        to_target = self.goal_pos - tensebot_avg_pos
        to_target[:, 2] = 0.0
        self.potentials = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.prev_potentials = self.potentials.clone()
        
        self.goal_reset = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        goal_ids = self.goal_reset.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_ids) > 0:
            self.reset_goal(goal_ids)

        # Measurements for rewards
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()
        
        self.frame_count = 0
        self.plot_buffer = []
        self.accumulated_reward = torch.zeros_like(self.rew_buf)

        camOffset = gymapi.Vec3(0, -1.5, 0.25)
        camTarget = gymapi.Vec3(self.tensebot_pos[0, 0, 0],self.tensebot_pos[0, 0, 1],self.tensebot_pos[0, 0, 2])
        self.gym.viewer_camera_look_at(self.viewer, None, camOffset+camTarget, camTarget)

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        # self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/RodAssembly/urdf/RodAssembly.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = self.angularDamping
        asset_options.max_angular_velocity = self.angularVelocity

        rod_assembly_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(rod_assembly_asset)

        goal_asset = self.gym.create_sphere(self.sim, 0.025)
        self.num_bodies = self.gym.get_asset_rigid_body_count(rod_assembly_asset)*3 + self.gym.get_asset_rigid_body_count(goal_asset) #3 rod assemblies per tensebot
        # self.num_actor = get_sim_actor_count
        
        pose = gymapi.Transform()

        self.rod_handles = []
        self.tensebot_handles = []
        self.goal_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            tensebot_handle = []
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            radius = 0.05
            thetas = [0, 3.1415*2/3, 3.1415*4/3] #0 deg, 120 deg, 240 deg
            for t, j in zip(thetas, range(len(thetas))):
                pose.p = gymapi.Vec3(radius*torch.cos(torch.tensor(t)), radius*torch.sin(torch.tensor(t)), 0.1)
                pose.r = gymapi.Quat.from_euler_zyx(-3.1415/4, 0, t) 
                rod_handle = self.gym.create_actor(env_ptr, rod_assembly_asset, pose, "rodassembly{}".format(j), i, 0, 0)

                rand_color = torch.rand((3), device=self.device)
                for j in range(self.num_bodies):
                    # self.gym.set_rigid_body_color(
                    #     env_ptr, tensebot_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.27, 0.1, 0.66))
                    self.gym.set_rigid_body_color(
                        env_ptr, rod_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(rand_color[0],rand_color[1],rand_color[2]))
                
                self.rod_handles.append(rod_handle)
                tensebot_handle.append(rod_handle)    


            self.tensebot_handles.append(tensebot_handle)
            self.envs.append(env_ptr)

            # Set Up the Goal Actor
            goal_pose = gymapi.Transform()
            goal_pose.p.y = self.goal_dist
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_pose, "goal", i, 1, 1)
            self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.8, 0.2))
            self.goal_handles.append(goal_handle)

            
        self.num_actors = self.gym.get_actor_count(self.envs[0])
        self.body_dict = self.gym.get_actor_rigid_body_dict(env_ptr, tensebot_handle[0])
        self.joint_dict = self.gym.get_actor_joint_dict(env_ptr, tensebot_handle[0])

        print('body_dict:')
        print(self.body_dict)
        for b in self.body_dict:
                print(b)
        print('joint_dict:')
        for j in self.joint_dict:
            print(j)        


    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:], self.goal_reset = compute_tensebot_reward(
            self.tensebot_pos,
            self.goal_pos,
            self.reset_buf,
            self.progress_buf,
            self.potentials,
            self.prev_potentials,
            self.max_episode_length,
            self.goal_threshold)
        
    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # print('self.root_state')
        # print(self.root_states[0,:])
        # print(self.root_states.shape)
        # time.sleep(1)

        self.obs_buf[:], self.potentials[:], self.prev_potentials[:] = compute_tensebot_observations(
            self.tensebot_pos,
            self.tensebot_ori,
            self.tensebot_linvel,
            self.tensebot_angvel,
            self.goal_pos, 
            self.potentials,
            self.actions, 
            self.dt)
        return self.obs_buf

    def reset_idx(self, env_ids):
        print('Resetting IDX! Env_IDs = {}'.format(env_ids))
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)*self.num_actors
        env_ids_int32 = torch.cat((env_ids_int32, env_ids_int32+1, env_ids_int32+2))

        self.tensebot_root_state[env_ids, :, :] = self.tensebot_initial_root_states[env_ids, :, :]

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.goal_reset[env_ids] = 1
        
        # actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        # self.root_states = gymtorch.wrap_tensor(actor_root_state)
        # self.initial_root_states = self.root_states.clone()
        # self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)


        # plt.plot([0,0,0])
        # plt.show()
        # if(self.plot_buffer):
        #     plot_data = np.array(self.plot_buffer)
        #     print(plot_data.shape)
        #     plt.plot(plot_data[:,0,0] + plot_data[:,1,0] + plot_data[:,2,0], label="Total Reward")
        #     plt.plot(plot_data[:,0,0], label="Progress Reward")
        #     plt.plot(plot_data[:,1,0], label="Height Reward")
        #     plt.plot(plot_data[:,2,0], label="Heading Reward")
        #     plt.ylabel('Reward')
        #     plt.xlabel('Steps')
        #     plt.grid()
        #     plt.legend(loc="lower right")
        #     plt.xlim([0, 500])
        #     plt.ylim([-0.1, 2.1])
        #     plt.show()
        #     self.plot_buffer = []

    def reset_goal(self, env_ids):
        print('reset_goal')
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # print('Resetting Goals! Env_IDs = {}'.format(env_ids))
        # print('Old Goal Position = {}'.format(self.goal_pos))

        env_ids_int32 = env_ids.to(dtype=torch.int32)*self.num_actors
        goal_pos_update = torch_rand_float(-self.goal_dist, self.goal_dist, (len(env_ids), 3), device=self.device)
        # goal_pos_update[:,0] = 1000.0
        # goal_pos_update[:,1] = 0.0
        goal_pos_update[:,2] = 0.1
        self.goal_pos[env_ids, :] = goal_pos_update
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.root_states),
                                              gymtorch.unwrap_tensor(env_ids_int32+3), len(env_ids_int32))

        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # tensebot_avg_pos = torch.mean(self.tensebot_pos, dim=1)
        tensebot_avg_pos = self.tensebot_pos[:,0,:]
        to_target = self.goal_pos[env_ids, :] - tensebot_avg_pos[env_ids, :]
        to_target[:, 2] = 0.0        
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.goal_reset[env_ids] = 0
        # print('New Goal Position = {}'.format(self.goal_pos))

    def pre_physics_step(self, actions):
        # print('actions')
        # print(actions)
        # print(actions.shape)
        # print(actions.to(self.device).squeeze().shape())
        self.actions = actions.clone().detach().to(self.device)
        self.calculate_tensegrity_forces(self.actions)

    def calculate_tensegrity_forces(self, actions):
        # # print('actions : {}'.format(actions))
        connection_list = []
        # (1,2),(4,5),(7,8) end point indicies (bottom, top) 
        # 0, 3, 6 are the body indicies
        # 9 is the goal index

        # This might need a low pass filter
        spring_length_multiplier = actions/4 + 1 # Multiplier range from 0.5 to 1.5
        # spring_length_multiplier = torch.rand((self.num_envs, 9), device=self.device)/4 + 1
        # spring_length_multiplier = torch.ones((self.num_envs, 9), device=self.device)*0.1
        
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

        #Connect All The Things... 

        
        forces = torch.zeros_like(self.rb_pos, device=self.device, dtype=torch.float)
        force_positions = self.rb_pos.clone()
        lin_vel_mat = torch.zeros((self.num_envs, 2, 3), device=self.device, dtype=torch.float) # Used in calculating damping force
        diff_matrix= torch.tensor([[-1, 1]], device=self.device, dtype=torch.float)

        num_lines = len(connection_list)
        line_vertices = torch.zeros((num_lines*2,3), device=self.device, dtype=torch.float)
        line_colors = torch.zeros((num_lines,3), device=self.device, dtype=torch.float)

        for connection, i in zip(connection_list, range(len(connection_list))):
            # print(connection)
            # Spring Force
            P1 = self.rb_pos[:, connection[0], :]
            P2 = self.rb_pos[:, connection[1], :]
            endpoint_vector = P1-P2
            # print('endpoint_vector.shape')
            # print(endpoint_vector.shape)
            spring_constant = 25
            damping_coff = 0.99
            spring_length = connection[2] * spring_length_multiplier[:, i]
            # print('spring_length.shape')
            # print(spring_length.shape)
            # print('P1.shape')
            # print(P1.shape)
            # print('P2.shape')
            # print(P2.shape)
            endpoint_distance = torch.norm(endpoint_vector, dim=1)
            # print('endpoint_distance.shape')
            # print(endpoint_distance.shape)

            endpoint_vector_normalized = torch.div(endpoint_vector, torch.unsqueeze(endpoint_distance,1).repeat(1,3))
            # print('endpoint_vector_normalized.shape')
            # print(endpoint_vector_normalized.shape)
            spring_force = spring_constant*(endpoint_distance-spring_length)
            # print('spring_force.shape')
            # print(spring_force.shape)
            # Set springs to only work for tension and not compression
            spring_force = torch.max(torch.tensor(spring_force), torch.zeros_like(spring_force))
            applied_force = torch.mul(endpoint_vector_normalized, torch.unsqueeze(spring_force,1).repeat(1,3))
            applied_force = torch.nan_to_num(applied_force, nan=0.0)
            # print('applied force')
            # print(appled_force.shape)
          
            # print('Spring {} Tension = {}'.format(i, spring_force))
            # print('forces.shape')
            # print(forces.shape)
            # print(connection[0])
            # print(connection[1])
            forces[:, connection[0], :] -= applied_force
            forces[:, connection[1], :] += applied_force
            # print('forces[0,:,:]')
            # print(forces[0,:,:])
            # print('applied_force[0,:]')
            # print(applied_force[0,:])
            # print('endpoint_vector_normalized')
            # print(endpoint_vector_normalized)
            # print(endpoint_distance)
            # Damping
            lin_vel_mat[:, 0, :] = self.rb_linvel[:, connection[0], :]
            lin_vel_mat[:, 1, :] = self.rb_linvel[:, connection[1], :]
            EVN_mat = torch.unsqueeze(endpoint_vector_normalized, 2)
            # print(lin_vel_mat.shape)
            # print(EVN_mat.shape)
            damping_force = torch.matmul(diff_matrix, torch.matmul(lin_vel_mat, EVN_mat))*damping_coff
            # print('damping_force.shape')
            # print(torch.squeeze(damping_force, dim=2).shape)
            # print('endpoint_vector_normalized.shape')
            # print(endpoint_vector_normalized.shape)
            damping_force_vector = endpoint_vector_normalized *torch.squeeze(damping_force, dim=2)
            # print('damping_force_vector.shape')
            # print(damping_force_vector.shape)
            damping_force_vector = torch.nan_to_num(damping_force_vector, nan=0.0)
            forces[:, connection[0], :] += damping_force_vector
            forces[:, connection[1], :] -= damping_force_vector
            
            # Draw Spring Connections? 
            line_vertices[i*2,:] = self.rb_pos[0, connection[0], :]
            line_vertices[i*2+1,:] = self.rb_pos[0, connection[1], :]
            line_colors[i,:] = torch.tensor([1.0, 1.0, 1.0])
        
        self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(force_positions), gymapi.ENV_SPACE)
        self.gym.clear_lines(self.viewer)
        self.gym.add_lines(self.viewer, self.envs[0], num_lines, line_vertices.cpu().detach(), line_colors.cpu().detach())
    

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        goal_ids = self.goal_reset.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_ids) > 0:
            self.reset_goal(goal_ids)

        self.compute_observations()
        self.compute_reward()

        # Look at the first actor
        env_idx = 0
        camOffset = gymapi.Vec3(0, -1.5, 0.25)
        camTarget = gymapi.Vec3(self.tensebot_pos[env_idx, 0, 0],self.tensebot_pos[env_idx, 0, 1],self.tensebot_pos[env_idx, 0, 2])
        camEnvOffset = gymapi.Vec3(0, 0, 0)
        # print(camOffset)
        # print(camTarget)
        # self.gym.viewer_camera_look_at(self.viewer, None, camOffset+camTarget+camEnvOffset, camTarget+camEnvOffset)
        # time.sleep(0.1)   
        # self.debug_printout(env_ids)

    def debug_printout(self, env_ids):
        self.accumulated_reward += self.rew_buf
        # print('potentials and previous potentials')
        # print(self.potentials)
        # print(self.prev_potentials)
        print('reward buf')
        print(self.rew_buf)
        if len(env_ids) > 0:
            self.accumulated_reward[env_ids] = 0
        print('self.accumulated_reward')
        print(self.accumulated_reward)
        # # print('DEBUG PRINTOUTS')
        # # body_height = self.obs_buf[:,2]
        # # up_projection = self.obs_buf[:,29]
        # # heading_projection = self.obs_buf[:, 30] 
        # # heading_reward = self.heading_weight * heading_projection    
        # # # aligning up axis and environment
        # # up_reward = torch.zeros_like(heading_reward)
        # # up_reward = torch.where(up_projection > 0.93, up_reward + self.up_weight, up_reward)
        # # # reward for duration of staying alive
        # # progress_reward = self.potentials - self.prev_potentials
        # # total_reward = progress_reward + up_reward + heading_reward]
        # xtream_rewards = torch.abs(self.rew_buf) > 5
        # # print('ProgressReward[3] : {} = {} - {}'.format(progress_reward[3], self.potentials[3], self.prev_potentials[3]))
        # # print('EnvReset[3], GoalReset[3] : {}, {}'.format(self.reset_buf[3], self.goal_reset[3]))
        # # print('Bot Pos, Goal Pos = {}, {}'.format(self.tensebot_pos[3,:], self.goal_pos[3,:]))
        # if(torch.any(xtream_rewards)):
        #     print('XTREAM REWARD DETECTED')
        #     xtream_idx = xtream_rewards.nonzero().cpu().detach().numpy()
        #     print("xtream index = {}".format(xtream_idx))
        #     print(self.rew_buf[xtream_idx])
        #     print('Progress Reward : {} = {} - {}'.format(progress_reward[xtream_idx], self.potentials[xtream_idx], self.prev_potentials[xtream_idx]))
        #     print('EnvReset, GoalReset : {},{}'.format(self.reset_buf[xtream_idx], self.goal_reset[xtream_idx]))
        #     time.sleep(10)
        #     print()
        # # print('{:.2f} = {:.2f} + {:.2f} + {:.2f}'.format(total_reward[0], heading_reward[0], up_reward[0], progress_reward[0]))

        # # print(' self.reset_buf')
        # # print( self.reset_buf)
        #     # tmp_progress_reward = self.potentials - self.prev_potentials
        #     # if( np.abs(tmp_progress_reward[0].cpu().detach().numpy()) > 1):
        #     #     print('{} : {} : {}'.format(tmp_progress_reward[0], self.potentials[0], self.prev_potentials[0]))
        #     #     time.sleep(1)
        # # tmp_height_reward = self.obs_buf[:,0]
        # # tmp_heading_reward = self.rew_buf - tmp_progress_reward
        # # self.plot_buffer.append((tmp_progress_reward.cpu().detach().numpy(),
        # #                         tmp_height_reward.cpu().detach().numpy(),
        # #                         tmp_heading_reward.cpu().detach().numpy()))
        

#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_tensebot_reward(
    tensebot_pos,
    goal_pos,
    reset_buf,
    progress_buf,
    potentials,
    prev_potentials,
    max_episode_length,
    goal_threshold):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float) -> Tuple[Tensor, Tensor, Tensor]

    # reward for duration of staying alive
    progress_reward = potentials - prev_potentials

    total_reward = progress_reward

    # reset agents
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    # tensebot_avg_pos = torch.mean(tensebot_pos, dim=1)
    tensebot_avg_pos = tensebot_pos[:,0,:]
    distance_to_goal = torch.norm(tensebot_avg_pos - goal_pos, dim=-1)
    goal_reached = torch.where(distance_to_goal < goal_threshold, 1, 0)
    goal_reset = torch.where(goal_reached==1, 1, 0)

    return total_reward, reset, goal_reset


@torch.jit.script
def compute_tensebot_observations(tensebot_pos,                 #Tensor
                                 tensebot_ori,                  #Tensor
                                 tensebot_linvel,               #Tensor
                                 tensebot_angvel,               #Tensor
                                 goal_pos,                      #Tensor
                                 potentials,                    #Tensor
                                 actions,                       #Tensor
                                 dt                             #float
                                 ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor]
    # tensebot_avg_pos = torch.mean(tensebot_pos, dim=1)
    tensebot_avg_pos = tensebot_pos[:,0,:]
    to_target = goal_pos - tensebot_avg_pos
    to_target[:, 2] = 0.0
    to_target_norm = torch.div(to_target, torch.unsqueeze(torch.norm(to_target, p=2, dim=-1),1).repeat(1,3))

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    # torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
    #     tensebot_ori, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

    # vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
    #     torso_quat, tensebot_linvel, tensebot_angvel, goal_pos, tensebot_pos)
    
    # obs_buf shapes: (53)
    # obs_buf[0:39] = Rod State x 3 : Pos(3), Ori(4), LinVel(3), AngVel(3)
    # obs_buf[39:42] = Goal Pos : Pos(3)
    # obs_buf[42:45] = vector to goal (3) 
    # obs_buf[45:53] = actions : Spring Length Multipliers (9)
    obs = torch.cat((tensebot_pos[:,0,:], tensebot_ori[:,0,:], tensebot_linvel[:,0,:], tensebot_angvel[:,0,:], 
                     tensebot_pos[:,1,:], tensebot_ori[:,1,:], tensebot_linvel[:,1,:], tensebot_angvel[:,1,:], 
                     tensebot_pos[:,2,:], tensebot_ori[:,2,:], tensebot_linvel[:,2,:], tensebot_angvel[:,2,:], 
                     goal_pos, to_target_norm, actions), dim=-1)

    return obs, potentials, prev_potentials_new