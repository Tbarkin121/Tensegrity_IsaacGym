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


class TenseBot6(VecTask):
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

        # Spring Length Mulitpliers (n)
        self.cfg["env"]["numActions"] = len(self.cfg["tensegrityParams"]["connections"])

        # obs_buf shapes: (81)
        # obs_buf[0:9] = Avg Pos (3), Avg LinVel (3), Avg AngVel (3) : 9
        # obs_buf[9:33] = Rod x N Ori(4) : 24
        # obs_buf[33:51] = Contact Bools (3 per support) (18)
        # obs_buf[51:54] = Goal Pos : Pos(3)
        # obs_buf[54:57] = vector to goal (3) 
        # obs_buf[57:81] = actions : Spring Length Multipliers (24 for T6)
        self.cfg["env"]["numObservations"] = 33 + self.cfg["env"]["numActions"]

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
        
        # get gym GPU root state tensor
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        print('root_state')
        print(self.root_states.cpu().detach().numpy())
        print(self.root_states.shape)
        print('num_envs {}, num_actors {}'.format(self.num_envs, self.num_actors))

        self.tensebot_pos = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0:6, 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.tensebot_ori = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0:6, 3:7] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.tensebot_linvel = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0:6, 7:10] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.tensebot_angvel = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0:6, 10:13] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.goal_pos = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 6, 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.tensebot_root_state = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0:6, :]
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
        
        contact_state_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_tensor = gymtorch.wrap_tensor(contact_state_tensor)   
        self.body_contact_force = self.contact_tensor.view(self.num_envs, self.num_bodies, 3)[:, 0:-1, :]
        
        tensebot_avg_pos = torch.mean(self.tensebot_pos, dim=1)
        to_target = self.goal_pos - tensebot_avg_pos
        to_target[:, 2] = 0.0
        self.potentials = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.prev_potentials = self.potentials.clone()
        
        self.goal_reset = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        goal_ids = self.goal_reset.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_ids) > 0:
            self.reset_goal(goal_ids)

        # Measurements for rewards
        # self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        # self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        # self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        # self.basis_vec0 = self.heading_vec.clone()
        # self.basis_vec1 = self.up_vec.clone()
        
        self.frame_count = 0
        self.plot_buffer = []
        self.accumulated_reward = torch.zeros_like(self.rew_buf)

        camOffset = gymapi.Vec3(0, -1.5, 0.25)
        camTarget = gymapi.Vec3(self.tensebot_pos[0, 0, 0],self.tensebot_pos[0, 0, 1],self.tensebot_pos[0, 0, 2])
        self.gym.viewer_camera_look_at(self.viewer, None, camOffset+camTarget, camTarget)
    
    def create_tensegrity_connections(self):
        # Defining the spring connection List
        self.connection_list = []
        self.connection_matrix = torch.zeros(self.num_bodies, len(self.cfg["tensegrityParams"]["connections"]), device=self.device)
        idx = 0
        for connection in self.cfg["tensegrityParams"]["connections"]:
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
            idx += 1
        self.connection_list = torch.tensor(self.connection_list, device=self.device)


    def create_tensegrity(self, env, asset, collision_group, collision_filter):
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
            SupportHandle = self.gym.create_actor(env, asset, initial_pose, name, collision_group, collision_filter, 0)    
            # Set the coloring (Fun)
            # rand_colors = torch.rand((3), device=self.device)
            self.gym.set_rigid_body_color(env, SupportHandle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))
            self.gym.set_rigid_body_color(env, SupportHandle, 1, gymapi.MESH_VISUAL, gymapi.Vec3(0.06, 0.97, 0.38))
            self.gym.set_rigid_body_color(env, SupportHandle, 2, gymapi.MESH_VISUAL, gymapi.Vec3(0.38, 0.06, 0.97))
            # Add to Tensegrity list for access later
            assembly_handles.append(SupportHandle)                
        return assembly_handles

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

        support_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(support_asset)

        goal_asset = self.gym.create_sphere(self.sim, 0.025)
        self.num_bodies = self.gym.get_asset_rigid_body_count(support_asset)*self.cfg["tensegrityParams"]["numSupports"] + self.gym.get_asset_rigid_body_count(goal_asset) #3 rod assemblies per tensebot
        # self.num_actor = get_sim_actor_count
        
        self.rod_handles = []
        self.tensebot_handles = []
        self.goal_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        self.create_tensegrity_connections()
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            # Creates a tensegrity bot for an environment
            # Returns a list of handles for the support actors
            self.tensebot_handles.append(self.create_tensegrity(env_ptr, support_asset, collision_group=i, collision_filter=0))
            self.envs.append(env_ptr)

            # Set Up the Goal Actor
            goal_pose = gymapi.Transform()
            goal_pose.p.y = self.goal_dist
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_pose, "goal", i, 1, 1)
            self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.8, 0.2))
            self.goal_handles.append(goal_handle)

            
        self.num_actors = self.gym.get_actor_count(self.envs[0])
        self.body_dict = self.gym.get_actor_rigid_body_dict(env_ptr, self.tensebot_handles[0][0])
        self.joint_dict = self.gym.get_actor_joint_dict(env_ptr, self.tensebot_handles[0][0])

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
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # print('self.root_state')
        # print(self.root_states[0,:])
        # print(self.root_states.shape)
        # time.sleep(1)
        self.obs_buf[:], self.potentials[:], self.prev_potentials[:] = compute_tensebot_observations(
            self.tensebot_pos,
            self.tensebot_ori,
            self.tensebot_linvel,
            self.tensebot_angvel,
            self.body_contact_force,
            self.goal_pos, 
            self.potentials,
            self.actions, 
            self.dt)
        return self.obs_buf

    def reset_idx(self, env_ids):
        # print('Resetting IDX! Env_IDs = {}'.format(env_ids))
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)*self.num_actors
        env_ids_int32 = torch.cat((env_ids_int32, env_ids_int32+1, env_ids_int32+2, env_ids_int32+3, env_ids_int32+4, env_ids_int32+5))

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
        # print('reset_goal')
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # print('Resetting Goals! Env_IDs = {}'.format(env_ids))
        # print('Old Goal Position = {}'.format(self.goal_pos))

        env_ids_int32 = env_ids.to(dtype=torch.int32)*self.num_actors + 6
        goal_pos_update = torch_rand_float(-self.goal_dist, self.goal_dist, (len(env_ids), 3), device=self.device)
        # goal_pos_update[:,0] = 1000.0
        # goal_pos_update[:,1] = 0.0
        goal_pos_update[:,2] = 0.025
        self.goal_pos[env_ids, :] = goal_pos_update
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.root_states),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # self.gym.refresh_actor_root_state_tensor(self.sim)
        tensebot_avg_pos = torch.mean(self.tensebot_pos, dim=1)
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
        # This might need a low pass filter
        spring_length_multiplier = actions*self.cfg["tensegrityParams"]["spring_length_change_factor"] + 1 
        # spring_length_multiplier = torch.ones((self.num_envs, len(self.cfg["tensegrityParams"]["connections"])), device=self.device)

        # forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)        
        forces = torch.zeros_like(self.rb_pos, device=self.device, dtype=torch.float)
        force_positions = self.rb_pos.clone()

        num_lines = len(self.connection_list)
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
        spring_lengths = self.connection_list[:, 2]*spring_length_multiplier

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
        # print(forces.shape)
        # time.sleep(1)
        forces = torch.matmul(torch.unsqueeze(self.connection_matrix, 0).repeat(self.num_envs,1,1), -applied_forces + damping_forces)
        forces = torch.nan_to_num(forces, nan=0.0)
        # print(forces)
        # time.sleep(1)
        
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
                                 tensebot_contact,               #Tensor
                                 goal_pos,                      #Tensor
                                 potentials,                    #Tensor
                                 actions,                       #Tensor
                                 dt                             #float
                                 ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor]
    # tensebot_avg_pos = torch.mean(tensebot_pos, dim=1)
    avg_pos = torch.mean(tensebot_pos, dim=1)
    avg_linvel = torch.mean(tensebot_linvel, dim=1)
    avg_angvel = torch.mean(tensebot_angvel, dim=1)
    to_target = goal_pos - avg_pos
    to_target[:, 2] = 0.0
    to_target_norm = torch.div(to_target, torch.unsqueeze(torch.norm(to_target, p=2, dim=-1),1).repeat(1,3))

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    # torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
    #     tensebot_ori, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

    # vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
    #     torso_quat, tensebot_linvel, tensebot_angvel, goal_pos, tensebot_pos)
    
    contact = torch.where(torch.sum(tensebot_contact, dim=-1) != 0, 1, 0)
    # obs_buf shapes: (81)
    # obs_buf[0:9] = Avg Pos (3), Avg LinVel (3), Avg AngVel (3) : 9
    # obs_buf[9:33] = Rod x N Ori(4) : 24
    # obs_buf[33:51] = Contact Bools (3 per support) (18)
    # obs_buf[51:54] = Goal Pos : Pos(3)
    # obs_buf[54:57] = vector to goal (3) 
    # obs_buf[39:81] = actions : Spring Length Multipliers (24 for T6)
    
    # obs = torch.cat((avg_pos, avg_linvel, avg_angvel, 
    #                  tensebot_ori[:,0,:], tensebot_ori[:,1,:], tensebot_ori[:,2,:],
    #                  tensebot_ori[:,3,:], tensebot_ori[:,4,:], tensebot_ori[:,5,:], 
    #                  contact, goal_pos, to_target_norm, actions), dim=-1)

        # obs_buf shapes: (81)
    # obs_buf[0:9] = Avg Pos (3), Avg LinVel (3), Avg AngVel (3) : 9
    # obs_buf[9:27] = Contact Bools (3 per support) (18)
    # obs_buf[27:30] = Goal Pos : Pos(3)
    # obs_buf[33:36] = vector to goal (3) 
    # obs_buf[36:60] = actions : Spring Length Multipliers (24 for T6)
    obs = torch.cat((avg_pos, avg_linvel, avg_angvel, 
                     contact, goal_pos, to_target_norm, actions), dim=-1)

    return obs, potentials, prev_potentials_new
    