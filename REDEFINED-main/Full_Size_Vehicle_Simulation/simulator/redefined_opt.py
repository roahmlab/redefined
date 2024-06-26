import torch
import torch.nn as nn
import cyipopt
import numpy as np
from scipy.io import loadmat, matlab
from torch.autograd import grad
import os
import json
import time

FRS_TYPE_SPD = 0
FRS_TYPE_DIR = 1
FRS_TYPE_LAN = 2
NUMERICAL_EPS = 1e-8

class distance(nn.Module):
    def __init__(self):
        super(distance, self).__init__()

    def forward(self, p, s1, s2, difference_cache, squared_distance_cache):                
        t_hat = torch.sum((p - s1) * difference_cache, dim=1, keepdim=True) / squared_distance_cache
        t_star = t_hat.clamp(0.,1.)
        s = s1 + t_star * difference_cache
        distance = torch.linalg.norm(p - s, dim=1)
        
        return distance
    
class sign(nn.Module):
    def __init__(self):
        super(sign, self).__init__()

    def forward(self, points, s1, s2, vertices_range, vertices_indices, min_y_cache, max_y_cache, x_check_cache):
        min_y = min_y_cache
        y_check = torch.logical_and(
            points[:, 1] >= min_y, points[:, 1] < max_y_cache)
        x_check = s2[:, 0] + (points[:, 1] - s2[:, 1]) * x_check_cache
        x_check = x_check >= points[:, 0]
        is_intersect = torch.logical_and(y_check, x_check).int()
        is_inside = is_intersect.cumsum(dim=0)[vertices_indices]

        is_inside[1:] = is_inside[1:] - is_inside[:-1]
        negative_indces = torch.where(is_inside == 1)[0]
        if torch.numel(negative_indces) > 0:
            range_indices = vertices_range[negative_indces]
            val = torch.tensor(
                [[1, -1]], device=points.device).repeat(range_indices.shape[0], 1)
            end_indices = range_indices[:-1, 1]
            duplicate_indices = end_indices[range_indices[:, 0][1:] == end_indices]
            
            inside_indices_mask = torch.zeros(points[:, 0].shape[0]+1, dtype=torch.long, device=points.device).scatter_(
                0, range_indices.view(-1), val.view(-1))
            inside_indices_mask[duplicate_indices] = 0
            inside_indices_mask = inside_indices_mask.cumsum(0).bool()[:-1]                
        else:
            inside_indices_mask = torch.zeros_like(points[:, 0], dtype=torch.bool)
            
        return inside_indices_mask
    

class distance_gradient(nn.Module):
    def __init__(self):
        super(distance_gradient, self).__init__()

    def forward(self, p, s1, s2):
        s2_minus_s1_square = torch.square(s2 - s1)
        s2_minus_s1_square_sum = torch.sum(s2_minus_s1_square, dim=1, keepdim=True) + NUMERICAL_EPS
        t_hat = torch.sum((p - s1) * (s2 - s1), dim=1, keepdim=True) / s2_minus_s1_square_sum
        t_star = t_hat.clamp(0.,1.)
        s = s1 + t_star * (s2 - s1)
        distance = torch.linalg.norm(p - s, dim=1, keepdim=True) + NUMERICAL_EPS
        coeff = torch.logical_and((t_hat < 1.), (t_hat > 0.)).float()
        
        dsdp = coeff * s2_minus_s1_square / s2_minus_s1_square_sum
        gradient = (p - s + (s - p) * dsdp * 0.) / distance
        
        return gradient        
    

def prepare_zono_batch(batch_zonos, device, ordered=False):
    batch_gen = batch_zonos[:, 1:, :]
    batch_gen[batch_gen[:, :, 1] < 0] *= -1

    # sort generators in counter-clockwise order
    if not ordered:
        angles = torch.atan2(batch_gen[:, :, 1], batch_gen[:, :, 0])
        # angles of padding generators are set to inf so that they are sorted to the end
        idx_padding_gen = (batch_gen == 0).all(dim=-1)
        angles[idx_padding_gen] = torch.inf

        _, indices = torch.sort(angles, dim=1)
        batch_gen = batch_gen.gather(1, indices.unsqueeze(-1).repeat(1, 1, 2))

    batch_zonos[:, 1:, :] = batch_gen
    return batch_zonos

def batch_zono2d_vertices(zonos, n_generators_list):
    """Return the vertices of a batch of 2d zonotopes.
    :param zonos: batch of zonotopes, 2d, can be unsimplified as a list of zonotopes
    :param n_generators_list: list of generator lengths
    :return (vertices, n_generators):
    """
    device = zonos.device
    n_generators: int = zonos.shape[1] - 1
    n_zonos: int = zonos.shape[0]
    centers = zonos[:, 0, :].view(n_zonos, 1, 2)
    generators = zonos[:, 1:, :]

    # coefficients of generators
    coeffs: torch.Tensor = torch.triu(
        torch.ones(n_generators, n_generators, device=device)
    ) + torch.tril(
        -1 * torch.ones(n_generators, n_generators, device=device), diagonal=-1
    )
    
    coeffs = coeffs.expand(n_zonos, *coeffs.shape)
    # -> n * m * 2
    # print(f"Coeff shape {coeffs.shape}, generators shape {generators.shape}")
    half_vertices = torch.bmm(coeffs, generators)

    return (
        torch.cat([half_vertices + centers, -half_vertices + centers], dim=1),
        n_generators_list,
    )


class REDEFINED_NLP():
    def __init__(self, FRS_data_list, obstacles_is_mirrored_list, agent_state, obstacles, mirrored_obstacles, goal, x_des_local, x_des_mirror, sdf_net, sign_net=None, gradient_net=None, num_problems=8, device='cuda', use_sum_as_obj=False, distance_buffer=0.01):
        self.x, self.y, self.h, self.u0, self.v0, self.r0 = agent_state
        self.x_des_local = x_des_local
        self.x_des_mirror = x_des_mirror
        self.num_problems = num_problems
        self.sdf_net = sdf_net
        self.sign_net = sign_net
        self.gradient_net = gradient_net
        
        self.device = device
        self.prev_p = np.ones((num_problems)) * np.nan
        self.best_p = None
        self.best_cost = np.ones((num_problems)) * np.inf
        self.curr_cost = np.ones((num_problems)) * np.inf
        self.use_sum_as_obj = use_sum_as_obj
        self.distance_buffer = distance_buffer
        self.num_obstacles = obstacles.shape[0]
        self.goal = goal.view(num_problems,3).to(self.device)
        self.obstacles_is_mirrored_list = obstacles_is_mirrored_list
        
        ### used for statistics ###
        self.num_constraint_calls = 0
        self.num_jacobian_calls = 0
        self.num_cost_calls = 0
        self.num_gradient_calls = 0
        
    def init_problem(self, FRS_data_list, obstacles_is_mirrored_list, obstacles, mirrored_obstacles):
        init_start = time.time()
        self.original_obstacles = obstacles
        problem_num_timesteps = []
    
        all_problem_total_timesteps = 0
        all_problem_obstacles = []
        all_problem_frs_zonotopes = []
        all_problem_num_generators = []
        all_problem_num_vertices_each_timestep_each_obs = []
        all_problem_num_vertices = []
        
        unmirrored_obstacles_range = []
        unmirrored_t_duration = []
        unmirrored_t_start = []
        mirrored_obstacles_range = []
        mirrored_t_duration = []
        mirrored_t_start = []
        max_num_generators = 0
        
        all_problem_brake_indices = []
        self.all_problem_cost_func_center = []
        self.all_problem_cost_func_g_p = []
        all_problem_c_h = []
        self.all_problem_c_p = []
        all_problem_c_r0 = []
        all_problem_c_u0 = []
        all_problem_c_v0 = []
        all_problem_g_p = []
        all_problem_g_r0 = []
        all_problem_g_u0 = []
        all_problem_g_v0 = []
        self.all_problem_p_ranges = []
        
        self.all_problem_vertices_indices = []
        
        for i_prob, FRS_data in enumerate(FRS_data_list):
            # keys: ['brake_index', 'c_h', 'c_p', 'c_r0', 'c_u0', 'c_v0', 'frs_zonotopes', 'g_p', 'g_r0', 'g_u0', 'g_v0', 'num_generators', 'p_minmax']
            # in this for loop, we need to get:
            # 0. preapre later variables
            # 1. obstacles and t_duration for swpet volume obstacle
            # 2. for each problem, get vertices num at each timestep for later indexing; similarly track the total num of vertices in each problem
            # 3. remember the max number of generators
            
            ### prepare variables
            all_problem_brake_indices.append(FRS_data['brake_index'])
            all_problem_c_h.append(torch.tensor(FRS_data['c_h'], device=self.device, dtype=torch.float32))
            self.all_problem_c_p.append(torch.tensor(FRS_data['c_p'], device=self.device, dtype=torch.float32))
            all_problem_c_r0.append(torch.tensor(FRS_data['c_r0'], device=self.device, dtype=torch.float32))
            all_problem_c_u0.append(torch.tensor(FRS_data['c_u0'], device=self.device, dtype=torch.float32))
            all_problem_c_v0.append(torch.tensor(FRS_data['c_v0'], device=self.device, dtype=torch.float32))
            all_problem_g_p.append(torch.tensor(FRS_data['g_p'], device=self.device, dtype=torch.float32))
            all_problem_g_r0.append(torch.tensor(FRS_data['g_r0'], device=self.device, dtype=torch.float32))
            all_problem_g_u0.append(torch.tensor(FRS_data['g_u0'], device=self.device, dtype=torch.float32))
            all_problem_g_v0.append(torch.tensor(FRS_data['g_v0'], device=self.device, dtype=torch.float32))
            self.all_problem_p_ranges.append(torch.tensor(FRS_data['p_minmax'], device=self.device, dtype=torch.float32))
            ###
            
            frs_zonotopes = torch.tensor(FRS_data['frs_zonotopes'], device=self.device, dtype=torch.float32)
            all_problem_frs_zonotopes.append(frs_zonotopes)
            prob_num_generators = torch.tensor(FRS_data['num_generators'], device=self.device, dtype=torch.int)
            
            all_problem_num_generators.append(prob_num_generators)
            problem_num_vertices = (prob_num_generators + 3) * 2
            all_problem_num_vertices_each_timestep_each_obs.append(problem_num_vertices)
            all_problem_num_vertices.append(problem_num_vertices.sum() * self.num_obstacles)
            self.all_problem_vertices_indices.append(problem_num_vertices.view(-1,1).repeat(1, self.num_obstacles).view(-1,1).cumsum(0))

            max_num_generators = max(max_num_generators, FRS_data['frs_zonotopes'].shape[1])
                             
            ### prepare obstacle swept volumes
            num_timesteps = frs_zonotopes.shape[0]
            problem_num_timesteps.append(num_timesteps)
            problem_t_duration = torch.tensor(FRS_data['duration_t'], device=self.device, dtype=torch.float32)
            t_start = torch.cat([torch.tensor([0.], device=self.device), problem_t_duration.cumsum(0)[:-1]])
            if not obstacles_is_mirrored_list[i_prob]:
                unmirrored_obstacles_range.append(torch.arange(all_problem_total_timesteps * self.num_obstacles, (all_problem_total_timesteps+num_timesteps) * self.num_obstacles))
                unmirrored_t_duration.append(problem_t_duration)
                unmirrored_t_start.append(t_start)
            else:
                mirrored_obstacles_range.append(torch.arange(all_problem_total_timesteps * self.num_obstacles, (all_problem_total_timesteps+num_timesteps) * self.num_obstacles))
                mirrored_t_duration.append(problem_t_duration)
                mirrored_t_start.append(t_start)
            ###
            
            all_problem_total_timesteps += num_timesteps
        
        ### Prepare obstacles to vertices. 
        all_problem_obstacles = torch.zeros((all_problem_total_timesteps * self.num_obstacles, 4, 2), device=self.device)
        if len(unmirrored_obstacles_range) > 0:
            all_problem_obstacles[torch.cat(unmirrored_obstacles_range)] = self.prepare_obstacles_swept_volume(obstacles=obstacles, t_duration=torch.cat(unmirrored_t_duration, dim=0), t_start=torch.cat(unmirrored_t_start, dim=0))
        if len(mirrored_obstacles_range) > 0:
            all_problem_obstacles[torch.cat(mirrored_obstacles_range)] = self.prepare_obstacles_swept_volume(obstacles=mirrored_obstacles, t_duration=torch.cat(mirrored_t_duration, dim=0), t_start=torch.cat(mirrored_t_start, dim=0))
        all_problem_frs_zonotopes_temp= torch.cat([torch.cat((f, torch.zeros(f.shape[0], max_num_generators-f.shape[1]+1, 2, device=self.device)), dim=1) for f in all_problem_frs_zonotopes], dim=0)
        all_problem_buffered_obstacles = torch.cat([all_problem_obstacles, all_problem_frs_zonotopes_temp[:, 1:, :].repeat(1, self.num_obstacles, 1).view(all_problem_total_timesteps * self.num_obstacles, max_num_generators, 2)], dim=1)
        all_problem_num_generators = (torch.cat(all_problem_num_generators, dim=0) + 3).repeat(1, self.num_obstacles).view(-1)
        prepared_zonotopes = prepare_zono_batch(all_problem_buffered_obstacles, device=self.device,ordered=False)
        all_problem_vertices, _ = batch_zono2d_vertices(prepared_zonotopes, all_problem_num_generators)
        # all problem vertices: (total num timesteps * num obs, max_num_generators, 2)
        ### Note that all_problem vertices contain padded vertices. We will need to eliminate those.
        
        # Get vertice indices, then get vertices
        all_problem_num_vertices_each_timestep_each_obs_temp = torch.cat(all_problem_num_vertices_each_timestep_each_obs)
        vertices_start_indices_first_half = torch.arange(0, all_problem_total_timesteps * self.num_obstacles, device=self.device) * (max_num_generators+3) * 2
        vertices_end_indices_first_half = vertices_start_indices_first_half.view(all_problem_total_timesteps, self.num_obstacles) + all_problem_num_vertices_each_timestep_each_obs_temp.view(-1,1) // 2
        vertices_end_indices_second_half = vertices_end_indices_first_half + max_num_generators + 3
        vertices_start_indices_second_half = vertices_end_indices_second_half.view(all_problem_total_timesteps, self.num_obstacles) - all_problem_num_vertices_each_timestep_each_obs_temp.view(-1,1) // 2 
        
        vertices_start_indices = torch.cat((vertices_start_indices_first_half.view(-1,1), vertices_start_indices_second_half.view(-1,1)), dim=1).view(-1,1)
        vertices_end_indices = torch.cat((vertices_end_indices_first_half.view(-1,1), vertices_end_indices_second_half.view(-1,1)), dim=1).view(-1,1)
        taken_indices_range = torch.cat((vertices_start_indices, vertices_end_indices), dim=1)

        all_problem_vertices = all_problem_vertices.view(-1,2)
        val = torch.tensor([[1, -1]], device=self.device).repeat(taken_indices_range.shape[0], 1)
        taken_indices_mask = torch.zeros(all_problem_vertices.shape[0]+1, dtype=torch.long, device=self.device).scatter_(0, taken_indices_range.view(-1), val.view(-1)).cumsum(0).bool()[:-1]
        
        s1_taken_vertices = all_problem_vertices[taken_indices_mask]
        s2_taken_vertices = all_problem_vertices[1:][taken_indices_mask[:-1]]
        ###
        
        ### Set inputs p, s1, s2. Note that we also pad p, s1, s2 here.
        self.max_num_vertices_each_problem = max(all_problem_num_vertices)
        padded_frs_center_points = []
        padded_g_p = []
        padded_s1_vertices = []
        padded_s2_vertices = []
        num_done_vertices = 0
        
        for i_prob, frs_zonotopes_each_problem in enumerate(all_problem_frs_zonotopes):
            num_vertices_each_timestep = all_problem_num_vertices_each_timestep_each_obs[i_prob]
            num_vertices_this_problem = all_problem_num_vertices[i_prob]
            frs_centers = frs_zonotopes_each_problem[:, 0, :]
            g_u0 = all_problem_g_u0[i_prob]
            g_v0 = all_problem_g_v0[i_prob]
            g_r0 = all_problem_g_r0[i_prob]
            g_p = all_problem_g_p[i_prob]
            brake_index = all_problem_brake_indices[i_prob]
            u0_base = (self.u0 - all_problem_c_u0[i_prob]) / g_u0[:,-1:] * g_u0
            v0_base = (self.v0 - all_problem_c_v0[i_prob]) / g_v0[:,-1:] * g_v0
            r0_base = (self.r0 - all_problem_c_r0[i_prob]) / g_r0[:,-1:] * g_r0
            frs_centers += u0_base[:,:2] + v0_base[:,:2] + r0_base[:,:2]
            cost_func_h = all_problem_c_h[i_prob] +  u0_base[brake_index,2:3] + v0_base[brake_index,2:3] + r0_base[brake_index,2:3]            
            self.all_problem_cost_func_center.append(torch.cat([frs_centers[brake_index], cost_func_h]))
            self.all_problem_cost_func_g_p.append(g_p[brake_index])
            g_p = g_p[:, [0,1,3]]
            padded_frs_center_points.append(frs_centers.repeat_interleave(repeats=num_vertices_each_timestep * self.num_obstacles, dim=0).view(-1,2))
            # pad frs center with -1
            padded_frs_center_points.append(torch.ones(self.max_num_vertices_each_problem - num_vertices_this_problem, 2, device=self.device) * -10000)
            padded_g_p.append(g_p.repeat_interleave(repeats=num_vertices_each_timestep * self.num_obstacles, dim=0).view(-1,3))
            padding_g_p = torch.zeros(self.max_num_vertices_each_problem - num_vertices_this_problem, 3, device=self.device)
            padding_g_p[:,-1] = 1
            padded_g_p.append(padding_g_p)
            # pad s1, s2 with meaningless large numbers
            padded_s1_vertices.append(s1_taken_vertices[num_done_vertices:num_done_vertices+num_vertices_this_problem])
            padded_s1_vertices.append(torch.ones(self.max_num_vertices_each_problem - num_vertices_this_problem, 2, device=self.device) * 10000)
            padded_s2_vertices.append(s2_taken_vertices[num_done_vertices:num_done_vertices+num_vertices_this_problem])
            padded_s2_vertices.append(torch.ones(self.max_num_vertices_each_problem - num_vertices_this_problem, 2, device=self.device) * 10010)
            num_done_vertices += num_vertices_this_problem
        
        self.frs_center_points = torch.cat(padded_frs_center_points, dim=0)
        g_p = torch.cat(padded_g_p, dim=0)
        self.g_p_multiplier = g_p[:, :2] / g_p[:, 2:3]
        self.all_problem_c_p = torch.stack(self.all_problem_c_p).view(-1,1)
        self.s1 = torch.cat(padded_s1_vertices, dim=0)
        self.s2 = torch.cat(padded_s2_vertices, dim=0)
        self.all_problem_num_vertices_each_timestep_each_obs = all_problem_num_vertices_each_timestep_each_obs
        self.all_problem_cost_func_center = torch.stack(self.all_problem_cost_func_center)
        self.all_problem_cost_func_g_p = torch.stack(self.all_problem_cost_func_g_p)
        
        self.all_problem_num_vertices = all_problem_num_vertices
        # now frs_center_points is a tensor of shape (max_num_vertices_each_problem * num_problems, 2). 
        
        # A few more processing for later computations
        self.difference_cache = self.s2 - self.s1
        self.squared_distance_cache = (torch.sum(torch.square(self.difference_cache), dim=1, keepdim=True) + NUMERICAL_EPS)
        self.min_y_cache = torch.min(self.s1[:, 1], self.s2[:, 1])
        self.max_y_cache = self.s1[:, 1] + self.s2[:, 1] - self.min_y_cache
        self.x_check_cache = (self.s1[:, 0] - self.s2[:, 0]) / (self.s1[:, 1] - self.s2[:, 1])

        return time.time() - init_start
    
    def remove_infeasible_bins(self, FRS_types, num_samples=10):
        remove_start = time.time()
        feasible_bins = torch.zeros(self.num_problems, device=self.device).bool()
        feasible_ps = torch.zeros(self.num_problems, device=self.device)
        p_ranges = torch.stack(self.all_problem_p_ranges)
        sampled_p = p_ranges[:,0:1] + (p_ranges[:,1:] - p_ranges[:,0:1]) / num_samples * torch.arange(0, num_samples, device=self.device)

        ### SPD CHANGE ###
        if FRS_TYPE_SPD in FRS_types:
            vertices_indices = torch.cat([self.all_problem_vertices_indices[i] + i * self.max_num_vertices_each_problem for i in range(self.num_problems) if FRS_types[i]==FRS_TYPE_SPD], dim=0)
            vertices_range_right = vertices_indices
            vertices_range_left = torch.cat([torch.cat((torch.tensor([[i * self.max_num_vertices_each_problem]], device=self.device), (self.all_problem_vertices_indices[i] + i * self.max_num_vertices_each_problem)[:-1]), dim=0) for i in range(self.num_problems) if FRS_types[i]==FRS_TYPE_SPD], dim=0)
            vertices_indices = vertices_indices - 1
            vertices_range = torch.cat((vertices_range_left, vertices_range_right), dim=1)
            spd_bin_mask = (torch.tensor(FRS_types, device=self.device, dtype=torch.int) == FRS_TYPE_SPD)
            num_spd_bins = spd_bin_mask.sum()
            g_p_multiplier = self.g_p_multiplier.view(self.num_problems, -1, 2)[spd_bin_mask].view(num_spd_bins, -1, 2)
            p = sampled_p[spd_bin_mask][:, 0:1]
            frs_center_points = self.frs_center_points.view(self.num_problems, -1, 2)[spd_bin_mask].view(-1, 2) + ((p - self.all_problem_c_p[spd_bin_mask]).view(num_spd_bins, 1, 1) * g_p_multiplier).view(-1,2)
            s1 = self.s1.view(self.num_problems, -1, 2)[spd_bin_mask].view(-1,2)
            s2 = self.s2.view(self.num_problems, -1, 2)[spd_bin_mask].view(-1,2)
            difference_cache = self.difference_cache.view(self.num_problems, -1, 2)[spd_bin_mask].view(-1,2)
            squared_distance_cache = self.squared_distance_cache.view(self.num_problems, -1, 2)[spd_bin_mask].view(-1,1)
            distances = self.sdf_net(frs_center_points, s1, s2, difference_cache, squared_distance_cache)
            min_y_cache = self.min_y_cache.view(self.num_problems, -1)[spd_bin_mask].view(-1)
            max_y_cache = self.max_y_cache.view(self.num_problems, -1)[spd_bin_mask].view(-1)
            x_check_cache = self.x_check_cache.view(self.num_problems, -1)[spd_bin_mask].view(-1)
            inside_indices_mask = self.sign_net(frs_center_points, s1, s2, vertices_range, vertices_indices, min_y_cache, max_y_cache, x_check_cache)
            coeff = torch.ones_like(distances)
            coeff[inside_indices_mask] = -1
            distances = (distances * coeff).view(num_spd_bins, -1)
            feasible_problem_mask = torch.logical_not(torch.any(distances <= self.distance_buffer, dim=1))
            feasible_indices = torch.arange(0, self.num_problems, device=self.device)[spd_bin_mask][feasible_problem_mask]
            feasible_bins[feasible_indices] = True
            feasible_ps[feasible_indices] = p[feasible_problem_mask].flatten()
        
        ### OTHER THAN SPD CHANGE (DIR, LAN) ###
        if FRS_TYPE_LAN in FRS_types or FRS_TYPE_DIR in FRS_types:
            bin_mask = torch.logical_not(spd_bin_mask)
            num_non_spd_bins = bin_mask.sum()
            non_spd_indices = [i for i in range(self.num_problems) if FRS_types[i]!=FRS_TYPE_SPD]
            vertices_indices = torch.cat([self.all_problem_vertices_indices[problem_index] + i * self.max_num_vertices_each_problem for i,problem_index in enumerate(non_spd_indices)], dim=0)
            vertices_range_right = vertices_indices
            vertices_range_left = torch.cat([torch.cat((torch.tensor([[i * self.max_num_vertices_each_problem]], device=self.device), (self.all_problem_vertices_indices[problem_index] + i * self.max_num_vertices_each_problem)[:-1]), dim=0) for i, problem_index in enumerate(non_spd_indices)], dim=0)
            vertices_indices = vertices_indices - 1
            vertices_range = torch.cat((vertices_range_left, vertices_range_right), dim=1)
            g_p_multiplier = self.g_p_multiplier.view(self.num_problems, -1, 2)[bin_mask]
            base_frs_center_points = self.frs_center_points.view(self.num_problems, -1, 2)[bin_mask].view(num_non_spd_bins, -1, 2)
            s1 = self.s1.view(self.num_problems, -1, 2)[bin_mask].view(-1,2)
            s2 = self.s2.view(self.num_problems, -1, 2)[bin_mask].view(-1,2)
            difference_cache = self.difference_cache.view(self.num_problems, -1, 2)[bin_mask].view(-1,2)
            squared_distance_cache = self.squared_distance_cache.view(self.num_problems, -1, 2)[bin_mask].view(-1,1)
            min_y_cache = self.min_y_cache.view(self.num_problems, -1)[bin_mask].view(-1)
            max_y_cache = self.max_y_cache.view(self.num_problems, -1)[bin_mask].view(-1)
            x_check_cache = self.x_check_cache.view(self.num_problems, -1)[bin_mask].view(-1)
            
            for i_sample in range(num_samples):
                p = sampled_p[bin_mask][:, i_sample:i_sample+1]
                frs_center_points = base_frs_center_points.view(-1,2) + ((p - self.all_problem_c_p[bin_mask]).view(num_non_spd_bins, 1, 1) * g_p_multiplier).view(-1,2)
                distances = self.sdf_net(frs_center_points, s1, s2, difference_cache, squared_distance_cache)
                
                inside_indices_mask = self.sign_net(frs_center_points, s1, s2, vertices_range, vertices_indices, min_y_cache, max_y_cache, x_check_cache)
                coeff = torch.ones_like(distances)
                coeff[inside_indices_mask] = -1
                distances = (distances * coeff).view(num_non_spd_bins, -1)
                feasible_problem_mask = torch.logical_not(torch.any(distances <= self.distance_buffer, dim=1))

                feasible_indices = torch.arange(0, self.num_problems, device=self.device)[bin_mask][feasible_problem_mask]
                feasible_bins[feasible_indices] = True
                feasible_ps[feasible_indices] = p[feasible_problem_mask].flatten()
                if feasible_bins.sum() == self.num_problems:
                    break
                
        if feasible_bins.sum() == 0:
            print("No feasible bins!?")  
            return [], [], [], [], []
        
        feasible_bin_indices = torch.arange(self.num_problems, device=self.device)[feasible_bins].cpu().tolist()
        feasible_ps = feasible_ps[feasible_bins].cpu().numpy()
        feasible_p_ranges = p_ranges[feasible_bins]
        feasible_p_lb = feasible_p_ranges[:,0].cpu().tolist()
        feasible_p_ub = feasible_p_ranges[:,1].cpu().tolist()
        self.best_p = feasible_ps.copy()
        
        ### adjust member variables
        num_vertices_feasible_bins = torch.stack(self.all_problem_num_vertices)[feasible_bins]
        self.max_num_vertices_each_problem = max(num_vertices_feasible_bins)
        
        self.frs_center_points = self.frs_center_points.view(self.num_problems, -1, 2)[feasible_bins, :self.max_num_vertices_each_problem, :]
        self.s1 = self.s1.view(self.num_problems, -1, 2)[feasible_bins, :self.max_num_vertices_each_problem, :].view(-1, 2)
        self.s2 = self.s2.view(self.num_problems, -1, 2)[feasible_bins, :self.max_num_vertices_each_problem, :].view(-1, 2)
        self.g_p_multiplier = self.g_p_multiplier.view(self.num_problems, -1, 2)[feasible_bins, :self.max_num_vertices_each_problem, :]
        
        self.difference_cache = self.difference_cache.view(self.num_problems, -1, 2)[feasible_bins, :self.max_num_vertices_each_problem, :].view(-1,2)
        self.squared_distance_cache = self.squared_distance_cache.view(self.num_problems, -1, 1)[feasible_bins, :self.max_num_vertices_each_problem, :].view(-1,1)
        self.min_y_cache = self.min_y_cache.view(self.num_problems, -1, 1)[feasible_bins, :self.max_num_vertices_each_problem, :].view(-1)
        self.max_y_cache = self.max_y_cache.view(self.num_problems, -1, 1)[feasible_bins, :self.max_num_vertices_each_problem, :].view(-1)
        self.x_check_cache = self.x_check_cache.view(self.num_problems, -1, 1)[feasible_bins, :self.max_num_vertices_each_problem, :].view(-1)

        self.all_problem_cost_func_g_p = self.all_problem_cost_func_g_p[feasible_bins]
        self.all_problem_cost_func_center = self.all_problem_cost_func_center[feasible_bins]
        self.all_problem_c_p = self.all_problem_c_p[feasible_bins]
        self.num_problems = torch.sum(feasible_bins).item()
        self.goal = self.goal[feasible_bin_indices]
                
        vertices_indices = torch.cat([self.all_problem_vertices_indices[prob_index] + i * self.max_num_vertices_each_problem for i, prob_index in enumerate(feasible_bin_indices)], dim=0)
        vertices_range_right = vertices_indices
        vertices_range_left = torch.cat([torch.cat((torch.tensor([[i * self.max_num_vertices_each_problem]], device=self.device), (self.all_problem_vertices_indices[prob_index] + i * self.max_num_vertices_each_problem)[:-1]), dim=0) for i, prob_index in enumerate(feasible_bin_indices)], dim=0)

        self.vertices_indices = vertices_indices - 1
        self.vertices_range = torch.cat((vertices_range_left, vertices_range_right), dim=1)
        self.prev_p = np.ones((self.num_problems)) * np.nan
        self.best_cost = np.ones((self.num_problems)) * np.inf
        self.curr_cost = np.ones((self.num_problems)) * np.inf
                
        return feasible_bin_indices, feasible_ps, feasible_p_lb, feasible_p_ub, time.time() - remove_start
        

    def prepare_obstacles_swept_volume(self, obstacles, t_duration, t_start):
        # process obstacles [[[c], [g1], [g2], [v]], ...] to swept volume format of [[c_sv], [g1], [g2], [g_sv]], ...]
        num_timesteps = t_duration.shape[0]
        num_obstacles = obstacles.shape[0]
        # repeat the obstacle set num_timesteps times for batch computation
        obstacles = obstacles.repeat(num_timesteps, 1, 1)
        # repeat the t_duration set num_obstacles times for batch computation
        t_start = t_start.view(-1,1).repeat(1,num_obstacles).view(num_obstacles * num_timesteps, 1)
        t_duration = t_duration.view(-1,1).repeat(1,num_obstacles).view(num_obstacles * num_timesteps, 1)
        # adjust obstacle center and generator to accound for swept volume
        obstacles[:, 0, :] += obstacles[:, -1, :] * 2 * t_start
        obstacles[:, -1, :] *= t_duration
        obstacles[:, 0, :] += obstacles[:, -1, :]
                
        return obstacles
    
    def set_ipopt_time(self, start_time, max_ipopt_time_allowed=None):
        self.ipopt_start_time = start_time
        if max_ipopt_time_allowed is not None:
            self.ipopt_allowed_time = max_ipopt_time_allowed
        else:
            self.ipopt_allowed_time = None

    def objective(self, p):
        # opt_variable = [p0, p1, ..., pn-1]
        self.compute_objective_and_gradient(p, compute_gradient=False)
        self.num_cost_calls += 1
        return self.obj

    def gradient(self, p):
        # opt_variable = [p0, p1, ..., pn-1]
        self.compute_objective_and_gradient(p, compute_gradient=True)
        self.num_gradient_calls += 1
        return self.grad

    def compute_objective_and_gradient(self, p, compute_gradient=True):
        goal_cost, goal_gradient = self.compute_goal_cost_constraint_and_jacobian(
            p, compute_jacobian=compute_gradient)
        self.curr_cost = goal_cost.detach().cpu().numpy()
        
        with torch.no_grad():
            if self.use_sum_as_obj:
                self.obj = self.curr_cost.sum()
            else:
                min_index = self.curr_cost.argmin()
                self.obj = self.curr_cost[min_index]

            if compute_gradient:
                if self.use_sum_as_obj:
                    self.grad = goal_gradient.detach().cpu().numpy()
                else:
                    self.grad = torch.zeros_like(goal_gradient)
                    self.grad[min_index] = goal_gradient[min_index]
                    self.grad = self.grad.detach().cpu().numpy()
            else:
                self.grad = None

        self.obj = self.obj
        self.grad = self.grad        

    def compute_goal_cost_constraint_and_jacobian(self, p, compute_jacobian=True):
        self.Wh = 10
        self.Wxy = 3
        
        p = torch.tensor(p, requires_grad=compute_jacobian, device=self.device).view(self.num_problems, 1)
        self.xyh = self.all_problem_cost_func_center + (p - self.all_problem_c_p) / \
            self.all_problem_cost_func_g_p[:, -1:] * self.all_problem_cost_func_g_p[:, :-1]
        goal_cost = self.Wh * torch.abs(self.xyh[:, 2:3] - self.goal[:, 2:3]) + \
            self.Wxy * \
            torch.linalg.norm(
                (self.xyh[:, 0:2] - self.goal[:, 0:2]), dim=1, keepdim=True) #  * torch.tensor([1.,1.1], device=self.xyh.device)
        if compute_jacobian:
            goal_jacobian = grad(outputs=goal_cost, inputs=p,
                                 grad_outputs=torch.ones_like(goal_cost))[0].view(-1)
        else:
            goal_jacobian = None

        return goal_cost.view(-1), goal_jacobian

    def compute_distance_constraint_and_jacobian(self, p, compute_jacobian=True):
        p = torch.tensor(p, requires_grad=False, device=self.device).view(self.num_problems, 1)
        with torch.no_grad():
            frs_center_points = self.frs_center_points.view(self.num_problems, -1, 2) + (p - self.all_problem_c_p).view(self.num_problems, 1, 1) * self.g_p_multiplier.view(self.num_problems, -1, 2)
            frs_center_points = frs_center_points.view(-1, 2)
            distances = self.sdf_net(frs_center_points, self.s1, self.s2, self.difference_cache, self.squared_distance_cache)
            inside_indices_mask = self.sign_net(frs_center_points, self.s1, self.s2, self.vertices_range, self.vertices_indices, self.min_y_cache, self.max_y_cache, self.x_check_cache)
            coeff = torch.ones_like(distances)
            coeff[inside_indices_mask] = -1
            distances = (distances * coeff).view(self.num_problems, -1)
            
            range_indices = torch.arange(self.num_problems, device=self.device)
            constraint_indices = torch.argmin(distances.clamp(self.distance_buffer), dim=1)
            constraint_distances = distances[range_indices,constraint_indices]

            if compute_jacobian:
                frs_center_points = frs_center_points.view(self.num_problems, -1, 2)[range_indices,constraint_indices].view(-1,2)
                s1 = self.s1.view(self.num_problems, -1, 2)[range_indices,constraint_indices].view(-1,2)
                s2 = self.s2.view(self.num_problems, -1, 2)[range_indices,constraint_indices].view(-1,2)
                sign = coeff.view(self.num_problems, -1, 1)[range_indices,constraint_indices]
                g_p = self.g_p_multiplier.view(self.num_problems, -1, 2)[range_indices,constraint_indices].view(-1,2)
                gradient_distances = (self.gradient_net(frs_center_points, s1, s2) * sign * g_p).sum(dim=1)
                jacobian_distances = gradient_distances
            else:
                jacobian_distances = None

        return constraint_distances, jacobian_distances

    def constraints(self, p):
        if (self.prev_p != p).any():
            self.num_constraint_calls += 1
            constraint_tensor, _ = self.compute_distance_constraint_and_jacobian(p, compute_jacobian=False)
            self.cons = constraint_tensor.detach().cpu().numpy()
            self.prev_p = p.copy()
            
            curr_time = time.time()
            if self.ipopt_allowed_time is None or (self.ipopt_allowed_time is not None and curr_time - self.ipopt_start_time <= self.ipopt_allowed_time):
                better_choice_mask = np.logical_and(self.curr_cost < self.best_cost, self.cons > self.distance_buffer)
                self.best_cost[better_choice_mask] = self.curr_cost[better_choice_mask]
                self.best_p[better_choice_mask] = p[better_choice_mask]

        return self.cons

    def jacobian(self, p):
        self.num_jacobian_calls += 1
        constraint_tensor, jacobian_tensor = self.compute_distance_constraint_and_jacobian(p, compute_jacobian=True)
        self.cons = constraint_tensor.detach().cpu().numpy()
        curr_time = time.time()
        if self.ipopt_allowed_time is None or (self.ipopt_allowed_time is not None and curr_time - self.ipopt_start_time <= self.ipopt_allowed_time):
            better_choice_mask = np.logical_and(self.curr_cost < self.best_cost, self.cons > self.distance_buffer)
            self.best_cost[better_choice_mask] = self.curr_cost[better_choice_mask]
            self.best_p[better_choice_mask] = p[better_choice_mask]
        self.jac = np.zeros((self.num_problems, self.num_problems))
        np.fill_diagonal(self.jac, jacobian_tensor.detach().cpu().numpy())
        self.prev_p = p.copy()
        return self.jac
    
    #@profile
    def get_final_solution(self, p_opt, feasible_bin_indices, FRS_types, bin_indices_i, bin_indices_j, print_info=False):
        # result_tuple = (pu, py, 1, manuever type, which bin)
        # NOTE: assume every p is feasible
        
        # Get the bin information corresponding to the lowest-cost one
        costs, _ = self.compute_goal_cost_constraint_and_jacobian(p_opt, compute_jacobian=False)
        if print_info:
            print(f"The costs correspinding to each bin is {costs.detach().cpu().tolist()}")
        delta_xyh = self.xyh.detach().cpu().tolist()
        min_cost_index = costs.argmin().item()
        optimized_p = p_opt[min_cost_index].item()
        min_cost_bin_index = feasible_bin_indices[min_cost_index]
        min_cost_FRS_type = FRS_types[min_cost_bin_index]
        min_cost_bin_index_i, min_cost_bin_index_j = bin_indices_i[min_cost_bin_index], bin_indices_j[min_cost_bin_index]
        
        # Adjust the final selected bin.
        min_cost_idx = min_cost_index
        kBigCost = 1.0e10
        cost_vals = costs.detach().cpu().tolist()
        init_manu_type = min_cost_FRS_type
        min_cost_is_lan = (min_cost_FRS_type == FRS_TYPE_LAN)
        min_cost_is_dir = (min_cost_FRS_type == FRS_TYPE_DIR)
        curr_min_cost = cost_vals[min_cost_index]
        kMinGoodHeading = 10.0 * torch.pi / 360.0
        has_okay_heading = (abs(self.h) <= kMinGoodHeading)
        
        if self.obstacles_is_mirrored_list[min_cost_bin_index]:
            init_delta_y = abs(delta_xyh[min_cost_idx][1] - self.x_des_mirror[1])
        else:
            init_delta_y = abs(delta_xyh[min_cost_idx][1] - self.x_des_local[1])
        
        initial_min_cost_close_enough = init_delta_y < 0.9
        init_speed_high_enough_for_lan = (self.u0 >= 20.0)
        wpt_is_close_n_diff_lane = (self.x_des_local[0] < 90) and (abs(self.x_des_local[1]) > 0.8)
        ini_min_cost_close_but_hi_spd_dir = initial_min_cost_close_enough and init_speed_high_enough_for_lan and min_cost_is_dir
        
        if not has_okay_heading:
            for i, i_bin in enumerate(feasible_bin_indices):
                manu_info = FRS_types[i_bin]
                cost = cost_vals[i]
                was_successful = cost < kBigCost # QC NOTE: this shuold be always true given our implementation
                manu_is_dir = (manu_info == FRS_TYPE_DIR)
                if ((not was_successful) or (not manu_is_dir)):
                    continue
                dh = abs(self.h + delta_xyh[i][2]) 
                if ((not min_cost_is_dir) or (dh < curr_min_cost)):
                    min_cost_idx = i
                    min_cost_is_dir = True
                    curr_min_cost = dh
        elif ((not min_cost_is_lan) and ((not initial_min_cost_close_enough) or ini_min_cost_close_but_hi_spd_dir) and (init_speed_high_enough_for_lan or wpt_is_close_n_diff_lane)):
            if print_info:
                print("[LAN] INITIAL VAL NOT CLOSE ENOUGH") 
            for i, i_bin in enumerate(feasible_bin_indices):
                manu_info = FRS_types[i_bin]
                cost = cost_vals[i]
                was_successful = cost < kBigCost # QC NOTE: this shuold be always true given our implementation
                manu_is_lan = (manu_info == FRS_TYPE_LAN)
                if ((not was_successful) or (not manu_is_lan)):
                    continue
                if self.obstacles_is_mirrored_list[i_bin]:
                    dy = abs(delta_xyh[i][1] - self.x_des_mirror[1])
                else:
                    dy = abs(delta_xyh[i][1] - self.x_des_local[1])
                if print_info:
                    print(f"[LAN] [DY]: {dy}" )
                if abs(dy) < 7.0:
                    if print_info:
                        print(f"[LAN] [Choosing DY]: {dy}")
                    if ((not min_cost_is_lan) or (cost < curr_min_cost)):
                        min_cost_idx = i
                        min_cost_is_lan = True
                        curr_min_cost = cost
        
        optimized_p = p_opt[min_cost_idx].item()
        min_cost_bin_index = feasible_bin_indices[min_cost_idx]
        min_cost_FRS_type = FRS_types[min_cost_bin_index]
        min_cost_bin_index_i = bin_indices_i[min_cost_bin_index]
        min_cost_bin_index_j = bin_indices_j[min_cost_bin_index]
        if self.obstacles_is_mirrored_list[min_cost_bin_index]:
            optimized_p *= -1
            
        if min_cost_FRS_type == 0:
            return (optimized_p, 0.0, 0, min_cost_FRS_type + 1, min_cost_bin_index_i, min_cost_bin_index_j), cost_vals
        else:
            return (self.u0, optimized_p, 0, min_cost_FRS_type + 1, min_cost_bin_index_i, min_cost_bin_index_j), cost_vals
        
    
def call_optimization(agent_state = [10.,0.,0.,20.,0.,0.], 
                      bin_indices_i = [29, 26],
                      bin_indices_j = [0, 0], 
                      FRS_types = [2, 2], 
                      way_point = [66, 3.7, 0.],
                      x_des_local = [90., 3.7, 0.],
                      x_des_mirror = [90., -3.7, 0.],
                      num_obstacles = 10,
                      obstacles_are_mirrored = [False, False],
                      obstacles=None, mirrored_obstacles=None, num_samples=10,
                      sdf_net=None, sign_net=None, gradient_net=None, print_info=False, save_optimization_statistics=False,
                      linear_solver='ma57', max_iter=15, max_wall_time=None, save_dir_prefix='', has_time_limit=False
):
    # driver function to call the optimization problem
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load the FRS from file
    FRS_spd_filename = 'frs_spd.npz'
    FRS_lan_filename = 'frs_lan.npz'
    FRS_dir_filename = 'frs_dir.npz'

    FRS_spd = np.load(FRS_spd_filename, allow_pickle=True)['FRS_spd']
    FRS_dir = np.load(FRS_dir_filename, allow_pickle=True)['FRS_dir']
    FRS_lan = np.load(FRS_lan_filename, allow_pickle=True)['FRS_lan']
    FRS = {FRS_TYPE_SPD: FRS_spd, FRS_TYPE_DIR: FRS_dir, FRS_TYPE_LAN: FRS_lan} # 0: spd, 1: dir, 2: lan
              
    num_obstacles = int(num_obstacles)
    obstacles = torch.tensor(obstacles).to(device=device, dtype=torch.float32).view(num_obstacles, 4, 2)
    mirrored_obstacles = torch.tensor(mirrored_obstacles).to(device=device, dtype=torch.float32).view(num_obstacles, 4, 2)

    # load the FRS bin
    num_problems = len(bin_indices_i)
    way_point = torch.tensor(way_point).view(num_problems, 3)
    FRS_data = []
    for i in range(num_problems):
        FRS_data.append(FRS[int(FRS_types[i])][int(bin_indices_i[i])][int(bin_indices_j[i])])
    
    distance_buffer = 0.01
    
    total_start = time.time()
    nlp_obj = REDEFINED_NLP(
        FRS_data_list=FRS_data, obstacles_is_mirrored_list=obstacles_are_mirrored,
        agent_state=agent_state,
        obstacles=obstacles, mirrored_obstacles=mirrored_obstacles,
        goal=way_point,
        x_des_local = x_des_local,
        x_des_mirror = x_des_mirror,
        sdf_net=sdf_net,
        sign_net=sign_net,
        gradient_net=gradient_net,
        num_problems=len(FRS_data),
        device=device,
        use_sum_as_obj=True,
        distance_buffer=distance_buffer,
    )
    preprocessing_time = nlp_obj.init_problem(FRS_data, obstacles_are_mirrored, obstacles, mirrored_obstacles)
    feasible_bin_indices, feasible_ps, feasible_p_lb, feasible_p_ub, remove_infeasible_bin_time = nlp_obj.remove_infeasible_bins(FRS_types=FRS_types, num_samples=num_samples)
    if print_info:
        print(f"Initializing problem takes {preprocessing_time} seconds")
        print(f"Removing feasible bins takes {remove_infeasible_bin_time} seconds")

    num_removed_bins = num_problems - len(feasible_bin_indices)
    num_problems = len(feasible_bin_indices)
    if not has_time_limit:
        max_wall_time = None
    if max_wall_time is None and max_iter is None:
        max_iter = 15
    num_limit = max_iter
    
    if save_optimization_statistics:
        save_dir = f'redefine_optimization_statistics_MaxIter{num_limit}_samples{num_samples}_maxWallTime{max_wall_time}'
        save_dir = os.path.join(save_dir_prefix, save_dir)
        if not os.path.exists(save_dir_prefix):
            os.makedirs(save_dir_prefix)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        result_dir = os.path.join(save_dir_prefix, 'results')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
    if num_problems == 0:
        # No feasible bins
        early_end = time.time()
        if print_info:
            print("No feasible bins.. Returning without going through optimization.")
        stats = {
            'status': -1,
            'runtime': early_end - total_start,
            'preprocessing_time': preprocessing_time,
            'remove_infeasible_bin_time': remove_infeasible_bin_time,
        }
        index = 0
        while os.path.exists(os.path.join(save_dir, f'{index}.json')):
            index += 1
        with open(os.path.join(save_dir, f'{index}.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        return (0, 0, 0, -1, -1, -1, early_end - total_start)
    
    # solve the NLP 
    nlp = cyipopt.Problem(
            n=num_problems,
            m=num_problems,
            problem_obj=nlp_obj,
            lb=feasible_p_lb,
            ub=feasible_p_ub,
            cl=[distance_buffer] * num_problems,
            cu=[1e20] * num_problems,
        )
    nlp.add_option('sb', 'yes')
    nlp.add_option('print_level', 0)
    
    ### IPOPT Performance settings
    ipopt_settings = {
        'tol': 1e-3,
        'acceptable_iter': 15,
        'acceptable_tol': 1e-3,
        'linear_solver': linear_solver,
    }
    skip_optimization = False
    if max_wall_time is not None:
        ipopt_settings['max_iter'] = max_iter
        ipopt_settings['max_wall_time'] = max_wall_time
        ipopt_settings['actual_max_ipopt_wall_time'] = max_wall_time - preprocessing_time - remove_infeasible_bin_time
        if ipopt_settings['actual_max_ipopt_wall_time'] > 0.:
            nlp.add_option('max_wall_time', ipopt_settings['actual_max_ipopt_wall_time'])
        else:
            skip_optimization = True
        if ipopt_settings['max_iter'] is not None:
            nlp.add_option('max_iter', ipopt_settings['max_iter'])
    else:
        ipopt_settings['max_iter'] = max_iter
        ipopt_settings['max_wall_time'] = None
        ipopt_settings['actual_max_ipopt_wall_time'] = None
        nlp.add_option('max_iter', ipopt_settings['max_iter'])
        
    nlp.add_option('tol', ipopt_settings['tol'])
    nlp.add_option('linear_solver', ipopt_settings['linear_solver'])
    
    nlp.add_option('acceptable_iter', ipopt_settings['acceptable_iter'])
    nlp.add_option('acceptable_tol', ipopt_settings['acceptable_tol'])

    if ipopt_settings['linear_solver'] == 'ma57':
        ipopt_settings['ma57_pre_alloc'] = 5.0
        nlp.add_option('hsllib', 'libcoinhsl.so') # for lib loading
        nlp.add_option('ma57_pre_alloc', ipopt_settings['ma57_pre_alloc'])
    ###
    
    if print_info:
        print("Starting to solve the optimization...")
    optimization_start = time.time()
    nlp_obj.set_ipopt_time(start_time=optimization_start, max_ipopt_time_allowed=ipopt_settings['actual_max_ipopt_wall_time'])
    if not skip_optimization:
        p_opt, info = nlp.solve(feasible_ps)
    t_opt = time.time() - optimization_start
    if print_info:
        print(f"Optimization ends with status={info['status']} with opt_variable={p_opt}, took {t_opt} seconds for optimization.")

    if skip_optimization or info['status'] != 0 or (max_wall_time is not None and t_opt > ipopt_settings['actual_max_ipopt_wall_time']):
        status = 1
        p_opt = nlp_obj.best_p
    else:
        status = 0
    get_final_solution_start = time.time()
    final_solution, cost = nlp_obj.get_final_solution(p_opt=p_opt,
                                      feasible_bin_indices=feasible_bin_indices,
                                      FRS_types=FRS_types,
                                      bin_indices_i=bin_indices_i,
                                      bin_indices_j=bin_indices_j,
                                      print_info=print_info)
    end_time = time.time()
    get_final_solution_time = end_time - get_final_solution_start
    total_time = end_time - total_start
    print(f"Totally spent {total_time} seconds")
    
    if save_optimization_statistics:
        stats = {
            'status': status,
            'runtime': total_time,
            'k': final_solution,
            'best_p': p_opt.tolist(),
            'cost': cost,
            'feasible_bin_indices': feasible_bin_indices,
            'preprocessing_time': preprocessing_time,
            'remove_infeasible_bin_time': remove_infeasible_bin_time,
            'optimization_time': t_opt, 
            'get_final_solution_time': get_final_solution_time,
            'num_removed_bins': num_removed_bins,
            'num_remaining_bins': num_problems,
            'num_cost_calls': nlp_obj.num_cost_calls,
            'num_gradient_calls': nlp_obj.num_gradient_calls,
            'num_constraint_calls': nlp_obj.num_constraint_calls,
            'num_jacobian_calls': nlp_obj.num_jacobian_calls,
            'ipopt_settings': ipopt_settings,
        }
        index = 0
        while os.path.exists(os.path.join(save_dir, f'{index}.json')):
            index += 1
        with open(os.path.join(save_dir, f'{index}.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        result_dir = os.path.join(save_dir_prefix, 'results')
        setting_filename = os.path.join(result_dir, 'ipopt_settings.json')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if not os.path.exists(setting_filename):
            with open(setting_filename, 'w') as f:
                json.dump(ipopt_settings, f, indent=2)
            
    return final_solution + (total_time,)


if __name__ == '__main__':
    result_tuple = call_optimization(agent_state=agent_state_mex, 
                    bin_indices_i = bin_indices_i, 
                    bin_indices_j = bin_indices_j, 
                    FRS_types = FRS_types, 
                    way_point = way_point,
                    x_des_local = x_des_local,
                    x_des_mirror = x_des_mirror,
                    num_obstacles = num_obstacles,
                    obstacles_are_mirrored = obstacles_are_mirrored,
                    obstacles=obstacles, mirrored_obstacles=mirrored_obstacles, 
                    sdf_net=sdf_net, sign_net=sign_net, gradient_net=gradient_net, num_samples=10, print_info=False,save_optimization_statistics=save_optimization_statistics,
                    linear_solver= 'ma57', max_iter=None, max_wall_time=None, save_dir_prefix=save_dir_prefix, has_time_limit=has_time_limit)

