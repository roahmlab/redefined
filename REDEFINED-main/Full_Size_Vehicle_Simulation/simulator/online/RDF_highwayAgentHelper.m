classdef RDF_highwayAgentHelper < agentHelper
    %% properties
    properties
        HLP;
        
        % partitions on u0
        u0_array        
        
        % reference data for plot
        ref_Z = [];
        proposed_ref_Z = [];
        t_real_start = [];
        t_proposed_start = [];
        
        prev_action = -1;
        cur_t0_idx = 1;
        saved_K = [];
        t_plan;    
        S 

        FRS_helper_handle = struct;
        truncating_factor;

        
        waypt_hist = [];
        K_hist = [];
        FRS_hist = {};
        mirror_hist = [];
        state_hist = [];
        type_manu_hist = [];
        time_hist = [];
        solve_time_hist = [];

        FRS_plot_struct = struct;
        FRS_u0_p_maps = struct;
        FRS_spd = struct;
        FRS_dir = struct;
        FRS_lan = struct;

        sdf_net = pyrunfile("models.py","distance_net");
        sign_net = pyrunfile("models.py","sign_net");
        gradient_net = pyrunfile("models.py","gradient_net");
        warmup_flag = 1;
        save_dir_prefix = '';
        max_planning_time = 99.0;
        save_initial_condition = 0;
        has_time_limit = false;

        %structures for replay mode
        replay_mode = 0;
        trial_replay_hist = {}
        K_idx = 1;

    end
    %% methods
    methods
        function AH = RDF_highwayAgentHelper(A,FRS_obj,HLP,varargin)
            AH@agentHelper(A,FRS_obj,varargin{:});
            AH.HLP = HLP;
            info_file_dir = load('dir_change_Ay_info.mat');
%             info_file_lan = load('lane_change_Ay_info.mat');
            AH.u0_array = info_file_dir.u0_vec; 
            AH.truncating_factor = 1;
        end

        
        function [K,tout] = gen_parameter_standalone(AH, world_info, agent_state,waypts)
            % main online planning function
            x_des = waypts(:,1);
            
            if (AH.cur_t0_idx > 1 && AH.prev_action == 2) || (AH.cur_t0_idx > 2 && AH.prev_action == 3)|| AH.prev_action  == 1
                AH.prev_action = -1;
                AH.cur_t0_idx = 1;
            end

            if AH.prev_action ~= -1 
                K = [AH.saved_K(1); AH.saved_K(2); AH.cur_t0_idx ;AH.prev_action];
                AH.cur_t0_idx = AH.cur_t0_idx + 1;
                return
            end

            x_des_mex = x_des;
            dyn_obs_mex = get_obs_mex(world_info.dyn_obstacles, world_info.bounds);
            agent_state_mex = agent_state(1:6);

            %% save the initial conditions
            if AH.save_initial_condition
                initial_condition_index = 0;
                dir_name = AH.save_dir_prefix + "/" + "initial_conditions/";
                if exist(dir_name, 'dir') == 0
                    mkdir(dir_name);
                end
                filename = dir_name + string(initial_condition_index) + '.mat';
                while exist(filename, 'file') ~= 0
                    initial_condition_index = initial_condition_index + 1;
                    filename = dir_name + string(initial_condition_index) + '.mat';
                end
                save(filename, "agent_state_mex", "x_des_mex", "dyn_obs_mex");
            end

            %% REDEFINED Specific problem setup
            
            %%%%%obstacle generator representation generation
            dyn_obs_local = dyn_obs_mex;
            x_des_local = world_to_local(agent_state_mex(1:3),x_des_mex);
            x_des_mirror = x_des_local;
            x_des_mirror(2) = -x_des_local(2);
%             world_to_local(agent_state_mex(1:3), dyn_obs_mex(1:3,:))
            dyn_obs_local(1:3,:) = world_to_local(agent_state_mex(1:3), dyn_obs_mex(1:3,:));

            %loop over FRS's to output the corresponding obstacle
            %generators in local frame

            obs_zono = zeros(8,length(dyn_obs_local));
            obs_zono_mirr = zeros(8,length(dyn_obs_local));
            


            for jj = 1:length(dyn_obs_local)
                cx = dyn_obs_local(1,jj);
                cy = dyn_obs_local(2,jj);
                h = dyn_obs_local(3,jj);
                v = dyn_obs_local(4,jj);  
                l = dyn_obs_local(5,jj);
                w = dyn_obs_local(6,jj);
                
                rot_h = [cos(h) -sin(h);
                         sin(h) cos(h)];
                rot_h_mirror = [cos(-h) -sin(-h);
                                sin(-h) cos(-h)];
                
                %mirrored centers
                cent_mirr = [cx;-cy];

                %generate rotated generators 
                gens = rot_h*[l/2 0;0 w/2];
                vgen = rot_h*[v/2;0];

                %generate rotated mirrored generators 
                gens_mirr = rot_h_mirror*[l/2 0;0 w/2];
                vgen_mirr = rot_h_mirror*[v/2;0];

                obs_zono_jj = [cx;cy;
                            reshape(gens,[],1);
                            vgen];
                obs_zono_mirr_jj  = [cent_mirr;
                            reshape(gens_mirr,[],1);
                            vgen_mirr];
               
                obs_zono(:,jj) = obs_zono_jj;
                obs_zono_mirr(:,jj) = obs_zono_mirr_jj;
                
            end
            
%             cut_i = 1;
%             cut_j = size(dyn_obs_local,2);
            %obs_zono(:,3:18) = []; %%%%TEMP
%             obstacles = reshape(obs_zono(:,cut_i:cut_j),[],1)';
%             mirrored_obstacles = reshape(obs_zono_mirr(:,cut_i:cut_j),[],1)';
            obstacles = reshape(obs_zono,[],1)';
            mirrored_obstacles = reshape(obs_zono_mirr,[],1)';

            %%%%%% Waypoint augmentation
            xdes_lan = [agent_state_mex(4)*6+4*6; x_des_local(2)*1.15; x_des_local(3)];
            xdes_lan_mirr = [agent_state_mex(4)*6+4*6; -x_des_local(2)*1.15; x_des_local(3)];

            xdes_dir = [agent_state_mex(4)*3+4*3; x_des_local(2)*0.7; x_des_local(3)];
            xdes_dir_mirr = [agent_state_mex(4)*3+4*3; -x_des_local(2)*0.7; x_des_local(3)];

            xdes_spd = [1.5*(min(x_des_local(1)/3,agent_state_mex(4)+4) + agent_state_mex(4)); x_des_local(2); x_des_local(3)];

            %%%%%% get FRS and bin info
            %get index for u0 to index FRS with
%             type_manu_all = ["Au","lan","dir"];

            
%             [~,idxu0] = min(abs(AH.u0_array - agent_state(4)));
%             M = AH.zono_full.M_mega{idxu0};
            u0_FRS_ranges = AH.FRS_u0_p_maps.u0map;
            urange_diff = abs(u0_FRS_ranges - agent_state(4));
%             [~,idxu0_minpot] = min(urange_diff);
            [~,idxmin]  = min(sum(urange_diff,2));
            idxu0 = min(idxmin);

%             idxu0_z = [];
%             for kk = 1:length(u0_FRS_ranges)
%                 if agent_state(4) >= u0_FRS_ranges(kk,1) && agent_state(4) <= u0_FRS_ranges(kk,2)
%                     idxu0_z = [idxu0_z,kk];
%                 end
%             end
%             idxu0 = min(idxu0_minpot);


            spd_FRS_ranges = AH.FRS_u0_p_maps.pmap(idxu0).spd;
            lan_FRS_ranges = AH.FRS_u0_p_maps.pmap(idxu0).lan;
            dir_FRS_ranges = AH.FRS_u0_p_maps.pmap(idxu0).dir;

            min_spd_au = 0; %min speed that the vehicle can change speed to in each maneuver
            max_spd_au = agent_state_mex(4) + 4; %max speed that the vehicle can speed up to in each maneuver
            bin_indices_i = [];
            bin_indices_j = [];
            FRS_types = [];
            obstacles_are_mirrored = [];
            way_point = [];
            spd_FRS_range_feas = [];

            for ii = 1:3
                if ii ==1
                    for i = 1:length(spd_FRS_ranges)
                            center_p = mean(spd_FRS_ranges(i,:));
                            if center_p >= min_spd_au && center_p <= max_spd_au
                                bin_indices_i = [bin_indices_i,idxu0-1];
                                bin_indices_j = [bin_indices_j,i-1];
                                FRS_types = [FRS_types,ii-1];
                                obstacles_are_mirrored = [obstacles_are_mirrored,0];
                                spd_FRS_range_feas = [spd_FRS_range_feas; spd_FRS_ranges(i,:)];
                                way_point = [way_point;xdes_spd];
                            end
    
                    end

                elseif ii == 2

                    %Don't look at direction changes if they are outside the linear regime (<= 7.0m/s)
                    if u0_FRS_ranges(idxu0,1) > 7 
                        for i = 1:length(dir_FRS_ranges)
                            %add double the number of indices to account
                            %for mirrored FRSes
                            bin_indices_i = [bin_indices_i,idxu0-1,idxu0-1];
                            bin_indices_j = [bin_indices_j,i-1,i-1];
                            FRS_types = [FRS_types,ii-1,ii-1];
                            obstacles_are_mirrored = [obstacles_are_mirrored,0,1]; 
                            way_point = [way_point;xdes_dir;xdes_dir_mirr];
                        end

                    end
                 
                elseif ii == 3

                    %Don't look at lane changes if they are outside the linear regime (<= 7.0m/s)
                    if u0_FRS_ranges(idxu0,1) > 7 
                        for i = 1:length(lan_FRS_ranges)
                            %add double the number of indices to account
                            %for mirrored FRSes
                            bin_indices_i = [bin_indices_i,idxu0-1,idxu0-1];
                            bin_indices_j = [bin_indices_j,i-1,i-1];
                            FRS_types = [FRS_types,ii-1,ii-1];
                            obstacles_are_mirrored = [obstacles_are_mirrored,0,1]; 
                            way_point = [way_point;xdes_lan;xdes_lan_mirr];
                        end

                    end
                end

            end

            %reshape waypoints
            way_point = way_point';

            %%%%%% get other inputs
            num_obstacles = floor(size(obstacles,2)/8);

            u0_Q = agent_state_mex(4);
            v0_Q = agent_state_mex(5);
            r0_Q = agent_state_mex(6);

            %% CONSTRAINT TIMING BLOCK
            %comment if we don't want to benchmark constraint evaluation
            %tim
%             num_k = 1000; %number of k values you want to evaluate
%             tic
%             times = time_cons_eval(AH,agent_state_mex, x_des_mex, dyn_obs_mex,...
%                     spd_FRS_ranges,lan_FRS_ranges,dir_FRS_ranges,...
%                     FRS_types,bin_indices_j,obstacles_are_mirrored,num_k);
% 
%             toc

            
            %% Run REDEFINED Optimizations




             if AH.replay_mode
                
                if AH.K_idx > length(AH.trial_replay_hist.K_hist)
                    k_mex = [0;0;1;-1];
                    K = [];
                    return
                end

                %%k_mex is [k_speedchange;k_lanechange;0;manu_type;FRS_idx1;FRS_idx2];
                k_mex = zeros(6,1);
                k_val = AH.trial_replay_hist.K_hist(AH.K_idx);
                type_manu = AH.trial_replay_hist.type_manu_hist(AH.K_idx);
                mirror_flag = AH.trial_replay_hist.mirror_hist(AH.K_idx);
                multiplier = -2*mirror_flag + 1; %want multiplier to be negative if mirrored and positive if not
                FRS = AH.trial_replay_hist.FRS_hist{1,AH.K_idx};

                if type_manu == 1
                    k_mex(1) = k_val;
                    AH.prev_action = -1; 
                    AH.cur_t0_idx = 1;
                    
                else
                    k_mex(1) = AH.trial_replay_hist.state_hist(4,AH.K_idx);
                    k_mex(2) = k_val;
                    AH.prev_action = type_manu;
                    AH.cur_t0_idx = 2;
                    AH.saved_K = k_mex;
                end
                k_mex(4) = type_manu;
                k_mex(3) = 1;
                K = k_mex(1:4);
                k = k_val;
                type_manu_all = ["Au","dir","lan"];
                type_text = type_manu_all(type_manu);
                [~,idxu0] = min(abs(AH.u0_array - agent_state(4)));
                indices = [];
                AH.K_idx = AH.K_idx+1;
                
             else
                max_planning_time = AH.max_planning_time;
                save_dir_prefix = AH.save_dir_prefix;
                has_time_limit = AH.has_time_limit;
    
                tic
                k_mex = pyrunfile("redefined_opt.py","result_tuple",agent_state_mex=agent_state_mex, ...
                                bin_indices_i = bin_indices_i,bin_indices_j=bin_indices_j,...
                                FRS_types = FRS_types, way_point = way_point,...
                                x_des_local=x_des_local, x_des_mirror=x_des_mirror,...
                                num_obstacles=num_obstacles,obstacles_are_mirrored = obstacles_are_mirrored,...
                                obstacles=obstacles, mirrored_obstacles=mirrored_obstacles,sdf_net=AH.sdf_net, sign_net=AH.sign_net, gradient_net=AH.gradient_net,...
                                save_optimization_statistics=1, max_wall_time=max_planning_time, save_dir_prefix=save_dir_prefix, has_time_limit=has_time_limit);
                t_temp = double(k_mex(7));
                
                AH.solve_time_hist = [AH.solve_time_hist, t_temp];
                indices = int32(k_mex(5:6))+1; 
                k_mex = double(k_mex(1:4));% k_mex = round(double(k_mex(1:4)),4);
                K = double(k_mex);
                K(3) = 1;
                idxu0 = idxu0 + 1; %compensate for index mismatch between plotting FRS and regular FRS
    
                %%%%TEMP
    %             indices(2) = 1;
    %             K(4) = 2;
                if K(end) == -1
                    K = [];
                    return
                else % visualize FRS and update for the agent
                    type_manu = K(4);
                    multiplier = 1;
                    mirror_flag = 0;
    
                    type_manu_all = ["Au","dir","lan"];
                    type_text = type_manu_all(type_manu);
                    M = AH.zono_full.M_mega{idxu0};
                    
                    if type_manu == 1
                        indices = flip(indices);
                        indices(2) = 1;
                        k = K(1);
                        FRS = M(type_text);
                        AH.prev_action = -1; 
                        AH.cur_t0_idx = 1;
                    else
                        k = K(2);
                        if k<0
                            multiplier = -1;
                            mirror_flag = 1;
                        end
                        indices = flip(indices);
                        indices(2) = 1;
                        FRS = M(type_text); 
                        AH.prev_action = type_manu;
                        AH.cur_t0_idx = 2;
                        AH.saved_K = K;
                    end
                    if size(FRS,1) == 1
                        FRS = FRS';
                    end
                    FRS = FRS{indices(1),indices(2)};
    
                end
             end

            AH.FRS_plot_struct.k = k;
            AH.FRS_plot_struct.type_manu = type_manu;
            AH.FRS_plot_struct.FRS = FRS;
            AH.FRS_plot_struct.mirror_flag = mirror_flag;
            AH.FRS_plot_struct.agent_state = agent_state;
            AH.FRS_plot_struct.multiplier = multiplier;
%                 AH.plot_selected_parameter_FRS(k,type_manu,FRS,mirror_flag,agent_state,multiplier);

            
            AH.waypt_hist = [AH.waypt_hist x_des];
            AH.K_hist = [AH.K_hist k];
            AH.FRS_hist{end+1} = FRS;
            AH.mirror_hist = [AH.mirror_hist mirror_flag];
            AH.type_manu_hist = [AH.type_manu_hist type_manu];
            AH.state_hist = [AH.state_hist agent_state];
            AH.time_hist = [AH.time_hist AH.A.time(end)];
            
            tout = 0; % place holder
        end
        
        function plot_FRS(AH)
            k = AH.FRS_plot_struct.k;
            type_manu = AH.FRS_plot_struct.type_manu;
            FRS = AH.FRS_plot_struct.FRS;
            mirror_flag = AH.FRS_plot_struct.mirror_flag;
            agent_state = AH.FRS_plot_struct.agent_state;
            multiplier = AH.FRS_plot_struct.multiplier;
            AH.plot_selected_parameter_FRS(k,type_manu,FRS,mirror_flag,agent_state,multiplier);
        end

        function plot_selected_parameter_FRS(AH,K,type_manu,FRS,mirror_flag,agent_state,multiplier)
            if ~isempty(K)
                %clear data and then plot
                AH.FRS_helper_handle.XData = cell(3,1);
                AH.FRS_helper_handle.YData = cell(3,1);
                if type_manu == 1 % 1: speed change. 2: direction change. 3: lane change
                    AH.plot_zono_collide_sliced(FRS,mirror_flag,agent_state,[K; 0],[0 0 1],2);
                else
                    AH.plot_zono_collide_sliced(FRS,mirror_flag,agent_state,[agent_state(4);K *multiplier],[0 1 0],2);
                end
            end
        end


        
        function [T, U, Z]=gen_ref(AH, K, reference_flag,agent_state, ref_time)
            % generate reference based on parameter and states
            if ~exist('agent_state','var')
                agent_info = AH.get_agent_info();
                agent_state = agent_info.state(:,end);
            end
            if ~exist('ref_time','var')
                ref_time = AH.A.time(end);
            end
            if ~exist('reference_flag','var')
                reference_flag = 1;
            end
            u_cur = agent_state(4) ;
            y_cur = agent_state(2) ;
            x_cur = agent_state(1) ;
            Au = K(1);
            Ay = K(2);
            t0_idx = K(3);
            
            t0 = (t0_idx-1)*AH.t_move;
            type_manu = K(4);
            if type_manu == 3 % 1: speed change. 2: direction change. 3: lane change
                [T, U,Z] = gaussian_T_parameterized_traj_with_brake(t0,Ay,Au,u_cur,[],0,1);
            else
                [T, U,Z] = sin_one_hump_parameterized_traj_with_brake(t0,Ay,Au,u_cur,[],0,1);
            end
            
            if reference_flag
                AH.ref_Z=[AH.ref_Z;x_cur+Z(1,:);y_cur+Z(2,:)];% for plotting
                AH.t_real_start = [AH.t_real_start;ref_time];
            else
                AH.proposed_ref_Z=[AH.proposed_ref_Z;x_cur+Z(1,:);y_cur+Z(2,:)];% for plotting
                AH.t_proposed_start = [AH.t_proposed_start;ref_time];
            end
            
            
        end

        function reset(AH,flags,eps_seed)
            if ~exist('eps_seed','var')
                AH.A.reset();
            else
                rng(eps_seed)
                AH.A.reset();
            end
            AH.flags = flags;
            AH.ref_Z = [];
            AH.proposed_ref_Z = [];
            AH.t_real_start = [];
            AH.t_proposed_start = [];
            AH.K_hist = [];
            AH.waypt_hist = [];
            AH.FRS_hist = {};
            AH.mirror_hist = [];
            AH.state_hist = [];
            AH.type_manu_hist = [];
            AH.time_hist = [];
            if ~isempty(AH.HLP)
                AH.HLP.reset();
                if ~isempty(AH.HLP.plot_data.waypoints)
                    AH.HLP.plot_data.waypoints.XData = [];
                    AH.HLP.plot_data.waypoints.YData = [];
                end
            end
        end
        
        %% plot functions
        function plot(AH)
            hold_check = false ;
            if ~ishold
                hold_check = true ;
                hold on
            end
            if ~isempty(AH.planned_path)
               plot(AH.planned_path(1,:),AH.planned_path(2,:),'k-','LineWidth',1);
            end
            text(-250,15,"u="+num2str(AH.A.state(4,end))+"m/s",'Color','red','FontSize',15)
            
            if hold_check
                hold off ;
            end
            
        end
        function plot_zono_collide_sliced(AH,FRS,mirror_flag,agent_state,K,color,slice_level)
            if K(2) == 0
                slice_dim = 11;
                k_slice = K(1);
            else
                slice_dim = 12;
                k_slice = K(2);
            end

            
            for t_idx = 1:5:length(FRS.vehRS_save) 
                if slice_level == 0
                    zono_one = FRS.vehRS_save{t_idx};
                elseif slice_level == 1
                    zono_one = zonotope_slice(FRS.vehRS_save{t_idx}, [7;8;9], [agent_state(4);agent_state(5);agent_state(6)]);
                elseif slice_level == 2
                    zono_one = zonotope_slice(FRS.vehRS_save{t_idx}, [7;8;9;slice_dim], [agent_state(4);agent_state(5);agent_state(6);k_slice]);
                else
                    error('unknown slice_level in plot selected zono');
                end
                h = plot(zono_one,[1,2],'Color',color);
                if mirror_flag
                    h.YData = - h.YData;
                end
                XY = [h.XData(:) h.YData(:)];                                    
                theta = agent_state(3);
                R=[cos(theta) -sin(theta); sin(theta) cos(theta)];
                rotXY=XY*R'; %MULTIPLY VECTORS BY THE ROT MATRIX
                Xqr = reshape(rotXY(:,1), size(h.XData,1), []);
                Yqr = reshape(rotXY(:,2), size(h.YData,1), []);
                %SHIFTING
                h.XData = Xqr+agent_state(1);
                h.YData = Yqr+agent_state(2);
                
                AH.FRS_helper_handle.XData{slice_level+1} = [h.YData nan AH.FRS_helper_handle.XData{slice_level+1}];
                AH.FRS_helper_handle.YData{slice_level+1} = [h.XData nan AH.FRS_helper_handle.YData{slice_level+1}];
            end
            

        end

    end
end

%% helper function to generate obstacle structure for c++
function obj_mex = get_obs_mex(dyn_obs, bounds)
    all_pts = dyn_obs{1};
    all_vels = dyn_obs{2};
    obj_mex = [];
    n_obs = length(all_vels);
    for dyn_obs_idx = 1:n_obs
        dyn_obs_pts_start_idx = ((dyn_obs_idx-1) * 6) + 1;
        curr_pts = all_pts(:, ...
            dyn_obs_pts_start_idx:dyn_obs_pts_start_idx+3);
        deltas = max(curr_pts,[],2) - min(curr_pts,[],2);
        means = mean(curr_pts, 2);
        dyn_c_x = means(1);
        dyn_c_y = means(2);
        dyn_length = deltas(1);
        dyn_width = deltas(2);
        dyn_velocity = all_vels(dyn_obs_idx);
        dyn_heading_rad = 0;
        obj_mex(:,end+1) = [dyn_c_x; 
                            dyn_c_y; 
                            dyn_heading_rad;
                            dyn_velocity;
                            dyn_length;
                            dyn_width];
    end
    xlo = bounds(1) ; xhi = bounds(2) ;
    ylo = bounds(3) ; yhi = bounds(4) ;
    dx = xhi - xlo;
    dy = yhi - ylo;
    x_c = mean([xlo, xhi]);
    y_c = mean([ylo, yhi]);
    b_thick = 0.01;
    b_half_thick = b_thick / 2.0;

    % Top
    obj_mex(:,end+1) = [x_c; yhi+b_half_thick; 0; 0; dx; b_thick];
    % Bottom
    obj_mex(:,end+1) = [x_c; ylo-b_half_thick; 0; 0; dx; b_thick];
    % Right
    obj_mex(:,end+1) = [xhi+b_half_thick; y_c; 0; 0; b_thick; dy];
    % Left
    obj_mex(:,end+1) = [xlo-b_half_thick; y_c; 0; 0; b_thick; dy];
end
