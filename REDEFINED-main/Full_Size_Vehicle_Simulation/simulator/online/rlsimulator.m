classdef rlsimulator < handle
    properties
        safety_layer = 'A'; % automatic waypoint 
        discrete_flag = 0;
        replace_action = 0; % 0 for original rl lib, 1 for replace with new action, 2 for with punishing reward b/w replaced and original
        plot_sim_flag = 1; %plot simulation figure 1
        plot_AH_flag;
        plot_adjust_flag = 0;% plot replacement process figure 2
        eval = 0; % used to set random seed, using the same seed generate same random environment
        AH
        W
        
        fig_num = 1
        epscur;
        time_vec=[];
        wp_hist = [];

        save_result = false;
        save_video = false;
        videoObj = [];
        video_dt = 0.01;
        video_dt_multiplier = 2;

        t_now = 0;

        plot_fancy_vehicle = false;
        visualize = 1;
        single_plan = 0; %flag to tell the simulator that these are single planning iterations

        save_dir_prefix = '';
    end

    methods
        %% constructor
        function S = rlsimulator(AH,W,varargin)
            S = parse_args(S,varargin{:}) ;
            S.AH = AH;
            S.W  = W;
            if S.visualize
                figure(1);
                set(gcf,'Position',[-491.8000 1.0682e+03 1900 842.4000]);
%                 set(gcf,'Position',[-491.8000 1.0682e+03 1900 842.4000]);
            end
            if S.save_video
                Videoname = sprintf('REDEFINE_SceIdx-%s', num2str(S.epscur));
                S.videoObj = VideoWriter(Videoname,'Motion JPEG AVI');
                S.videoObj.FrameRate = 1/(S.video_dt * S.video_dt_multiplier);
                open(S.videoObj);
            end
        end
        
        %% step
        function [Observation,Reward,IsDone,LoggedSignals,varargout] = step(S,action)
            %step in time and check for done flags
            tic
            agent_info = S.AH.get_agent_info();
            world_info = S.W.get_world_info(agent_info);%where obstacles are
            % move 
            [action_replaced, replace_distance, stuck, k, wp] = S.AH.advanced_move(action,world_info);
            agent_info = S.AH.get_agent_info();
%             collision = S.W.collision_check(agent_info);
            collision = 0;
            
            %if first planning iteration is not feasible don't try plot anything
            if isempty(S.AH.FRS_hist)
                IsDone = 4;
                Observation =[]; 
                Reward = 0;%S.W.getRew(agent_info,Observation);
            else
                % visualization
                if ~S.plot_fancy_vehicle && S.visualize
                    S.plot();
                    scatter(wp(1), wp(2), 360,'k','x','LineWidth',5);
                end
                if S.visualize
                    while abs(S.t_now - S.AH.A.time(end)) > S.video_dt
                        if S.plot_fancy_vehicle
                            S.plot();
                            scatter(wp(1), wp(2), 360,'k','x','LineWidth',5);
                        end
                        % ego vehicle
                        idx = find(S.AH.A.time <= S.t_now, 1, 'last');
                        alpha = (S.t_now - S.AH.A.time(idx)) / (S.AH.A.time(idx+1) - S.AH.A.time(idx));
                        state_now = S.AH.A.state(:,idx)*(1-alpha) + S.AH.A.state(:,idx+1)*alpha;
                        xy_now = state_now(1:2);
                        h_now = state_now(3);
                        if ~S.plot_fancy_vehicle
                            footprint = [cos(h_now), -sin(h_now); sin(h_now) cos(h_now)]*[-2.4, 2.4, 2.4, -2.4, -2.4; -1.1, -1.1, 1.1, 1.1, -1.1]+xy_now;
                            S.AH.A.plot_data.footprint.Vertices = footprint';
                        else
                            footprint = [cos(h_now), -sin(h_now); sin(h_now) cos(h_now)]*[-2.4, 2.4, 2.4, -2.4, -2.4; -1.1, -1.1, 1.1, 1.1, -1.1]+xy_now;
                            plot_vehicle(xy_now(1), xy_now(2), h_now, [0,0,0]/255, [140,140,140]/255, 1);
                        end
                        plot([S.AH.A.state(1,1:idx),xy_now(1)],[S.AH.A.state(2,1:idx),xy_now(2)],'k','LineWidth',2);
        
                        % move other vehicle
                        num_static_obstacle = S.W.num_cars - S.W.num_moving_cars-1;
                        if S.plot_fancy_vehicle % plot static vehicle
                            for i = 1:num_static_obstacle
                                x_now = S.W.envCars(i+1,1);
                                y_now = S.W.envCars(i+1,3);
                                plot_vehicle(x_now,y_now,0, [255,255,255]/255, [200,200,200]/255, 1);
                            end
                        end
                        for i = 1:S.W.num_moving_cars
                            v_now = S.W.envCars(i+num_static_obstacle+1,2); 
                            x_now = S.W.envCars(i+num_static_obstacle+1,1) + v_now * S.t_now;
                            y_now = S.W.envCars(i+num_static_obstacle+1,3); 
                            if S.plot_fancy_vehicle
                                plot_vehicle(x_now,y_now,0, [255,255,255]/255, [200,200,200]/255, 1);
                            else
                                xmid = 0.5*( min(S.W.plot_data.obstacles_seen.XData(:,i+num_static_obstacle)) +  max(S.W.plot_data.obstacles_seen.XData(:,i+num_static_obstacle)) );
                                ymid = 0.5*( min(S.W.plot_data.obstacles_seen.YData(:,i+num_static_obstacle)) +  max(S.W.plot_data.obstacles_seen.YData(:,i+num_static_obstacle)) );
                                S.W.plot_data.obstacles_seen.XData(:,i+num_static_obstacle) =  S.W.plot_data.obstacles_seen.XData(:,i+num_static_obstacle) - xmid + x_now;
                                S.W.plot_data.obstacles_seen.YData(:,i+num_static_obstacle) =  S.W.plot_data.obstacles_seen.YData(:,i+num_static_obstacle) - ymid + y_now;
                            end
                        end
                        
                        % check collision here
                        for i = 1:S.W.num_cars-1
                            v_now = S.W.envCars(i+1,2); 
                            x_now = S.W.envCars(i+1,1) + v_now * S.t_now;
                            y_now = S.W.envCars(i+1,3);
                            obs = [-2.4, 2.4, 2.4, -2.4, -2.4; -1.1, -1.1, 1.1, 1.1, -1.1]+[x_now; y_now];
                            if ~collision 
                                [bla,~] = polyxpoly(footprint(1,:),footprint(2,:),obs(1,:),obs(2,:));
                                if ~isempty(bla)
                                    collision = 1;
                                end
                            end
                        end
    
                        if xy_now(1)+200 <= 1030
                            xlim([xy_now(1)-10, xy_now(1)+200]);
                        else
                            xlim([1030-210,1030]);
                        end
        %                 xlim([90 280]);
                        
                        ylim([-5, 12]);
                        title("Speed="+num2str(state_now(4),'%.1f')+" [m/s]");
                        S.t_now = S.t_now + S.video_dt_multiplier * S.video_dt;
                        if S.save_video
                            frame = getframe(gcf);
                            writeVideo(S.videoObj, frame);
                        end
                        pause(S.video_dt_multiplier * S.video_dt)
                    end
                end
    
    
                S.wp_hist{end+1} = wp;
                agent_info.replace_distance = replace_distance;
                Observation =[]; %S.W.get_ob(agent_info);
                Reward = 0;%S.W.getRew(agent_info,Observation);
    
                if collision
                    warning("A collision Happened!");
                end
                goal_check = S.W.goal_check(agent_info);
                
                IsDone = S.determine_isDone_flag(collision,action_replaced,stuck,goal_check);
            end
                
            

            if S.eval &&( IsDone == 1 || IsDone == 3 ||IsDone == 4 || IsDone == 5) && S.save_result
                Filename = sprintf('/REDEFINE_IsDone_%s-SimID_%s.mat', num2str(IsDone), num2str(S.epscur));
                Filename = strcat(S.save_dir_prefix, '/results', Filename);
                ref_Z = S.AH.ref_Z;
                proposed_ref_Z = S.AH.proposed_ref_Z;
                T= S.AH.T;
                t_real_start_arr = S.AH.t_real_start;
                t_proposed_start_arr = S.AH.t_proposed_start;
                t_move =  S.AH.t_move;
%                 plotting_param = S.AH.FRS_plotting_param;
                envCars = S.W.envCars;
                hist_info.K_hist = S.AH.K_hist;
                hist_info.FRS_hist = S.AH.FRS_hist;
                hist_info.mirror_hist = S.AH.mirror_hist;
                hist_info.type_manu_hist = S.AH.type_manu_hist;
                hist_info.state_hist = S.AH.state_hist;
                hist_info.time_hist = S.AH.time_hist;
                hist_info.wp_hist = S.wp_hist;
                hist_info.solve_time_hist = S.AH.solve_time_hist;                
                save(Filename,'hist_info','agent_info','world_info','ref_Z','proposed_ref_Z','T','t_move','t_real_start_arr','t_proposed_start_arr','envCars')
            end
            
            drawnow;
            LoggedSignals = struct;
            
            % send output action if nargout > 4
            if nargout > 4
                varargout = {action_replaced,k} ;
            end
            t_step = toc;
            S.time_vec = [S.time_vec t_step];
        end
        
        %% reset
        function [iniOb, LoggedSignals] = reset(S)
            LoggedSignals = struct;
            flags = struct;
            flags.discrete_flag = S.discrete_flag;
            flags.replace_action = S.replace_action;
            flags.safety_layer = S.safety_layer;
            if S.eval
                S.W.setup(S.epscur);
                S.AH.reset(flags,S.epscur);
            else
                S.W.setup();
                S.AH.reset(flags);
            end
            iniOb =[];
%             S.plot();
        end
        
        %% helper functions
        function plot(S)

            if S.plot_sim_flag
                figure(S.fig_num) ;
                cla ; hold on ; axis equal ;

                % plot road
                if ~check_if_plot_is_available(S.W,'road_lanes')
                    road_lanes ={};
                    w= 2;
                    road_lanes{1} =  fill([-S.W.start_line S.W.goal(1)+500 S.W.goal(1)+500 -S.W.start_line -S.W.start_line],[-0.7 -0.7 3*S.W.lanewidth+0.7 3*S.W.lanewidth+0.7 -0.7]-0.5*S.W.lanewidth,[190,190,190]/256); % JL_plot: road 
                    road_lanes{2} =  plot([-S.W.start_line,S.W.goal(1)+500],[3*S.W.lanewidth, 3*S.W.lanewidth]-0.5*S.W.lanewidth,'LineWidth',w,'Color',[255, 255, 255]/255);
                    road_lanes{3} =  plot([-S.W.start_line,S.W.goal(1)+500],[2*S.W.lanewidth, 2*S.W.lanewidth]-0.5*S.W.lanewidth,'--','LineWidth',w,'Color',[1 1 1]);
                    road_lanes{4} =  plot([-S.W.start_line,S.W.goal(1)+500],[S.W.lanewidth, S.W.lanewidth]-0.5*S.W.lanewidth,'--','LineWidth',w,'Color',[1 1 1]);
                    road_lanes{5} =  plot([-S.W.start_line,S.W.goal(1)+500],[0, 0]-0.5*S.W.lanewidth,'LineWidth',w,'Color',[255, 255, 255]/255);
                    S.W.plot_data.road_lanes = road_lanes;
                end

%                 goal_pos = make_box(S.W.goal_radius*2);
%                 patch(goal_pos(1,:)+S.W.goal(1),goal_pos(2,:)+S.W.goal(2),[1 0 0]) ;
                S.AH.plot_FRS(); 
                if ~S.plot_fancy_vehicle
                    S.W.plot(S.AH.A.get_agent_info); % JL_plot: obstacle vehicles are plotted here. Modify S.W.plot_data.obstacles_seen.XData and YData to change vehicle's location once S.plot finishes its execution. 
                                                     % NOTE: this is not actual obstacle plotting! it's just a place holder    
                    S.AH.plot_A(); % plot ego vehicle    
                end
                
                xlabel('x [m]');
                ylabel('y [m]');
                set(gca,'FontSize',15)
                if  S.plot_AH_flag
                    if S.plot_adjust_flag
                        S.AH.plot_adjust()
                    end
                end
%                 xlim([S.AH.A.state(1,end)-10 S.AH.A.state(1,end)+100]);
            end

            
        end
        
        function [IsDone] = determine_isDone_flag(S,collision,action_replaced,stuck,goal_check)
            if collision && action_replaced
                IsDone = 3;
            elseif collision
                IsDone = 1;
            elseif goal_check
                IsDone = 5;
            elseif stuck
                IsDone = 4;
            elseif S.single_plan && (S.AH.prev_action ==-1 || (S.AH.cur_t0_idx > 2 && S.AH.prev_action == 3))
                IsDone = 5; %force isDone to be 5 if no crash or stop after single iteration
            elseif action_replaced
                IsDone = 2;
            else
                IsDone = 0;
            end
        end
    end
end
