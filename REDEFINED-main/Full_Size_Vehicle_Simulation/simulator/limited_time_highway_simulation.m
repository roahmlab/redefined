% Main function for REDEFINE simulation

% load FRS
frs_filename = 'car_frs.mat';
if ~exist('frs','var')
    disp('Loading frs')
    frs = load(frs_filename) ;
else
    disp('table already loaded') ;
end

visualize = 1;
plot_sim_flag = visualize;
plot_AH_flag = 1;
save_result = false; % make it true if you want to save the simulation data
save_video = false;
plot_fancy_vehicle = false;

%% Error catching for simulation setup
if save_video && ~visualize
    error("Error. Visualize flag needs to enabled on if save_video flag is enabled")
end

%% set up required objects for simulation
lanewidth = 3.7;
bounds = [0, 2000, -(lanewidth / 2) - 1, ((lanewidth/2) * 5) + 1];
goal_radius = 12;
world_buffer = 1 ;
t_move = 3;
t_plan = 3;
t_failsafe_move = 3;
verbose_level = 0;
num_ego_vehicles = 1;
num_moving_cars = 30;
num_static_cars = 0;
num_total_cars = num_ego_vehicles + num_moving_cars + num_static_cars;
hlp_lookahead = 90;
SIM_MAX_DISTANCE = 400; %maximum distance along highway to spawn a vehicle. Should not be >900 since highway is only 1000m
min_obs_start_dist = 50; %min distance away from ego vehicle that the obstacle vehicles can be spawned
car_safe_dist = 1; %min allowable distance between obstacle vehicles

save_initial_condition = 0;
num_trials = 1000;
max_planning_times = [0.35 0.30 0.25];

for max_p_t_index = 1:length(max_planning_times)
    max_planning_time = max_planning_times(max_p_t_index);
    save_dir_prefix = "RecedingHorizonPlanning/" + string(num_trials) + "Trials_" + string(num_moving_cars) + "Obs_" + string(SIM_MAX_DISTANCE) + "obsLength_" + string(max_planning_time) + "MaxTime";
    for j = 1:num_trials
        % RESET simulation environment
        World = dynamic_car_world( 'bounds', bounds, ...
            'buffer', world_buffer, 'goal', [1010;3.7], ...
            'verbose', verbose_level, 'goal_radius', goal_radius, ...
            'num_cars', num_total_cars, 'num_moving_cars', num_moving_cars, ...
            't_move_and_failsafe', t_move+t_failsafe_move, ...
            'SIM_MAX_DISTANCE',SIM_MAX_DISTANCE,'min_obs_start_dist',min_obs_start_dist, ...
             'car_safe_dist',car_safe_dist) ;
    
        Agent = highway_cruising_10_state_agent; % takes care of vehicle states and dynamics
        Agent.desired_initial_condition = [10;0; 0; 20;0;0;20;0;0;0];
        Agent.integrator_type= 'ode45';
        HLP = simple_highway_HLP; % high-level planner
        HLP.lookahead = hlp_lookahead; 
        AgentHelper = RDF_highwayAgentHelper(Agent,frs,HLP,'t_plan',t_plan,'t_move',t_move,'t_failsafe_move',t_failsafe_move,...
            'verbose',verbose_level,'save_dir_prefix', save_dir_prefix, 'max_planning_time', max_planning_time, 'save_initial_condition', save_initial_condition, 'has_time_limit', true); % takes care of online planning
        AgentHelper.FRS_u0_p_maps = load("u0_p_maps.mat");
    
        Simulator = rlsimulator(AgentHelper,World,'plot_sim_flag',plot_sim_flag,'plot_AH_flag',plot_AH_flag,'save_result',save_result,...
            'plot_fancy_vehicle', plot_fancy_vehicle, 'save_video', save_video,'visualize', visualize, 'epscur',j, 'save_dir_prefix', save_dir_prefix);
    
        AgentHelper.S = Simulator;
        Simulator.eval = 1; %turn on evaluation so summary will be saved
    
       
    
        rng(j+1);
        IsDone4 = 0;
        Simulator.epscur = j;
        Simulator.reset();
        for i = 1:4000
            i
            AgentHelper.planned_path = [linspace(0,1000);repmat([0;0],1,100)];
            [~,~,IsDone,LoggedSignal]=Simulator.step([rand*2-1;rand*2-1]);
            if IsDone == 1 || IsDone == 3 || IsDone == 4 || IsDone == 5
                %crash
                %      crash with safety layer on
                %                      safely stopped but stuck
                %                                           reached goal!
                if Simulator.save_video
                    close(Simulator.videoObj);
                end
                break
            end
        end
        pause(1)
    end
end


done = 'Simulation Complete'
