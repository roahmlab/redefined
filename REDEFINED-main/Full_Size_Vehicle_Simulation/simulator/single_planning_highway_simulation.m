% Main function for REDEFINE simulation

% load FRS
frs_filename = 'car_frs.mat';
if ~exist('frs','var')
    disp('Loading frs')
    frs = load(frs_filename) ;
else
    disp('table already loaded') ;
end


visualize = 0; %need to turn off to stop visualization
plot_sim_flag = visualize; %if visualize is on plot_sim_flag should also be on
plot_AH_flag = 1;
save_result = true; % make it true if you want to save the simulation data
save_video = false; %make it true if you want to save videos of the trials
plot_fancy_vehicle = false;

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
num_moving_cars = 10;
num_static_cars = 0;
num_total_cars = num_ego_vehicles + num_moving_cars + num_static_cars;
hlp_lookahead = 90;
max_planning_time = 0.35;

SIM_MAX_DISTANCE = 200; %maximum distance along highway to spawn a vehicle. Should not be >900 since highway is only 1000m
min_obs_start_dist = 10; %min distance away from ego vehicle that the obstacle vehicles can be spawned
car_safe_dist = 1; %min allowable distance between obstacle vehicles
num_planning_iteration = 2;
single_plan = 1; %flag to tell the simulator that these are single planning iterations
save_initial_condition = 1;
num_trials = 500;
save_dir_prefix = string(num_trials) + "Trials_" + string(num_moving_cars) + "Obs_" + string(SIM_MAX_DISTANCE) + "Length_" + string(max_planning_time) + "MaxTime";

%% Error catching for simulation setup
if save_video && ~visualize
    error("Error. Visualize flag needs to enabled on if save_video flag is enabled")
end

%%
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

    %randomly sample vehicle initial 
    %Note 
    vmin = 18; %minimum velocity
    vmax = 22; %maximum velocity
    v_start = vmin + rand * (vmax-vmin); %starting velocity
    y_start = 3.7 * randi([0,2]);

%     Agent.desired_initial_condition = [10;0; 0; 20;0;0;20;0;0;0];
    Agent.desired_initial_condition = [10;y_start; 0; v_start;0;0;v_start;0;0;0];
    Agent.integrator_type= 'ode45';
    HLP = simple_highway_HLP; % high-level planner
    HLP.lookahead = hlp_lookahead; 
    AgentHelper = RDF_highwayAgentHelper(Agent,frs,HLP,'t_plan',t_plan,'t_move',t_move,'t_failsafe_move',t_failsafe_move,...
        'verbose',verbose_level, 'save_dir_prefix', save_dir_prefix, 'max_planning_time', max_planning_time, 'save_initial_condition', save_initial_condition, 'has_time_limit', true); % takes care of online planning
    AgentHelper.FRS_u0_p_maps = load("u0_p_maps.mat");

    Simulator = rlsimulator(AgentHelper,World,'plot_sim_flag',plot_sim_flag,'plot_AH_flag',plot_AH_flag,'save_result',save_result,...
        'plot_fancy_vehicle', plot_fancy_vehicle, 'save_video', save_video,'visualize', visualize, 'epscur',j, ...
        'single_plan',single_plan, 'save_dir_prefix', save_dir_prefix);

    AgentHelper.S = Simulator;
    Simulator.eval = 1; %turn on evaluation so summary will be saved

    rng(j+1);
    IsDone4 = 0;
    Simulator.epscur = j;
    Simulator.reset();
    for i = 1:num_planning_iteration
        i
        AgentHelper.planned_path = [linspace(0,1000);repmat([0;0],1,100)];
        [~,~,IsDone,LoggedSignal]=Simulator.step([rand*2-1;rand*2-1]);

        if AgentHelper.prev_action ~= 3 
            if Simulator.save_video
                    close(Simulator.videoObj);
            end
            break
        end
        if IsDone == 1 || IsDone == 3 || IsDone == 4 || IsDone == 5
            %crash
            %      crash with safety layer on
            %                      safely stopped but stuck
            %                                           reached goal!
            if i == num_planning_iteration || IsDone == 1 || IsDone == 3 || IsDone == 4
                if Simulator.save_video
                    close(Simulator.videoObj);
                end
            end
            break
        end
    end
    pause(1)
end



done = 'Simulation Complete'
