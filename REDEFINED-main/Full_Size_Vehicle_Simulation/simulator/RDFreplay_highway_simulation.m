% Main function to replay REDEFINE simulation trials

% load FRS
frs_filename = 'car_frs.mat';
if ~exist('frs','var')
    disp('Loading frs')
    frs = load(frs_filename) ;
else
    disp('table already loaded') ;
end

visualize = 1; %need to turn off to stop visualization
plot_sim_flag = visualize; %if visualize is on plot_sim_flag should also be on
plot_AH_flag = 1;
save_result = false; % make it true if you want to save the simulation data
save_video = true; %make it true if you want to save videos of the trials
plot_fancy_vehicle = true;

replay_mode = 1; %flag that will replay a previous trial using the selected K_logs

% switch with path to folder that contains trial logs. This will make a directory struct that contains info on all the files in that folder
log_dir = dir('./log_dirs'); 

%list of trials to replay
trials_to_replay = [];
number_of_trials = length(trials_to_replay); %number of trials you want to replay

%initalize array to store indexs in log_dir that correspond to the listed
%trial numbers in trials_to_replay
trial_idx = [];

disp('Getting together the set of trials we want to replay')
%index for counting number of trials
tic
for kk = 1:length(trials_to_replay)
    for i = 3:length(log_dir) %start from 3 because typically the first two entries are not actual files
        desired_string = extractBetween(log_dir(i).name,'-SimID_','.mat');
        if isequal(desired_string{1,1},num2str(trials_to_replay(1,kk)))
            trial_idx = [trial_idx,i];
            continue
        end
    end
end
toc

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
num_moving_cars = 15;
num_static_cars = 3;
num_total_cars = num_ego_vehicles + num_moving_cars + num_static_cars;
hlp_lookahead = 90;


for j = 1:number_of_trials 

    trial_replaydata = load([log_dir(trial_idx(j)).folder,'/',log_dir(trial_idx(j)).name]); %load the file for each trial number in trials_to_replay
    num_moving_cars = sum(trial_replaydata.envCars(:,2) > 0);
    num_static_cars = sum(trial_replaydata.envCars(:,2) == 0) - 1;
    num_total_cars = num_ego_vehicles + num_moving_cars + num_static_cars;

    % RESET simulation environment
    World = dynamic_car_world( 'bounds', bounds, ...
        'buffer', world_buffer, 'goal', [1010;3.7], ...
        'verbose', verbose_level, 'goal_radius', goal_radius, ...
        'num_cars', num_total_cars, 'num_moving_cars', num_moving_cars, ...
        't_move_and_failsafe', t_move+t_failsafe_move) ;

    Agent = highway_cruising_10_state_agent; % takes care of vehicle states and dynamics
    Agent.desired_initial_condition = [10;0; 0; 20;0;0;20;0;0;0];
    Agent.integrator_type= 'ode45';
    HLP = simple_highway_HLP; % high-level planner
    HLP.lookahead = hlp_lookahead; 
    AgentHelper = RDF_highwayAgentHelper(Agent,frs,HLP,'t_plan',t_plan,'t_move',t_move,'t_failsafe_move',t_failsafe_move,...
        'verbose',verbose_level,'replay_mode',replay_mode,'trial_replay_hist',trial_replaydata.hist_info); % takes care of online planning
    AgentHelper.FRS_u0_p_maps = load("u0_p_maps.mat");
    Simulator = rlsimulator(AgentHelper,World,'plot_sim_flag',plot_sim_flag,'plot_AH_flag',plot_AH_flag,'save_result',save_result,...
        'plot_fancy_vehicle', plot_fancy_vehicle, 'save_video', save_video,'visualize', visualize, 'epscur',trials_to_replay(j));

    AgentHelper.S = Simulator;
    Simulator.eval = 1; %turn on evaluation so summary will be saved

   

    rng(j+1);
    IsDone4 = 0;
    Simulator.epscur = j;
    Simulator.reset();
    World.envCars = trial_replaydata.envCars; %overwrite world setup so that you can setup according to replay trial data

    for i = 1:4000
        i
        AgentHelper.planned_path = [linspace(0,1000);repmat([0;0],1,100)];
        [~,~,IsDone,LoggedSignal]=Simulator.step([rand*2-1;rand*2-1]);
        %if previous action was not a lane change (prev_action~=3), then
        %break so that it is only 1 planning iteration
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



done = 'Simulation Complete'
