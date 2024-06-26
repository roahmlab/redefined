#include <ros/callback_queue.h>                         // for CallbackQueue
#include <rover_control_msgs/GenParamInfo.h>            // for GenParamInfo
#include <rover_control_msgs/RoverDebugStateStamped.h>  // for RoverDebugSta...
#include <sensor_msgs/Joy.h>                            // for Joy
#include <state_estimator/velocity_states.h>            // for velocity_states
#include <std_msgs/Bool.h>                              // for Bool
#include <std_msgs/Float64.h>                           // for Float64
#include <std_msgs/Int8.h>                              // for Int8
#include <tf2_msgs/TFMessage.h>                         // for TFMessage
#include <vesc_msgs/VescStateStamped.h>                 // for VescStateStamped

#include <cmath>     // for M_PI
#include <cstdlib>   // for abs
#include <iostream>  // for operator<<, endl
#include <limits>    // for numeric_limits
#include <string>    // for allocator
#include <vector>    // for vector

#include "consts.hpp"              // for kMaxServoCmdT...
#include "fl_llc.hpp"              // for RoverCmd, Rov...
#include "fmt/core.h"              // for print
#include "manu_type.hpp"           // for ManuToUint8
#include "ros/duration.h"          // for Duration, Wal...
#include "ros/forwards.h"          // for TimerEvent
#include "ros/init.h"              // for getGlobalCall...
#include "ros/node_handle.h"       // for NodeHandle
#include "ros/publisher.h"         // for Publisher
#include "ros/subscriber.h"        // for Subscriber
#include "ros/time.h"              // for Time, TimeBase
#include "ros/timer.h"             // for Timer
#include "ros_gamepad.hpp"         // for GamepadInfo
#include "rover_control_mode.hpp"  // for RoverControlMode
#include "rover_state.hpp"         // for ComplexHeadin...
#include "sensor_msgs/Imu.h"       // for Imu
#include "simple_util.hpp"         // for ClampAbs, InC...
#include "std_msgs/Header.h"       // for Header_, Head...
#include "trajectory_sym.hpp"      // for Trajectory
#include "trajectory_values.hpp"   // for GetDefaultTra...
#include "vesc_msgs/VescState.h"   // for VescState_

namespace roahm {

namespace {

/// Extracts yaw from ROS IMU message, if it was represented in RPY ordering
/// \param msg the ROS IMU message
/// \return the yaw from the message, if represented in RPY order
inline double GetImuYaw(const sensor_msgs::Imu& msg) {
  return GetOrientationYaw(msg.orientation);
}

/// Limit's the rate of change of the input value
/// \param x the variable to limit the rate of
/// \param prev_val the prior variable, at \f$ t_\text{now} - \Delta t \f$
/// \param dt the time since the prior variable \p prev_val
/// \param max_rate_of_change the maximum rate of change in units per time
/// \return \f$
/// \begin{cases}
/// x & |\frac{\textrm{d}}{\textrm{d}t} x| \leq \textrm{max rate} \\
/// \textrm{prev} + (\textrm{max rate} \cdot \Delta t) &
/// \frac{\textrm{d}}{\textrm{d}t} x > \textrm{max rate} \\
/// \textrm{prev} - (\textrm{max rate} \cdot \Delta t) &
/// \frac{\textrm{d}}{\textrm{d}t} x < -\textrm{max rate} \\
/// \end{cases}
/// \f$
inline double LimitRate(double x, double prev_val, double dt,
                        double max_rate_of_change) {
  double deriv = (x - prev_val) / dt;
  if (deriv > max_rate_of_change) {
    x = prev_val + max_rate_of_change * dt;
  } else if (deriv < -max_rate_of_change) {
    x = prev_val - max_rate_of_change * dt;
  }
  return x;
}

/// Clamps the input value to \f$ [-\textrm{abs_bound}, +\textrm{abs_bound} ]\f$
/// the provided bound must be \f$ \geq 0 \f$
/// \param val the value to clamp
/// \param abs_bound the radius of the interval to clamp to, centered at 0
/// \return \f$ \begin{cases}
/// x & x \in [-\textrm{abs_bound}, \textrm{abs_bound}] \\
/// +\textrm{abs_bound} & x > +\textrm{abs_bound} \\
/// -\textrm{abs_bound} & x < -\textrm{abs_bound}
/// \end{cases}
/// \f$
inline double ClampAbs(double val, double abs_bound) {
  // Clamps val to [-abs_bound, abs_bound]
  // abs_bound must be >= 0
  if (val > abs_bound) {
    return abs_bound;
  }
  if (val < -abs_bound) {
    return -abs_bound;
  }
  return val;
  // int lt = val < -abs_bound;
  // int gt = val > abs_bound;
  // return (val * (1 - (lt + gt))) + (abs_bound * (-lt + gt));
}

}  // namespace
double TimeSubtract(const ros::Time& a, const ros::Time& b) {
  try {
    return (a - b).toSec();
  } catch (...) {
    return std::numeric_limits<double>::infinity();
  }
}

class MessageTimeChecker {
 private:
  ros::Time local_time_;
  ros::Time remote_time_;
  double min_dt_;
  double max_dt_;
  bool first_reading_;
  bool second_reading_;
  bool is_ok_;

 public:
  MessageTimeChecker(double min_dt, double max_dt)
      : local_time_{},
        remote_time_{},
        min_dt_{min_dt},
        max_dt_{max_dt},
        first_reading_{false},
        second_reading_{false},
        is_ok_{false} {}

  bool IsOk() const { return is_ok_; }

  bool EstopCriteriaMet() const { return (not IsOk()) and HasSecondReading(); }

  bool HasSecondReading() const { return second_reading_; }

  void ResetFirstSecondRead() {
    first_reading_ = false;
    second_reading_ = false;
  }

  bool Update(ros::Time local_time_new, ros::Time remote_time_new) {
    if (not first_reading_) {
      first_reading_ = true;
    } else if (not second_reading_) {
      second_reading_ = true;
    }

    // If we haven't gotten at least two readings, consider it not okay.
    is_ok_ = second_reading_;

    double delta_local = TimeSubtract(local_time_new, local_time_);
    double delta_remote = TimeSubtract(remote_time_new, remote_time_);
    is_ok_ &= InClosedOpenInterval(delta_local, min_dt_, max_dt_);
    is_ok_ &= InClosedOpenInterval(delta_remote, min_dt_, max_dt_);
    local_time_ = local_time_new;
    remote_time_ = remote_time_new;
    return is_ok_;
  }
};

class StateTracker {
 private:
  constexpr static double kPreGenParamRampSpeed = 0.0;

  // Maximum allowable time between updates, where
  // dt in [min dt, max dt)
  constexpr static double kMaxGeneralDt = 1.5;
  constexpr static double kMaxUvrDt = kMaxGeneralDt;
  constexpr static double kMaxVescSensorsCoreDt = kMaxGeneralDt;
  constexpr static double kMaxJoyDt = kMaxGeneralDt;

  // Minimum allowable dt.
  // Setting this to 0 enforces that no negative dt's
  // are allowed.
  constexpr static double kMinDt = 0.0;
  // Conversion factors, all multipliers in the form kXToY
  // such that {val in unit X} * kXToY = {val in unit Y}
  constexpr static double kMetersPerSecToRpm = 2272.0;
  constexpr static double kSteeringAngleToServoGain = -0.84;
  constexpr static bool kUseCartographerForHeading = true;

  // Times that the last update came in from given topics,
  // used to check if it has been an unsafe amount of time
  // since receiving an update, and stop if it has.
  MessageTimeChecker joy_checker_;
  MessageTimeChecker uvr_checker_;
  MessageTimeChecker vesc_core_checker_;

  RoverState state_;
  ros::Time traj_start_time_;
  ros::Duration traj_t0_offset_;
  Trajectory traj_info_;
  ros::Publisher vel_heading_pub_;
  // State Publishers
  ros::Publisher rover_debug_state_pub_;
  ros::Publisher rover_debug_u_pub_;
  // Drive Mode
  ros::Publisher drive_mode_pub_;
  // Command Publishers
  ros::Publisher speed_pub_;
  ros::Publisher current_pub_;
  ros::Publisher servo_pub_;
  ros::Publisher brake_pub_;
  ros::Publisher fp_req_pub_;
  GamepadInfo gamepad_state_;
  std::vector<Trajectory> trajectories_;
  std::size_t traj_gen_idx_;

  double servo_offset_;
  double out_speed_m_s_prev_;
  double out_servo_rad_prev_;
  double hd_prev_;
  RoverControlMode control_mode_;
  RoverDriveMode drive_mode_;
  bool deadman_held_;
  bool is_estopped_;
  bool driving_enabled_;
  bool w_near_ramp_;
  bool have_requested_new_params_;
  bool have_received_new_params_;
  bool quit_gen_param_;
  bool trajectories_modified_since_init_;
  ros::Time w_near_ramp_t0_;

  void RequestNewTrajectoryParameters() {
    // Requesting new parameters
    have_requested_new_params_ = true;
    have_received_new_params_ = false;

    rover_control_msgs::RoverDebugStateStamped out_msg{};
    out_msg.u0 = traj_info_.u0_;
    out_msg.h0 = traj_info_.h0_;
    out_msg.au = traj_info_.au_;
    out_msg.ay = traj_info_.ay_;
    out_msg.manu_type = ManuToUint8(traj_info_.manu_type_);
    out_msg.running_auto = IsRunningAuto(control_mode_);
    out_msg.deadman_held = deadman_held_;
    out_msg.estopped = is_estopped_;
    out_msg.driving_enabled = driving_enabled_;

    out_msg.x = state_.x_;
    out_msg.y = state_.y_;
    out_msg.h = state_.GetHeading();
    out_msg.u = state_.u_;
    out_msg.v = state_.v_;
    out_msg.r = state_.r_;
    out_msg.w = state_.w_;
    out_msg.r_err_sum = state_.r_err_sum_;
    out_msg.h_err_sum = state_.h_err_sum_;

    fp_req_pub_.publish(out_msg);
  }

  void PublishDouble(ros::Publisher& pub, double val) {
    std_msgs::Float64 float_msg;
    float_msg.data = val;
    pub.publish(float_msg);
  }

  void DecrementTrajGenIdx() {
    if (traj_gen_idx_ == 0 and not trajectories_.empty()) {
      traj_gen_idx_ = trajectories_.size() - 1;
    } else {
      --traj_gen_idx_;
    }
  }

  void IncrementTrajGenIdx() {
    ++traj_gen_idx_;
    if (not trajectories_.empty()) {
      traj_gen_idx_ %= trajectories_.size();
    } else {
      traj_gen_idx_ = 0;
    }
  }

  void LoadNextTrajectoryInfo() {
    Trajectory traj_info = (traj_gen_idx_ < trajectories_.size())
                               ? trajectories_[traj_gen_idx_]
                               : Trajectory{};
    traj_info_ = traj_info;
  }

 public:
  explicit StateTracker(ros::NodeHandle& nh)
      : joy_checker_{kMinDt, kMaxJoyDt},
        uvr_checker_{kMinDt, kMaxUvrDt},
        vesc_core_checker_{kMinDt, kMaxVescSensorsCoreDt},
        state_{},
        traj_start_time_{},
        traj_t0_offset_{},
        traj_info_{},
        vel_heading_pub_{
            nh.advertise<std_msgs::Float64>("/state_out/vel_heading", 1)},
        rover_debug_state_pub_{
            nh.advertise<rover_control_msgs::RoverDebugStateStamped>(
                "/state_out/rover_debug_state_out", 1)},
        rover_debug_u_pub_{
            nh.advertise<std_msgs::Float64>("/state_out/callback_u", 1)},
        drive_mode_pub_{nh.advertise<std_msgs::Int8>("/drive_mode", 1)},
        speed_pub_{
            nh.advertise<std_msgs::Float64>("/vesc/commands/motor/speed", 1)},
        current_pub_{
            nh.advertise<std_msgs::Float64>("/vesc/commands/motor/current", 1)},
        servo_pub_{nh.advertise<std_msgs::Float64>(
            "/vesc/commands/servo/position", 1)},
        brake_pub_{
            nh.advertise<std_msgs::Float64>("/vesc/commands/motor/brake", 1)},
        fp_req_pub_{nh.advertise<rover_control_msgs::RoverDebugStateStamped>(
            "/fp_req", 1)},
        gamepad_state_{},
        trajectories_{},
        traj_gen_idx_{0},
        servo_offset_{},
        out_speed_m_s_prev_{},
        out_servo_rad_prev_{},
        hd_prev_{},
        control_mode_{RoverControlMode::kTeleop},
        drive_mode_{RoverDriveMode::kSpeedControl},
        deadman_held_{false},
        is_estopped_{false},
        driving_enabled_{false},
        w_near_ramp_{false},
        have_requested_new_params_{false},
        have_received_new_params_{false},
        quit_gen_param_{false},
        trajectories_modified_since_init_{false},
        w_near_ramp_t0_{} {
    constexpr double kDefaultServoOffset = 0.5;
    const std::string servo_offset_topic = "/roahm/servo_offset";
    nh.param<double>(servo_offset_topic, servo_offset_, kDefaultServoOffset);
    FillTrajectories();
  }

  void FillTrajectories() {
    trajectories_modified_since_init_ = false;
    traj_gen_idx_ = 0;
    trajectories_.clear();
    trajectories_ = GetDefaultTrajectories();
  }

  void StartNextTrajectory(bool increment, bool use_hd_as_h0) {
    fmt::print("StartNextTrajectory\n");
    traj_info_.h0_ = use_hd_as_h0 ? hd_prev_ : state_.GetHeading();
    if (traj_info_.manu_type_ != ManuType::kSpdChange) {
      traj_info_.u0_ = traj_info_.u0_goal_;
    } else {
      traj_info_.u0_ = state_.u_;
    }

    // Reset error sums between trajectories
    state_.h_err_sum_ = 0;
    state_.r_err_sum_ = 0;
    state_.u_err_sum_ = 0;

    fmt::print("Compute new\n ay: {}\n au: {}\n u0: {}\n", traj_info_.ay_,
               traj_info_.au_, traj_info_.u0_);
    if (not traj_info_.HasValidType()) {
      fmt::print("INVALID MANU TYPE {}\n", ManuToUint8(traj_info_.manu_type_));
    }

    // Reset current trajectory start time
    traj_start_time_ = ros::Time::now();
    traj_t0_offset_ = ros::Duration{traj_info_.t0_offset_};

    if (increment) {
      IncrementTrajGenIdx();
    }
  }
  void StartNextTrajectory(bool increment) {
    StartNextTrajectory(increment, false);
  }
  void StartNextTrajectory() { StartNextTrajectory(true); }

  void CallbackTf(const tf2_msgs::TFMessage& msg) {
    for (const auto& trans : msg.transforms) {
      if (trans.header.frame_id == "map" or trans.header.frame_id == "/map") {
        if (trans.child_frame_id == "base_link" or
            trans.child_frame_id == "/base_link") {
          state_.x_ = trans.transform.translation.x;
          state_.y_ = trans.transform.translation.y;
          const auto& q_in = trans.transform.rotation;
          if constexpr (kUseCartographerForHeading) {
            const double yaw = ToAngle(GetOrientationYaw(q_in) - (M_PI / 2.0));
            state_.SetHeading(yaw);
          }
        }
      }
    }
  }

  void CallbackGenParamOut(const rover_control_msgs::GenParamInfo& msg) {
    double au = msg.au;
    double ay = msg.ay;
    double t0_offset = msg.t0_offset;
    ManuType manu_type = ManuFromUint8(msg.manu_type);
    have_received_new_params_ = true;
    trajectories_.emplace_back(au, ay, au, au, manu_type, t0_offset);
  }

  void CallbackJoy(const sensor_msgs::Joy& msg) {
    joy_checker_.Update(ros::Time::now(), msg.header.stamp);
    gamepad_state_.Update(msg);
    deadman_held_ = gamepad_state_.l_bump_.Held();
    if (gamepad_state_.r_bump_.Pressed()) {
      EstopReset();
    }
    if (gamepad_state_.start_.Pressed()) {
      if (control_mode_ != RoverControlMode::kTeleop) {
        SwitchToMode(RoverControlMode::kTeleop);
      }
    }
    if (gamepad_state_.a_.Pressed()) {
      if (trajectories_modified_since_init_) {
        FillTrajectories();
      }
      LoadNextTrajectoryInfo();
      SwitchToMode(RoverControlMode::kRampUp);
    }
    if (gamepad_state_.x_.Pressed()) {
      if (trajectories_modified_since_init_) {
        FillTrajectories();
      }
      LoadNextTrajectoryInfo();
      SwitchToMode(RoverControlMode::kRampChain);
    }
    // While driving:
    //   (1) request new trajectory
    //   (2) while waiting
    //      keep running old trajectory
    //   (3) on new trajectory received
    //      start running it
    if (gamepad_state_.y_.Pressed()) {
      trajectories_modified_since_init_ = true;
      trajectories_.clear();
      quit_gen_param_ = false;
      have_requested_new_params_ = false;
      traj_info_.u0_goal_ = state_.u_;
      w_near_ramp_ = false;
      w_near_ramp_t0_ = ros::Time::now();
      RequestNewTrajectoryParameters();
      SwitchToMode(RoverControlMode::kRampGenParam);
    }
    if (gamepad_state_.dpad_up_.Pressed() and
        (control_mode_ == RoverControlMode::kNone or
         control_mode_ == RoverControlMode::kTeleop)) {
      IncrementTrajGenIdx();
    }
    if (gamepad_state_.dpad_down_.Pressed() and
        (control_mode_ == RoverControlMode::kNone or
         control_mode_ == RoverControlMode::kTeleop)) {
      DecrementTrajGenIdx();
    }
    if (gamepad_state_.dpad_left_.Pressed()) {
      // Switch to current control mode
      drive_mode_ = RoverDriveMode::kCurrentControl;
      std_msgs::Int8 drive_mode_msg;
      drive_mode_msg.data = 1;
      drive_mode_pub_.publish(drive_mode_msg);
    }
    if (gamepad_state_.dpad_right_.Pressed()) {
      // Switch to speed control mode to go backwards in manual operation
      drive_mode_ = RoverDriveMode::kSpeedControl;
      std_msgs::Int8 drive_mode_msg;
      drive_mode_msg.data = 0;
      drive_mode_pub_.publish(drive_mode_msg);
    }
  }
  void CheckForEstop() {
    // Verifies that all of the required signals
    // have been updated within an acceptable amount of
    // time. If not, this indicates a problem.

    // Note: this function compares timestamps from incoming
    // messages to local ROS time, which may have some
    // offset/lag.

    // Verify that each dt is in [min dt, max dt) after messages
    //   have started flowing
    // X_ok means that *estop* should not trigger based on the X criteria alone,
    //   however it does not mean that driving should necessarily be enabled.
    const bool joy_bad = joy_checker_.EstopCriteriaMet();
    const bool uvr_bad = uvr_checker_.EstopCriteriaMet();
    const bool vesc_sensors_core_bad = vesc_core_checker_.EstopCriteriaMet();
    is_estopped_ = is_estopped_ or uvr_bad or joy_bad or vesc_sensors_core_bad;
    if (is_estopped_) {
      if (uvr_bad) {
        fmt::print("uvr_bad:               {}\n", uvr_bad);
      }
      if (joy_bad) {
        fmt::print("joy_bad:               {}\n", joy_bad);
      }
      if (vesc_sensors_core_bad) {
        fmt::print("vesc_sensors_core_bad: {}\n", vesc_sensors_core_bad);
      }
    }
  }
  void CallbackVescSensorsCore(const vesc_msgs::VescStateStamped& msg) {
    vesc_core_checker_.Update(ros::Time::now(), msg.header.stamp);
    state_.w_ = msg.state.speed / kMetersPerSecToRpm;
  }
  void UpdateStateUvr(ros::Time t, double u, double v, double r) {
    uvr_checker_.Update(ros::Time::now(), t);
    state_.u_ = u;
    state_.v_ = v;
    state_.r_ = r;
  }
  void CallbackUvrEstimator(const state_estimator::velocity_states& msg) {
    std_msgs::Float64 msg_float;
    msg_float.data = msg.u;
    rover_debug_u_pub_.publish(msg_float);
    UpdateStateUvr(msg.header.stamp, msg.u, msg.v, msg.r);
  }

  void CallbackControlLoop(const ros::TimerEvent& timer_event) {
    auto dt_real =
        TimeSubtract(timer_event.current_real, timer_event.last_real);
    ControlLoop(dt_real);
  }
  void CallbackEstopReset(const std_msgs::Bool& msg) { EstopReset(); }

  void EstopReset() {
    traj_info_ = decltype(traj_info_){};
    is_estopped_ = false;
    vesc_core_checker_.ResetFirstSecondRead();
    uvr_checker_.ResetFirstSecondRead();
    joy_checker_.ResetFirstSecondRead();
  }

  void SwitchToMode(RoverControlMode new_mode) { control_mode_ = new_mode; }

  void ControlLoop(double dt) {
    // Sometimes we get error in our control loop timing or run slightly past
    // the end, so clamp all t in [t_max, t_max + delta] = t_max
    // to just continue the previous command for an iteration or two
    const double traj_max_t = 1.5;
    constexpr double kTrajMaxTimeThresh = 0.03;
    const double traj_max_with_thresh = traj_max_t + kTrajMaxTimeThresh;

    const bool is_current_mode = drive_mode_ == RoverDriveMode::kCurrentControl;
    const bool is_speed_mode = drive_mode_ == RoverDriveMode::kSpeedControl;

    // Time that the trajectory has been running
    double traj_run_time = TimeSubtract(ros::Time::now(), traj_start_time_);
    if (InClosedInterval(traj_run_time, traj_max_t, traj_max_with_thresh)) {
      traj_run_time = traj_max_t;
    }

    // Time to use for trajectory lookup. Since we do not always start at
    // t0 = 0s, this can differ from traj_run_time by a constant offset.
    const double trajectory_time = traj_run_time + traj_t0_offset_.toSec();

    CheckForEstop();
    driving_enabled_ = (!is_estopped_) and deadman_held_ and
                       vesc_core_checker_.HasSecondReading() and
                       joy_checker_.HasSecondReading() and
                       ((control_mode_ == RoverControlMode::kTeleop) or
                        uvr_checker_.HasSecondReading());
    RoverCmd cmd{};
    rover_control_msgs::RoverDebugStateStamped out_msg{};
    out_msg.traj_time = trajectory_time;
    out_msg.u0 = traj_info_.u0_;
    out_msg.h0 = traj_info_.h0_;
    out_msg.au = traj_info_.au_;
    out_msg.ay = traj_info_.ay_;
    out_msg.manu_type = ManuToUint8(traj_info_.manu_type_);
    out_msg.control_dt = dt;
    if (driving_enabled_) {
      constexpr double kMaxSpeedDeltaPerSec = 4.0;             // [(m/s)/s]
      constexpr double kRampUpSpeedDeltaPerSec = 0.4;          // [(m/s)/s]
      constexpr double kMaxSpeedMeterPerSec = 2.5;             // [m/s]
      constexpr double kMaxServoDeltaPerSec = 3.0;             // [rad/s]
      constexpr double kMaxServoDeltaPerSecTeleop = 3.0;       // [rad/s]
      constexpr double kMaxCurrentDraw = 50.0;                 // [amps]
      constexpr double kTeleopCurrentAmplificationConst = 20;  // [rad/s]

      if (IsRampMode(control_mode_)) {
        constexpr double kNearRampGoalThreshold = 0.05;  // [m/s]
        const double signed_dist_to_ramp_goal = state_.w_ - traj_info_.u0_goal_;
        const double dist_to_ramp_goal = std::abs(signed_dist_to_ramp_goal);
        if (dist_to_ramp_goal <= kNearRampGoalThreshold or
            (signed_dist_to_ramp_goal > -0.05 and is_current_mode)) {
          if (!w_near_ramp_) {
            w_near_ramp_ = true;
            w_near_ramp_t0_ = ros::Time::now();
          }

          const double time_near_ramp_speed =
              TimeSubtract(ros::Time::now(), w_near_ramp_t0_);
          constexpr double kMinTimeNearRamp = 0.0;
          if (time_near_ramp_speed > kMinTimeNearRamp or
              std::abs(traj_info_.u0_goal_) <= 1.0e-5) {
            if (control_mode_ == RoverControlMode::kRampGenParam) {
              constexpr double kMaxTimeBeforeRampGenParam = 1.50;
              if (!have_requested_new_params_) {
                // Haven't requested new parameters... requesting now
                trajectories_.clear();
                RequestNewTrajectoryParameters();
              } else if (have_received_new_params_) {
                // Parameters received: Switching to EXEC GEN PARAM
                SwitchToMode(RoverControlMode::kExecGenParam);
                if (not trajectories_.empty()) {
                  traj_gen_idx_ = trajectories_.size() - 1;
                } else {
                  traj_gen_idx_ = 0;
                }
                LoadNextTrajectoryInfo();
                StartNextTrajectory(false, false);
                RequestNewTrajectoryParameters();
              } else if (time_near_ramp_speed > kMaxTimeBeforeRampGenParam) {
                // No parameters received: switching to NONE
                SwitchToMode(RoverControlMode::kNone);
              }
            }
          }
        } else {
          w_near_ramp_ = false;
        }

        // Will be automatically rate limited
        if (is_current_mode) {
          cmd.w_ = 20;  // amps
        } else if (is_speed_mode) {
          cmd.w_ = traj_info_.u0_goal_;
        }
        cmd.delta_ = 0.0;
      } else if (IsExecMode(control_mode_)) {
        if (traj_info_.HasValidType() > 0) {
          cmd = FeedbackLinController(state_, dt, trajectory_time, traj_info_,
                                      drive_mode_);

          //
          // Convert from desired wheel speed to current output (amps)
          //

          // (((1.0 / (r_w * w_cmd / u) + 1.0) * (mu_bar * m * g * l_r) / l)
          // + fric) / kt
          const double K_t = 0.4642;
          const double c_fric_1 = 8.0897;
          const double c_fric_2 = 10.5921;
          const double mu_bar = RtdConsts::kMuBar;
          const double g = RtdConsts::kGravConst;
          const double m = RtdConsts::kRoverMass;
          const double l_r = RtdConsts::kLr;
          const double L = RtdConsts::kLengthTotal;
          const double r_w = RtdConsts::kWheelRadius;
          const double u = state_.u_;
          const double numerator1 =
              (mu_bar * m * g * l_r) * (1.0 - (u / (cmd.w_ * r_w)));
          const double fric = c_fric_1 * std::tanh(c_fric_2 * state_.u_);
          const double numerator2 = fric + (numerator1 / L);
          cmd.w_ = numerator2 / K_t;
          // TODO keep?
          if (u < 0.05) {
            cmd.delta_ = 0;
          }
          // TODO why?
          if (cmd.ud_ == 0.0) {
            cmd.w_ = -1.0;
          }
        }

        const double max_traj_time = traj_info_.MaxTrajTime();
        const bool exit_exec = trajectory_time > max_traj_time;

        if (control_mode_ == RoverControlMode::kExecGenParam) {
          const double pre_brake_time = GetPreBrakeTime(traj_info_.manu_type_);
          const bool has_run_non_brake_traj = trajectory_time > pre_brake_time;
          if (control_mode_ == RoverControlMode::kExecChain) {
            fmt::print("Time: {}/{}\n", trajectory_time, max_traj_time);

            // currently the idx is increased *after* each new traj, so 0 comes
            // after we have selected the last trajectory
            const bool is_last_traj = (traj_gen_idx_ == 0);
            const bool should_brake =
                (trajectories_.size() <= 1) or is_last_traj;
            if (has_run_non_brake_traj and not should_brake and not exit_exec) {
              LoadNextTrajectoryInfo();
              // Don't stitch hd together since FRS is currently not generated
              // with any hd error
              StartNextTrajectory(true, false);
            }
          } else if (control_mode_ == RoverControlMode::kExecGenParam) {
            if (traj_run_time > 1.50 and not quit_gen_param_) {
              // We should already have next trajectory, so ask for the
              // trajectory following the next one. if
              // (!have_requested_new_params_) {
              //  RequestNewTrajectoryParameters();
              //}

              const bool is_more_trajectories =
                  (traj_gen_idx_ + 1) < trajectories_.size();
              fmt::print("Traj gen idx:  {}\n", traj_gen_idx_);
              fmt::print("Traj Gen Inf:  {}\n", trajectories_.size());
              fmt::print("Is More Traj:  {}\n", is_more_trajectories);
              if (is_more_trajectories) {
                have_requested_new_params_ = false;
                IncrementTrajGenIdx();
                LoadNextTrajectoryInfo();
                StartNextTrajectory(false, false);
                RequestNewTrajectoryParameters();
              } else {
                quit_gen_param_ = true;
                fmt::print("Quitting gen param. Params not received in time\n");
              }
            }
          }

          if (exit_exec) {
            SwitchToMode(RoverControlMode::kNone);
          }

          // Write desired states to msg output
          out_msg.ud = cmd.ud_;
          out_msg.uddot = cmd.uddot_;
          out_msg.vd = cmd.vd_;
          out_msg.vddot = cmd.vddot_;
          out_msg.rd = cmd.rd_;
          out_msg.rddot = cmd.rddot_;
          out_msg.hd = cmd.hd_;
          hd_prev_ = cmd.hd_;
        } else if (control_mode_ == RoverControlMode::kTeleop) {
          const double left_y = gamepad_state_.left_stick_y_;
          const double right_x = gamepad_state_.right_stick_x_;
          if (is_speed_mode) {
            cmd.w_ = left_y * kMaxSpeedMeterPerSec;
          } else if (is_current_mode) {
            cmd.w_ = left_y * kTeleopCurrentAmplificationConst;
          }
          cmd.delta_ = -right_x * RtdConsts::kMaxServoCmd;
        }

        if (control_mode_ == RoverControlMode::kNone) {
          cmd.w_ = 0.0;
          cmd.delta_ = 0.0;
        } else {
          if (is_speed_mode) {
            const double spd_rate_limit = IsRampMode(control_mode_)
                                              ? kRampUpSpeedDeltaPerSec
                                              : kMaxSpeedDeltaPerSec;
            // Limit maximum speed rate of change [m/(s^2)]
            cmd.w_ = LimitRate(cmd.w_, out_speed_m_s_prev_, dt, spd_rate_limit);
            // Limit maximum overall speed [m/s]
            cmd.w_ = ClampAbs(cmd.w_, kMaxSpeedMeterPerSec);
          } else if (is_current_mode) {
            cmd.w_ = ClampAbs(cmd.w_, kMaxCurrentDraw);
          }

          const double srv_rate_limit =
              (control_mode_ == RoverControlMode::kTeleop)
                  ? kMaxServoDeltaPerSecTeleop
                  : kMaxServoDeltaPerSec;
          // Limit maximum rate of change [rad/s]
          cmd.delta_ =
              LimitRate(cmd.delta_, out_servo_rad_prev_, dt, srv_rate_limit);
          // Limit maximum overall command [rad]
          cmd.delta_ = ClampAbs(cmd.delta_, RtdConsts::kMaxServoCmd);
        }
      } else {
        if (deadman_held_) {
          const bool vesc_second_reading =
              vesc_core_checker_.HasSecondReading();
          const bool uvr_second_reading = uvr_checker_.HasSecondReading();
          const bool joy_second_reading = joy_checker_.HasSecondReading();
        }
        cmd.w_ = 0.0;
        cmd.delta_ = 0.0;
      }

      out_msg.running_auto = IsRunningAuto(control_mode_);
      out_msg.deadman_held = deadman_held_;
      out_msg.estopped = is_estopped_;
      out_msg.driving_enabled = driving_enabled_;

      out_msg.x = state_.x_;
      out_msg.y = state_.y_;
      out_msg.h = state_.GetHeading();
      out_msg.u = state_.u_;
      out_msg.v = state_.v_;
      out_msg.r = state_.r_;
      out_msg.w = state_.w_;
      out_msg.r_err_sum = state_.r_err_sum_;
      out_msg.h_err_sum = state_.h_err_sum_;

      // Store outputs
      out_msg.w_cmd = cmd.w_;

      out_msg.delta_cmd = cmd.delta_;
      out_speed_m_s_prev_ = cmd.w_;
      out_servo_rad_prev_ = cmd.delta_;

      // Convert commands to proper units and send them
      std_msgs::Float64 speed_msg;
      std_msgs::Float64 current_msg;
      std_msgs::Float64 servo_msg;

      servo_msg.data = cmd.delta_ * kSteeringAngleToServoGain + servo_offset_;
      if (is_current_mode) {
        current_msg.data = cmd.w_;
        if (current_msg.data <= 0) {
          brake_pub_.publish(current_msg);
        } else {
          current_pub_.publish(current_msg);
        }
      } else if (is_speed_mode) {
        speed_msg.data = cmd.w_ * kMetersPerSecToRpm;
        speed_pub_.publish(speed_msg);
      }

      servo_pub_.publish(servo_msg);
      out_msg.header.stamp = ros::Time::now();
      rover_debug_state_pub_.publish(out_msg);
    }
  }
};
}  // namespace roahm
int main(int argc, char** argv) {
  ros::init(argc, argv, "rover_controller");
  ros::NodeHandle nh;
  roahm::StateTracker state_tracker(nh);
  fmt::print("Loading...\n");

  ros::Subscriber sub_tf =
      nh.subscribe("/tf", 3, &roahm::StateTracker::CallbackTf, &state_tracker);
  ros::Subscriber sub_reset_estop =
      nh.subscribe("/CtrlTest/EstopReset", 1,
                   &roahm::StateTracker::CallbackEstopReset, &state_tracker);
  ros::Subscriber sub_gen_param_out =
      nh.subscribe("/gen_param_out", 1,
                   &roahm::StateTracker::CallbackGenParamOut, &state_tracker);
  ros::Subscriber sub_joy = nh.subscribe(
      "/vesc/joy", 1, &roahm::StateTracker::CallbackJoy, &state_tracker);
  ros::Subscriber sub_vesc_sensors_core = nh.subscribe(
      "/vesc/sensors/core", 1, &roahm::StateTracker::CallbackVescSensorsCore,
      &state_tracker);
  ros::Subscriber sub_state_est =
      nh.subscribe("/state_estimator/states", 1,
                   &roahm::StateTracker::CallbackUvrEstimator, &state_tracker);
  ros::Timer timer =
      nh.createTimer(ros::Duration(::roahm::RtdConsts::kControlLoopDt),
                     &roahm::StateTracker::CallbackControlLoop, &state_tracker);

  fmt::print("Done Loading\n");
  ros::spin();
}
