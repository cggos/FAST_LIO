#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include "use-ikfom.hpp"
#include "preprocess.h"

// IMU 处理模块：负责系统初始化、IMU 前向传播（状态预测）和点云运动畸变补偿（后向传播）
// 调用关系：laserMapping.cpp 中每帧调用 ImuProcess::Process(meas, kf, pcl_out)

/// 前置配置

// 初始化所需的最小 IMU 帧数（静止状态下收集该数量帧后完成重力/偏置初始化）
#define MAX_INI_COUNT (10)

// 点云按时间戳（curvature 字段，单位 ms）升序排列的比较函数
// 用于 UndistortPcl 中对点云排序，保证后向传播时按时间顺序处理
const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

// IMU 处理类：封装 IMU 初始化、前向积分传播与点云去畸变
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();

  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  // 设置 LiDAR-IMU 外参（LiDAR 在 IMU 体坐标系中的位姿）
  void set_extrinsic(const V3D &transl, const M3D &rot);  // 平移 + 旋转矩阵
  void set_extrinsic(const V3D &transl);                   // 仅平移，旋转置单位阵
  void set_extrinsic(const MD(4,4) &T);                    // 4×4 变换矩阵
  // 设置 IMU 噪声参数（从 yaml 读取后传入）
  void set_gyr_cov(const V3D &scaler);      // 陀螺仪测量噪声标准差（3轴）
  void set_acc_cov(const V3D &scaler);      // 加速度计测量噪声标准差
  void set_gyr_bias_cov(const V3D &b_g);   // 陀螺偏置随机游走标准差
  void set_acc_bias_cov(const V3D &b_a);   // 加速度偏置随机游走标准差

  Eigen::Matrix<double, 12, 12> Q;  // 过程噪声协方差矩阵（运行时按帧更新对角块）

  // 主处理函数：入口
  // 内部流程：初始化阶段→IMU_init；初始化完成后→UndistortPcl（前向+后向传播）
  void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);

  ofstream fout_imu;          // IMU 调试日志文件流
  V3D cov_acc;                // 加速度计噪声方差（初始化阶段由统计估计，之后由 yaml 覆盖）
  V3D cov_gyr;                // 陀螺仪噪声方差
  V3D cov_acc_scale;          // yaml 配置的加速度计噪声方差（覆盖初始化估计值）
  V3D cov_gyr_scale;          // yaml 配置的陀螺仪噪声方差
  V3D cov_bias_gyr;           // 陀螺偏置随机游走方差
  V3D cov_bias_acc;           // 加速度偏置随机游走方差
  double first_lidar_time;    // 第一帧 LiDAR 时间戳（用于相对时间计算）
  int lidar_type;             // LiDAR 类型枚举（影响 MARSIM 特殊处理分支）

 private:
  // 静止状态初始化：Welford 在线均值估计重力方向和陀螺偏置
  // 每帧调用直到 N > MAX_INI_COUNT，初始化 kf_state 的重力/偏置/协方差
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);

  // 核心函数：前向传播 + 后向点云去畸变
  // 前向：在每个 IMU 时刻调用 kf_state.predict()，同时记录 IMU 轨迹 IMUpose
  // 后向：对点云每个点，根据其时间戳在 IMUpose 中插值出对应姿态，补偿到帧末坐标系
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);

  PointCloudXYZI::Ptr cur_pcl_un_;            // 去畸变后的点云（临时存储）
  sensor_msgs::ImuConstPtr last_imu_;          // 上一帧最后一个 IMU 消息（跨帧连续传播用）
  deque<sensor_msgs::ImuConstPtr> v_imu_;      // IMU 消息队列（内部缓存）
  vector<Pose6D> IMUpose;                      // 前向传播轨迹：每个 IMU 时刻的 6DoF 位姿+加速度/角速度
                                               // 用于后向传播时为每个 LiDAR 点插值姿态
  vector<M3D>    v_rot_pcl_;                   // 点云各点对应旋转（未使用，遗留）
  M3D Lidar_R_wrt_IMU;                         // LiDAR 相对 IMU 的旋转外参 R_IL
  V3D Lidar_T_wrt_IMU;                         // LiDAR 相对 IMU 的平移外参 t_IL
  V3D mean_acc;                                // IMU 初始化阶段：加速度均值（估计重力方向）
  V3D mean_gyr;                                // IMU 初始化阶段：角速度均值（估计陀螺偏置）
  V3D angvel_last;                             // 上一 IMU 帧的偏置修正角速度（跨帧中值积分用）
  V3D acc_s_last;                              // 上一 IMU 帧的世界系加速度（含重力补偿）
  double start_timestamp_;                     // 处理起始时间戳
  double last_lidar_end_time_;                 // 上一 LiDAR 帧结束时间（用于跨帧 IMU 传播的时间对齐）
  int    init_iter_num = 1;                    // 初始化迭代计数（Welford 算法的 N）
  bool   b_first_frame_ = true;               // 是否为首帧（首帧需重置并记录 first_lidar_time）
  bool   imu_need_init_ = true;               // IMU 是否仍在初始化阶段
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 1;
  Q = process_noise_cov();
  cov_acc       = V3D(0.1, 0.1, 0.1);
  cov_gyr       = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last     = Zero3d;
  Lidar_T_wrt_IMU = Zero3d;
  Lidar_R_wrt_IMU = Eye3d;
  last_imu_.reset(new sensor_msgs::Imu());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() 
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_    = true;
  start_timestamp_  = -1;
  init_iter_num     = 1;
  v_imu_.clear();
  IMUpose.clear();
  last_imu_.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

// IMU 静止初始化：用 Welford 在线算法估计均值，初始化重力方向和陀螺偏置
// 每帧调用一次，N 为累计 IMU 样本数（跨帧累积）
// 初始化完成后设置：
//   g_W = -mean_acc/||mean_acc|| * G（重力方向与加速度方向相反）
//   b_g = mean_gyr（静止时角速度均值即为陀螺偏置）
//   外参 = 配置文件中的初始值
//   P（23×23协方差矩阵）= 对角阵，外参/偏置分量给小初值以表示已知精度
void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
  V3D cur_acc, cur_gyr;

  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    // 用第一帧 IMU 初始化均值，避免零初值的跳变
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    first_lidar_time = meas.lidar_beg_time;
  }

  // Welford 在线均值/方差算法（数值稳定，避免大数相减）
  // mean_N = mean_{N-1} + (x_N - mean_{N-1}) / N
  // var_N  = var_{N-1}*(N-1)/N + (x_N - mean_N)² * (N-1)/N²
  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    // 更新各轴方差（逐元素）
    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

    N ++;
  }

  // 用统计均值初始化滤波器状态
  state_ikfom init_state = kf_state.get_x();
  // 重力方向：加速度均值的反方向，模长归一化为 G_m_s2（9.81 m/s²）
  init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2);
  init_state.bg  = mean_gyr;          // 陀螺偏置初值 = 静止均值
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;  // LiDAR-IMU 外参（来自配置文件）
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;
  kf_state.change_x(init_state);

  // 初始化协方差矩阵 P（23×23），各分量含义：
  // [0~2]:   rot 初始方差=1（姿态未知）
  // [3~5]:   pos 初始方差=1
  // [6~8]:   offset_R_L_I 初始方差=0.00001（外参已知，较小）
  // [9~11]:  offset_T_L_I 初始方差=0.00001
  // [12~14]: vel 初始方差=1
  // [15~17]: bg 初始方差=0.0001
  // [18~20]: ba 初始方差=0.001
  // [21~22]: grav 初始方差=0.00001（重力方向已初始化，较准确）
  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
  init_P.setIdentity();
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;   // offset_R_L_I
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001; // offset_T_L_I
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001; // bg
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;  // ba
  init_P(21,21) = init_P(22,22) = 0.00001;                 // grav（S2，2维切空间）
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back();
}

// 前向传播 + 后向运动畸变补偿
// ═══════════════════════════════════════════
// 【前向传播阶段】
//   在每两个相邻 IMU 测量之间，用中值积分调用 kf_state.predict(dt, Q, in)
//   同时将每个时刻的状态保存到 IMUpose（离散轨迹）
//
// 【后向去畸变阶段】
//   将点云中每个点（采集于 t_i）补偿到帧末（t_e）的 LiDAR 坐标系：
//   P_comp = R_IL^T · [R_{WI,e}^T · (R_{WI,i}·(R_IL·P_i + t_IL) + T_ei) - t_IL]
//   其中 R_{WI,i} 由 IMUpose 插值：R_{WI,i} = R_head · Exp(ω_avr · δt)
//         T_ei = p_{WI,i} + v·δt + 0.5·a·δt² - p_{WI,e}（世界系位移差）
// ═══════════════════════════════════════════
void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
{
  // 将上一帧的最后一个 IMU 加入队列头部，确保积分覆盖从上帧结束到本帧结束
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();

  double pcl_beg_time = meas.lidar_beg_time;
  double pcl_end_time = meas.lidar_end_time;

  // MARSIM 仿真器无运动畸变，时间定义为 [上帧结束, 本帧开始]（空帧间隔）
  if (lidar_type == MARSIM) {
      pcl_beg_time = last_lidar_end_time_;
      pcl_end_time = meas.lidar_beg_time;
  }

  // 点云按 curvature（时间偏移，ms）升序排列，便于后向传播时顺序处理
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);

  // 用当前滤波器状态初始化 IMU 轨迹（时间偏移=0，对应 pcl_beg_time）
  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  // 存入第一个 pose：offset_time=0，加速度/角速度/速度/位置/旋转均来自当前状态
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

  // ──────────────────────────────────────
  // 前向传播：在每个 IMU 时间段做中值积分
  // ──────────────────────────────────────
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;
  double dt = 0;
  input_ikfom in;

  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

    // 跳过早于上帧结束时间的 IMU 段（避免重复积分）
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;

    // 中值积分：取相邻两 IMU 测量的平均值作为该时段的代表测量值
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // 将加速度计原始量纲（含未知 scale）归一化到 m/s²（用初始化阶段估计的 mean_acc.norm() 做 scale 修正）
    acc_avr = acc_avr * G_m_s2 / mean_acc.norm();

    // 计算积分时间步长（头帧若早于上帧结束，则从 last_lidar_end_time_ 起算）
    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }

    // 更新过程噪声 Q 对角块（运行时按实际配置填写）
    in.acc  = acc_avr;
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;      // σ²_ng
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;       // σ²_na
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;  // σ²_nbg
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;  // σ²_nba

    // iEKF 预测步：x̂ = x ⊞ f(x,u)·dt，P̂ = F·P·F^T + G·Q·G^T
    kf_state.predict(dt, Q, in);

    // 保存当前时刻的 IMU 状态到轨迹（供后向传播插值）
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;     // 偏置修正角速度（跨帧延续用）
    // 世界系加速度 = R·(a_m - b_a) + g
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for(int i=0; i<3; i++) acc_s_last[i] += imu_state.grav[i];

    // offset_time = 相对 pcl_beg_time 的时间偏移（秒）
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  // 补充传播到帧末（可能 LiDAR 帧结束时间晚于最后一个 IMU 时间）
  // note: pcl_end > imu_end → dt>0（正向），否则 dt<0（向回）
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in);  // 用最后一段的 in 继续传播到帧末

  imu_state = kf_state.get_x();  // 帧末最优预测状态
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  // ──────────────────────────────────────
  // 后向传播：对每个 LiDAR 点做运动畸变补偿
  // 目标：将时刻 t_i 采集的点 P_i^L 变换到帧末 t_e 时刻的 LiDAR 坐标系
  //
  // 补偿公式：
  //   R_i = R_head · Exp(ω_avr · δt_i)           （t_i 时刻的 IMU 姿态）
  //   T_ei = p_i + v_i·δt + 0.5·a·δt² - p_e      （t_i 到帧末的位移差，世界系）
  //   P_comp = R_IL^T · [R_{WI,e}^T·(R_i·(R_IL·P_i^L + t_IL) + T_ei) - t_IL]
  // ──────────────────────────────────────
  if (pcl_out.points.begin() == pcl_out.points.end()) return;

  if(lidar_type != MARSIM){  // MARSIM 仿真点云无畸变，跳过
      auto it_pcl = pcl_out.points.end() - 1;  // 从时间最晚的点开始（与 IMUpose 倒序对应）

      // 倒序遍历 IMUpose，找到点所在的 IMU 时间段 [head, tail)
      for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
      {
          auto head = it_kp - 1;
          auto tail = it_kp;
          R_imu<<MAT_FROM_ARRAY(head->rot);    // head 时刻的 IMU 旋转矩阵
          vel_imu<<VEC_FROM_ARRAY(head->vel);   // head 时刻的 IMU 速度
          pos_imu<<VEC_FROM_ARRAY(head->pos);   // head 时刻的 IMU 位置
          acc_imu<<VEC_FROM_ARRAY(tail->acc);   // 该段的世界系加速度（来自 tail）
          angvel_avr<<VEC_FROM_ARRAY(tail->gyr); // 该段的偏置修正角速度

          // 处理所有落在当前 IMU 段 [head.offset_time, ...) 内的 LiDAR 点
          for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
          {
              // 点相对 head 的时间偏移（秒）
              dt = it_pcl->curvature / double(1000) - head->offset_time;

              // t_i 时刻的 IMU 姿态（在 head 基础上积分 dt）
              M3D R_i(R_imu * Exp(angvel_avr, dt));

              V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
              // 世界系位移差 T_ei（从 t_i 到帧末 t_e）
              V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
              // 补偿公式：P_comp^L = R_IL^T·[R_{WI,e}^T·(R_i·(R_IL·P_i+t_IL) + T_ei) - t_IL]
              V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (
                  imu_state.rot.conjugate() * (
                      R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei
                  ) - imu_state.offset_T_L_I
              );

              // 覆盖写入去畸变后的坐标（仍在 LiDAR 系，但统一到帧末时刻）
              it_pcl->x = P_compensate(0);
              it_pcl->y = P_compensate(1);
              it_pcl->z = P_compensate(2);

              if (it_pcl == pcl_out.points.begin()) break;
          }
      }
  }
}

void ImuProcess::Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();

  if(meas.imu.empty()) {return;};
  ROS_ASSERT(meas.lidar != nullptr);

  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();
    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
      // ROS_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
      //          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return;
  }

  UndistortPcl(meas, kf_state, *cur_pcl_un_);

  t2 = omp_get_wtime();
  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}
