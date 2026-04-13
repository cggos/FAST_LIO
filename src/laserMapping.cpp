// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

// iEKF 初始化等待时间（秒）：前 0.1s 的帧只做 IMU 初始化，不进行地图匹配
#define INIT_TIME           (0.1)
// LiDAR 点测量噪声方差（m²），对应 iEKF 测量协方差 R = LASER_POINT_COV * I
// 值越小表示点面配准越可信，对状态更新影响越大
#define LASER_POINT_COV     (0.001)
// 时间日志数组最大帧数（约 720000 帧 = ~20 小时@10Hz）
#define MAXN                (720000)
// 每隔 PUBFRAME_PERIOD 帧发布一次点云（避免发布频率过高占用带宽）
#define PUBFRAME_PERIOD     (20)

/*** 性能计时变量（runtime_pos_log=true 时启用） ***/
double kdtree_incremental_time = 0.0;   // ikd-Tree 增量插入耗时
double kdtree_search_time = 0.0;        // ikd-Tree KNN 搜索耗时
double kdtree_delete_time = 0.0;        // ikd-Tree 地图裁剪删除耗时
// 各帧时间序列（s_plot 系列用于 matplotlibcpp 可视化）
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0;          // h_share_model 中 KNN+平面拟合耗时累计
double solve_time = 0;          // H 矩阵计算耗时累计
double solve_const_H_time = 0;  // 常数 H 迭代耗时（未使用）
int    kdtree_size_st = 0;       // 每帧开始时 ikd-Tree 节点数
int    kdtree_size_end = 0;      // 每帧结束时 ikd-Tree 节点数
int    add_point_size = 0;       // 本帧新增点数
int    kdtree_delete_counter = 0; // 本帧删除点数
// 运行时标志位
bool   runtime_pos_log = false;  // 是否记录详细时间/位姿日志（yaml: runtime_pos_log_enable）
bool   pcd_save_en = false;      // 是否保存 PCD 文件（yaml: pcd_save/pcd_save_en）
bool   time_sync_en = false;     // 是否启用软件时间同步（yaml: common/time_sync_en）
bool   extrinsic_est_en = true;  // 是否在线估计 LiDAR-IMU 外参（yaml: mapping/extrinsic_est_en）
bool   path_en = true;           // 是否发布路径 topic
/**************************/

float res_last[100000] = {0.0};      // 上一轮迭代各点的点面距离（用于有效点筛选）
float DET_RANGE = 300.0f;            // 地图管理探测范围（m），yaml: mapping/det_range
const float MOV_THRESHOLD = 1.5f;    // 地图滑动触发阈值倍数（距离边界 < 1.5*DET_RANGE 时触发滑动）
double time_diff_lidar_to_imu = 0.0; // LiDAR 相对 IMU 的时间偏移（s），yaml: common/time_offset_lidar_to_imu

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
int lidar_type;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

// 点坐标变换：LiDAR 体坐标系 → 世界系（使用传入的 iEKF 状态）
// p_W = R_WI · (R_IL · p_L + t_IL) + p_WI
// 用于 h_share_model 中每次迭代用当前估计状态变换点
void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

// 点坐标变换：LiDAR 体坐标系 → 世界系（使用全局 state_point，即最新收敛状态）
// 用于 map_incremental / publish_frame 等模块
void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;       // 当前局部地图的三维边界框（立方体）
bool Localmap_Initialized = false;  // 局部地图是否已完成首次初始化

// 滑动窗口地图管理：维护以当前 LiDAR 位置为中心的立方体局部地图
// 当传感器接近地图边界（距离 < MOV_THRESHOLD * DET_RANGE = 1.5 * det_range）时，
// 将地图窗口整体平移，并从 ikd-Tree 中删除超出范围的旧点
// 作用：防止地图无限增长，保持 KNN 搜索效率
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    // 将 X 轴参考点从 LiDAR 系转到世界系（用于 FOV 方向判断，实际此处主要用 pos_lid）
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;  // 当前 LiDAR 在世界系的位置

    // 首次初始化：以当前位置为中心创建 cube_len × cube_len × cube_len 的地图边界
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }

    // 计算当前位置到地图各边界的距离
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);  // 到负方向边界的距离
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);  // 到正方向边界的距离
        // 任一方向距离过近则触发地图滑动
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;

    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    // 计算每次移动的距离（保证移动后仍有足够余量）
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            // 靠近负方向边界：地图向负方向移动，删除正方向超出部分
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;  // 要删除的 box
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            // 靠近正方向边界：地图向正方向移动，删除负方向超出部分
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;  // 要删除的 box
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();  // 回收被删除点（当前为空操作）
    double delete_begin = omp_get_wtime();
    // 从 ikd-Tree 中批量删除超出地图范围的点
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

// LiDAR 回调（Velodyne / Ouster / MARSIM：标准 PointCloud2 格式）
// 调用 Preprocess::process 解析点云并统一时间戳，压入 lidar_buffer
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    // 时间戳回跳检测（rosbag 循环播放或传感器重启时可能发生）
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);       // 格式解析 + 时间戳归一化 → ptr（curvature=ms 偏移）
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;  // 记录预处理耗时
    mtx_buffer.unlock();
    sig_buffer.notify_all();        // 通知主线程有新数据
}

double timediff_lidar_wrt_imu = 0.0;  // 软件时间同步：LiDAR 相对 IMU 的时间偏差（s）
bool   timediff_set_flg = false;       // 是否已完成一次自动时差估计

// LiDAR 回调（Livox 系列：CustomMsg 格式，含每点时间戳 offset_time）
// 额外处理软件时间同步（time_sync_en=true 时）：
//   检测 LiDAR 与 IMU 时间戳差值 > 1s → 估计时差 timediff_lidar_wrt_imu
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    // 检测未同步（硬件同步关闭时）：时差 > 10s 属于严重未同步，打印警告
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    // 软件时间同步（time_sync_en=true）：首次检测到 LiDAR-IMU 时差 > 1s 时记录偏差
    // 后续 imu_cbk 中用此偏差修正 IMU 时间戳，使两者对齐
    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);       // Livox CustomMsg 解析：offset_time/1e6 → curvature(ms)
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

// IMU 回调：时间戳修正（hardware offset 或软件同步偏差）后压入 imu_buffer
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count ++;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    // 固定时间偏移修正（yaml: common/time_offset_lidar_to_imu，单位 s）
    // 将 IMU 时间戳提前 time_diff_lidar_to_imu，使其与 LiDAR 时间轴对齐
    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);

    // 软件时间同步偏差修正（time_sync_en=true 时使用）
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();
    mtx_buffer.lock();

    // 时间戳回跳检测
    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;
    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;  // 在线估计的 LiDAR 平均扫描周期（s），用于异常帧端时间推算
int    scan_num = 0;               // 有效帧计数（用于在线均值计算）

// 时间同步打包函数：将 lidar_buffer 和 imu_buffer 中时间对齐的数据打包到 MeasureGroup
// 返回 true 表示已成功准备好一帧完整数据（IMU 覆盖整个 LiDAR 扫描时间段）
// 策略：等待 IMU 数据时间戳 >= lidar_end_time 后才打包（保证 IMU 覆盖完整帧）
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    // 将 lidar_buffer 队首帧推入 meas（lidar_pushed 标志防止重复推入）
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();

        // 推算 lidar_end_time：用最后一个点的时间偏移（curvature/1000 秒）
        // 异常情况（点数过少或时间戳异常）则用历史均值估计
        if (meas.lidar->points.size() <= 1)  // 点数不足，用均值
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            // 最后一点时间偏移异常偏小（< 半个周期），用均值
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            // 正常：end = beg + 最后点的时间偏移（ms→s）
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            // 在线更新平均扫描时间（Welford 均值）
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }
        if(lidar_type == MARSIM)
            lidar_end_time = meas.lidar_beg_time;  // MARSIM 无扫描时间，帧结束=帧开始

        meas.lidar_end_time = lidar_end_time;
        lidar_pushed = true;
    }

    // 等待 IMU 缓冲中有足够的数据（最新 IMU 时间 >= lidar_end_time）
    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    // 从 imu_buffer 中取出覆盖 [lidar_beg_time, lidar_end_time] 的所有 IMU 消息
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

// 增量地图更新：将本帧 iEKF 收敛后的下采样点云插入 ikd-Tree
// 插入策略：基于体素中心距离判断是否插入（避免冗余点，维持均匀地图密度）
//   - 若当前点所在体素内已有比它更靠近体素中心的点 → 跳过（need_add=false）
//   - 若当前点所在体素在地图中没有邻居 → 直接插入（PointNoNeedDownsample）
//   - 否则插入并由 ikd-Tree 内部做增量下采样（PointToAdd, downsample=true）
int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;              // 需要插入且允许 ikd-Tree 内部下采样的点
    PointVector PointNoNeedDownsample;   // 体素内无邻居，直接插入不下采样的点
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);

    for (int i = 0; i < feats_down_size; i++)
    {
        // 将点从 LiDAR 系变换到世界系（使用当前收敛状态 state_point）
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));

        // EKF 未初始化完成时，直接插入所有点（初始化地图）
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;

            // 计算该点所在体素的中心坐标（体素边长 = filter_size_map_min）
            PointType mid_point;
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i], mid_point);  // 当前点到体素中心的距离²

            // 若最近邻点不在同一体素（所有轴偏差均 > 半体素），说明该体素为空 → 直接插入
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min
             && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min
             && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            // 检查最近邻中是否有比当前点更靠近体素中心的点（已有更好代表）
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;  // 已有更接近体素中心的点，当前点冗余，不插入
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);  // 地图为空或未初始化，直接插入
        }
    }

    double st_time = omp_get_wtime();
    // 批量插入（downsample=true：允许 ikd-Tree 内部做增量下采样）
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    // 直接插入（downsample=false：体素无邻居，不做下采样）
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));  // 等待发布的点云缓冲
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());          // 等待存盘的点云累积缓冲

// 发布当前帧点云（世界坐标系）：topic /cloud_registered
// dense_pub_en=true → 发布去畸变完整帧（feats_undistort）
// dense_pub_en=false → 发布降采样后的点云（feats_down_body）
// 同时根据 pcd_save_en 将完整帧累积到 pcl_wait_save，
// 每 pcd_save_interval 帧写出一个 .pcd 文件到 ROOT_DIR/PCD/
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        // dense_pub_en: 发布完整点云(feats_undistort) vs 降采样点云(feats_down_body)
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        // 将 LiDAR 系点云转换到世界系（含 RGB 着色，用 intensity 代替颜色）
        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;  // 累积到存盘缓冲

        static int scan_wait_num = 0;
        scan_wait_num ++;
        // 达到存盘间隔 → 写出 PCD 文件（注意：写盘会阻塞主循环）
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

// 发布当前帧点云（IMU 机体坐标系）：topic /cloud_registered_body
// 用于与 IMU 对齐的外部感知任务（无需知道外参）
void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    // LiDAR 系 → IMU 系（外参 offset_R_L_I / offset_T_L_I）
    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

// 发布 iEKF 本次迭代实际参与更新的有效特征点（世界系）：topic /cloud_effected
// 即通过点面距离阈值检验的 effct_feat_num 个点，用于调试 H 矩阵质量
void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

// 发布 ikd-Tree 当前地图点（世界系）：topic /Laser_map
// 注意：featsFromMap 由 lasermap_fov_segment() 填充，为局部滑动窗口内的点
void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

// 将当前 iEKF 状态（位置+姿态四元数）写入 ROS pose 消息字段
template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
}

// 发布里程计：topic /Odometry（nav_msgs/Odometry）+ TF camera_init → body
// 协方差矩阵从 iEKF P 矩阵提取：
//   state_point 维度顺序：pos(0-2), rot(3-5), ...
//   ROS Odometry covariance 前 6×6 对应 [rot(3-5), pos(0-2)]（ROS 约定：rot 在前）
//   因此 k = i<3 ? i+3 : i-3 实现 pos/rot 维度对调
void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        // ROS covariance 顺序：[rx,ry,rz,px,py,pz]；P 顺序：[px,py,pz,rx,ry,rz]
        // 通过 k = i<3 ? i+3 : i-3 将 P 的 pos/rot 块对调填入 covariance
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    // 同步广播 TF 变换：camera_init → body（供 RViz 坐标系关联使用）
    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );
}

// 发布轨迹：topic /path（nav_msgs/Path）
// 每 10 帧追加一个 PoseStamped，避免 path 消息过大导致 RViz 崩溃
void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0)
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

// iEKF 测量模型回调函数（由 esekf::update_iterated_dyn_share_modified 每次迭代调用）
// 功能：
//   1. 将当前迭代状态 s 下的点云变换到世界系
//   2. 在 ikd-Tree 中搜索每点的 5 个最近邻，拟合局部平面
//   3. 计算每点的点面距离残差 r_j = n̂ᵀ·p_j^W + d
//   4. 筛选有效点（质量分数 s > 0.9）
//   5. 计算测量雅可比矩阵 H（effct_feat_num × 12）和残差向量 h（effct_feat_num）
//
// 输出写入 ekfom_data：
//   h_x：雅可比矩阵 H（∂残差/∂状态，仅前12列：pos/rot/offset_R/offset_T）
//   h：残差向量（负的点面距离）
//   valid：若无有效点则置 false，跳过本次更新
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear();   // 有效点（LiDAR 系坐标）
    corr_normvect->clear();   // 对应平面法向量（xyz=法向量，intensity=点面距离）
    total_residual = 0.0;

    // ─── 阶段1：最近邻搜索 + 平面拟合 + 残差计算（OpenMP 并行）───
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i];   // 去畸变后的点（LiDAR 系）
        PointType &point_world = feats_down_world->points[i];  // 世界系坐标（本次迭代更新）

        // 用当前迭代状态 s 将点变换到世界系：p_W = R_WI·(R_IL·p_L + t_IL) + p_WI
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
        auto &points_near = Nearest_Points[i];

        // converge=true 时（首次迭代或上次迭代后状态有变化）才重新搜索最近邻
        // converge=false 时复用上一轮的 points_near，节省搜索时间
        if (ekfom_data.converge)
        {
            // ikd-Tree K 近邻搜索（k=5），返回最近邻点和距离平方
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            // 有效性检查：邻居数量不足 或 最远邻居距离² > 5m²（约 2.24m）则无效
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        // 用 5 个最近邻拟合平面，pabcd = [nx, ny, nz, d]（esti_plane 来自 common_lib.h）
        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))  // 平面拟合阈值 0.1m
        {
            // 点面距离：r_j = n̂ᵀ·p_j^W + d
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            // 质量分数：s = 1 - 0.9·|r_j|/||p_j^L||
            // 含义：点面距离相对点到原点距离的比值越小越好，s > 0.9 表示比值 < ~11%
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);          // 平面法向量 x
                normvec->points[i].y = pabcd(1);          // 平面法向量 y
                normvec->points[i].z = pabcd(2);          // 平面法向量 z
                normvec->points[i].intensity = pd2;        // 复用 intensity 存点面距离 r_j
                res_last[i] = abs(pd2);
            }
        }
    }

    // ─── 阶段2：收集有效点 ───
    effct_feat_num = 0;
    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];  // 有效点（LiDAR 系）
            corr_normvect->points[effct_feat_num] = normvec->points[i];          // 对应法向量+距离
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;  // 无有效点，本次 iEKF 迭代跳过
        ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;  // 平均点面距离（监控配准质量）
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();

    // ─── 阶段3：计算测量雅可比矩阵 H 和残差向量 h ───
    // H 的列索引对应状态切空间的前 12 维：[pos(3), rot(3), offset_R(3), offset_T(3)]
    // 速度/偏置/重力对点面距离无直接影响（h_x 第12列以后为0）
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);  // p_j^L（LiDAR 系）
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);   // [p_j^L]×（反对称矩阵）
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;  // p_j^I（IMU 系）
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);  // [p_j^I]×（IMU 系反对称矩阵）

        // 对应平面的法向量（世界系单位向量）
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);  // n̂_j（世界系）

        // 计算 H 矩阵各列（测量雅可比）
        // C = R_WI^T · n̂_j   （法向量转到 IMU 系）
        // 用于简化后续对 rot/offset 的偏导计算
        V3D C(s.rot.conjugate() * norm_vec);
        // A = [p_j^I]× · C = ∂h/∂rot_tangent
        // 推导：∂(n̂^T·R_WI·p^I)/∂δφ = n̂^T·R_WI·[p^I]×·(-δφ) → ∂h/∂δφ = A^T
        V3D A(point_crossmat * C);

        if (extrinsic_est_en)
        {
            // B = [p_j^L]× · R_IL^T · C = ∂h/∂offset_R_tangent（外参旋转的偏导）
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);
            // H 行：[∂h/∂p(3), ∂h/∂rot(3), ∂h/∂offset_R(3), ∂h/∂offset_T(3)]
            //       = [n̂^T,      A^T,         B^T,              C^T           ]
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            // 外参固定时，offset_R/offset_T 列置零（不参与更新）
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        // 测量残差：z̃_j = -r_j = -(n̂^T·p_j^W + d)
        // norm_p.intensity 中存储的就是 r_j（正值 = 点在平面外侧）
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);
    nh.param<string>("map_file_path",map_file_path,"");
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",filter_size_map_min,0.5);
    nh.param<double>("cube_side_length",cube_len,200);
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<double>("mapping/fov_degree",fov_deg,180);
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());

    p_pre->lidar_type = lidar_type;
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    p_imu->lidar_type = lidar_type;
    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/Odometry", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 100000);
//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);  // 主循环以 5000Hz 轮询，实际处理频率由传感器决定
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit) break;
        ros::spinOnce();  // 处理 ROS 回调（填充 lidar_buffer / imu_buffer）

        // sync_packages：检查 lidar_buffer 和 imu_buffer 是否有完整的一帧数据
        // 条件：imu 数据时间覆盖了整个 LiDAR 帧 [lidar_beg_time, lidar_end_time]
        if(sync_packages(Measures))
        {
            // 跳过第一帧（仅记录起始时间，不处理点云）
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;
            match_time = 0; kdtree_search_time = 0.0; solve_time = 0;
            solve_const_H_time = 0; svd_time = 0;
            t0 = omp_get_wtime();

            // ── 步骤1：IMU 处理 ──
            // 包含：初始化阶段 → IMU_init；或正常阶段 → 前向传播 + 后向去畸变
            // 输出：feats_undistort（去畸变点云，LiDAR 系帧末坐标）
            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();  // 前向传播后的预测状态（帧末）
            // LiDAR 在世界系的位置（用于地图管理 FOV 检查）
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            // iEKF 初始化标志：前 INIT_TIME=0.1s 内不做地图匹配，仅初始化
            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

            // ── 步骤2：滑动窗口地图裁剪 ──
            // 检查当前位置是否接近地图边界，必要时平移地图窗口并删除旧点
            lasermap_fov_segment();

            // ── 步骤3：体素下采样 ──
            // 将去畸变点云下采样（filter_size_surf_min），减少 iEKF 计算量
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);  // 输出：feats_down_body（LiDAR 系）
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();

            // ── 步骤4：首帧建立 ikd-Tree ──
            if(ikdtree.Root_Node == nullptr)
            {
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    ikdtree.Build(feats_down_world->points);  // 首次建树
                }
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();

            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            normvec->resize(feats_down_size);        // 存放各点对应平面法向量
            feats_down_world->resize(feats_down_size); // 存放各点世界系坐标（迭代更新）

            // 记录更新前状态（调试日志）
            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // 调试用：将地图中所有点提取到 featsFromMap（改为 if(1) 可查看地图点云）
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);  // 每点对应的 KNN 搜索结果（跨迭代复用）

            t2 = omp_get_wtime();

            // ── 步骤5：迭代 iEKF 更新（核心：scan-to-map 配准 + 状态估计）──
            // 内部循环调用 h_share_model 计算 H 和 h，然后做 Kalman 更新
            // LASER_POINT_COV：LiDAR 点测量噪声方差，控制更新步长
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);

            // 取收敛后的最优状态
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);  // 欧拉角（仅日志用）
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;  // LiDAR 世界系位置
            // 将旋转四元数转为 geometry_msgs/Quaternion（用于发布 odometry/tf）
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            // 发布里程计（位姿 + 协方差），同时广播 TF（camera_init → body）
            publish_odometry(pubOdomAftMapped);

            // ── 步骤6：增量地图更新 ──
            // 将本帧有效点插入 ikd-Tree（基于体素中心距离判断）
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }

        status = ros::ok();
        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
