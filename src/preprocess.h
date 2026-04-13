#ifndef PREPROCESS_H
#define PREPROCESS_H

// LiDAR 点云预处理模块
// 将各型 LiDAR 原始消息转换为统一格式 PointCloudXYZI：
//   - 过滤盲区点（< blind 米）
//   - 抽帧（每隔 point_filter_num 取一点）
//   - 时间戳归一化：统一存入 curvature 字段（ms，相对帧首偏移）
//   - 可选特征提取（平面点/边缘点分类）

#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver/CustomMsg.h>

using namespace std;

#define IS_VALID(a)  ((abs(a)>1e8) ? true : false)  // 数值溢出检查

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

// LiDAR 类型（对应 yaml 中 preprocess/lidar_type）
enum LID_TYPE{AVIA = 1, VELO16, OUST64, MARSIM}; // 1=Livox, 2=Velodyne, 3=Ouster, 4=MARSIM
// 时间戳单位（对应 yaml 中 preprocess/timestamp_unit），最终统一转换为 ms
enum TIME_UNIT{SEC = 0, MS = 1, US = 2, NS = 3};
// 特征类型（仅 feature_enabled=true 时有效）
enum Feature{Nor, Poss_Plane, Real_Plane, Edge_Jump, Edge_Plane, Wire, ZeroPoint};
enum Surround{Prev, Next};     // 相邻方向（前/后）
enum E_jump{Nr_nor, Nr_zero, Nr_180, Nr_inf, Nr_blind};  // 相邻点深度跳变类型

// 特征提取辅助结构：存储点的几何属性，用于平面/边缘判断
struct orgtype
{
  double range;       // 水平距离 √(x²+y²)（用于盲区检查）
  double dista;       // 与下一点的欧氏距离（用于边缘跳变判断）
  double angle[2];    // 与相邻点的夹角（特征分类辅助）
  double intersect;   // 与上一平面段的交叉角（连续性判断）
  E_jump edj[2];      // [Prev, Next] 方向的跳变类型
  Feature ftype;      // 最终分类结果
  orgtype()
  {
    range = 0;
    edj[Prev] = Nr_nor;
    edj[Next] = Nr_nor;
    ftype = Nor;
    intersect = 2;
  }
};

namespace velodyne_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      float time;
      uint16_t ring;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, time, time)
    (uint16_t, ring, ring)
)

namespace ouster_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      uint32_t t;
      uint16_t reflectivity;
      uint8_t  ring;
      uint16_t ambient;
      uint32_t range;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    // use std::uint32_t to avoid conflicting with pcl::uint32_t
    (std::uint32_t, t, t)
    (std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring)
    (std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range)
)

class Preprocess
{
  public:
  Preprocess();
  ~Preprocess();

  // 处理 Livox CustomMsg（Avia/Horizon/Mid-70 等固态 LiDAR 专用消息格式）
  void process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
  // 处理标准 PointCloud2（Velodyne / Ouster / MARSIM）
  void process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
  void set(bool feat_en, int lid_type, double bld, int pfilt_num);

  // 中间/输出点云缓冲（公开以便调试发布）
  PointCloudXYZI pl_full;            // 当前帧所有有效点（含无特征点）
  PointCloudXYZI pl_corn;            // 边缘特征点（feature_enabled=true 时填充）
  PointCloudXYZI pl_surf;            // 平面特征点 / 直接输出（feature_enabled=false 时为全部点）
  PointCloudXYZI pl_buff[128];       // 按线号分组的点云缓冲（最多 128 线）
  vector<orgtype> typess[128];       // 对应每线的点属性数组（特征提取中间结果）

  // 时间单位换算系数：time_field × time_unit_scale = 毫秒
  float time_unit_scale;

  // 配置参数（从 yaml/laserMapping.cpp 读取后设置）
  int lidar_type;          // LiDAR 类型（LID_TYPE 枚举）
  int point_filter_num;    // 抽帧间隔（每 N 个点取 1 个，减少计算量）
  int N_SCANS;             // 激光线数（Velodyne 16/32/64，Ouster 64）
  int SCAN_RATE;           // 扫描频率（Hz），用于 Velodyne 无时间戳时的角度推算
  int time_unit;           // 时间字段单位（TIME_UNIT 枚举）
  double blind;            // 盲区半径（m），近距离点（噪声大）过滤阈值
  bool feature_enabled;    // 是否启用特征提取（默认 false，FAST-LIO2 直接用所有点）
  bool given_offset_time;  // Velodyne 消息是否包含每点时间戳（否则用角度推算）
  ros::Publisher pub_full, pub_surf, pub_corn;  // 调试发布器（未在 laserMapping 中注册，暂未使用）

  private:
  // 各型 LiDAR 解析函数：读取原始消息，填充 pl_surf
  void avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg);   // Livox 固态 LiDAR
  void oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);    // Ouster 64 线
  void velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);  // Velodyne 系列
  void sim_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);       // MARSIM 仿真

  // 特征提取（feature_enabled=true 时调用）：对单条扫描线按几何特征分类
  void give_feature(PointCloudXYZI &pl, vector<orgtype> &types);
  void pub_func(PointCloudXYZI &pl, const ros::Time &ct);
  // 平面判断：检查从点 i 开始的局部段是否为平面（返回 0/1/2）
  int  plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, uint &i_nex, Eigen::Vector3d &curr_direct);
  bool small_plane(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct);
  // 边缘跳变判断（深度突变 → 遮挡边缘）
  bool edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir);

  // 特征提取参数（在构造函数中初始化，均已转换为 cos 值）
  int group_size;              // 平面拟合窗口点数
  double disA, disB;           // 点间距离阈值
  double inf_bound;            // 无穷远点判断阈值
  double limit_maxmid;         // 点间距比值上限（平面连续性）
  double limit_midmin;
  double limit_maxmin;
  double p2l_ratio;            // 点到线距离比值（特征判断）
  double jump_up_limit;        // 跳变上限角余弦（对应 170°）
  double jump_down_limit;      // 跳变下限角余弦（对应 8°）
  double cos160;               // 160° 的余弦（共面性判断）
  double edgea, edgeb;         // 边缘判断参数
  double smallp_intersect;     // 小平面交叉角余弦（172.5°）
  double smallp_ratio;         // 小平面距离比值
  double vx, vy, vz;           // 临时向量差（复用避免重复分配）
};
#endif