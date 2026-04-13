#include "preprocess.h"

#define RETURN0     0x00
#define RETURN0AND1 0x10

Preprocess::Preprocess()
  :feature_enabled(0), lidar_type(AVIA), blind(0.01), point_filter_num(1)
{
  inf_bound = 10;
  N_SCANS   = 6;
  SCAN_RATE = 10;
  group_size = 8;
  disA = 0.01;
  disA = 0.1; // B?
  p2l_ratio = 225;
  limit_maxmid =6.25;
  limit_midmin =6.25;
  limit_maxmin = 3.24;
  jump_up_limit = 170.0;
  jump_down_limit = 8.0;
  cos160 = 160.0;
  edgea = 2;
  edgeb = 0.1;
  smallp_intersect = 172.5;
  smallp_ratio = 1.2;
  given_offset_time = false;

  jump_up_limit = cos(jump_up_limit/180*M_PI);
  jump_down_limit = cos(jump_down_limit/180*M_PI);
  cos160 = cos(cos160/180*M_PI);
  smallp_intersect = cos(smallp_intersect/180*M_PI);
}

Preprocess::~Preprocess() {}

void Preprocess::set(bool feat_en, int lid_type, double bld, int pfilt_num)
{
  feature_enabled = feat_en;
  lidar_type = lid_type;
  blind = bld;
  point_filter_num = pfilt_num;
}

void Preprocess::process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{  
  avia_handler(msg);
  *pcl_out = pl_surf;
}

void Preprocess::process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
  switch (time_unit)
  {
    case SEC:
      time_unit_scale = 1.e3f;
      break;
    case MS:
      time_unit_scale = 1.f;
      break;
    case US:
      time_unit_scale = 1.e-3f;
      break;
    case NS:
      time_unit_scale = 1.e-6f;
      break;
    default:
      time_unit_scale = 1.f;
      break;
  }

  switch (lidar_type)
  {
  case OUST64:
    oust64_handler(msg);
    break;

  case VELO16:
    velodyne_handler(msg);
    break;

  case MARSIM:
    sim_handler(msg);
    break;
  
  default:
    printf("Error LiDAR Type");
    break;
  }
  *pcl_out = pl_surf;
}

// ═══════════════════════════════════════════════════════
// Livox Avia / Horizon / Mid-70 等固态 LiDAR 解析
// 消息格式：livox_ros_driver/CustomMsg，含每点时间偏移 offset_time（ns）
// 时间戳处理：offset_time / 1e6 → curvature（ms，相对帧首偏移）
// 有效点过滤：
//   - tag & 0x30 == 0x10 → 单回波（Single Return），可靠
//   - tag & 0x30 == 0x00 → 强回波（Strong Return），最可靠
//   - 其他 tag 值（0x20 弱回波、0x30 杂散光）舍弃
// ═══════════════════════════════════════════════════════
void Preprocess::avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();
  double t1 = omp_get_wtime();
  int plsize = msg->point_num;

  pl_corn.reserve(plsize);
  pl_surf.reserve(plsize);
  pl_full.resize(plsize);

  for(int i=0; i<N_SCANS; i++)
  {
    pl_buff[i].clear();
    pl_buff[i].reserve(plsize);
  }
  uint valid_num = 0;

  if (feature_enabled)  // 特征提取模式：按线号分组后做平面/边缘分类
  {
    for(uint i=1; i<plsize; i++)
    {
      // 过滤条件：线号在有效范围内，且 tag 表示单回波或强回波
      if((msg->points[i].line < N_SCANS) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
      {
        pl_full[i].x = msg->points[i].x;
        pl_full[i].y = msg->points[i].y;
        pl_full[i].z = msg->points[i].z;
        pl_full[i].intensity = msg->points[i].reflectivity;
        // curvature 字段复用为时间偏移（ms）：offset_time(ns) / 1e6 → ms
        pl_full[i].curvature = msg->points[i].offset_time / float(1000000); //use curvature as time of each laser points

        bool is_new = false;
        if((abs(pl_full[i].x - pl_full[i-1].x) > 1e-7) 
            || (abs(pl_full[i].y - pl_full[i-1].y) > 1e-7)
            || (abs(pl_full[i].z - pl_full[i-1].z) > 1e-7))
        {
          pl_buff[msg->points[i].line].push_back(pl_full[i]);
        }
      }
    }
    static int count = 0;
    static double time = 0.0;
    count ++;
    double t0 = omp_get_wtime();
    for(int j=0; j<N_SCANS; j++)
    {
      if(pl_buff[j].size() <= 5) continue;
      pcl::PointCloud<PointType> &pl = pl_buff[j];
      plsize = pl.size();
      vector<orgtype> &types = typess[j];
      types.clear();
      types.resize(plsize);
      plsize--;
      for(uint i=0; i<plsize; i++)
      {
        types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
        vx = pl[i].x - pl[i + 1].x;
        vy = pl[i].y - pl[i + 1].y;
        vz = pl[i].z - pl[i + 1].z;
        types[i].dista = sqrt(vx * vx + vy * vy + vz * vz);
      }
      types[plsize].range = sqrt(pl[plsize].x * pl[plsize].x + pl[plsize].y * pl[plsize].y);
      give_feature(pl, types);
      // pl_surf += pl;
    }
    time += omp_get_wtime() - t0;
    printf("Feature extraction time: %lf \n", time / count);
  }
  else  // 直接模式（FAST-LIO2 默认）：不做特征分类，所有有效点进 pl_surf
  {
    for(uint i=1; i<plsize; i++)
    {
      if((msg->points[i].line < N_SCANS) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
      {
        valid_num ++;
        // 抽帧：每 point_filter_num 个有效点取 1 个（降低点密度，减少后续计算量）
        if (valid_num % point_filter_num == 0)
        {
          pl_full[i].x = msg->points[i].x;
          pl_full[i].y = msg->points[i].y;
          pl_full[i].z = msg->points[i].z;
          pl_full[i].intensity = msg->points[i].reflectivity;
          // curvature = 时间偏移(ms)，供 IMU 去畸变使用
          pl_full[i].curvature = msg->points[i].offset_time / float(1000000); // curvature unit: ms

          // 去重（与前一点坐标完全相同则跳过） + 盲区过滤（距离 < blind）
          if(((abs(pl_full[i].x - pl_full[i-1].x) > 1e-7)
              || (abs(pl_full[i].y - pl_full[i-1].y) > 1e-7)
              || (abs(pl_full[i].z - pl_full[i-1].z) > 1e-7))
              && (pl_full[i].x * pl_full[i].x + pl_full[i].y * pl_full[i].y + pl_full[i].z * pl_full[i].z > (blind * blind)))
          {
            pl_surf.push_back(pl_full[i]);
          }
        }
      }
    }
  }
}

void Preprocess::oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();
  pcl::PointCloud<ouster_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.size();
  pl_corn.reserve(plsize);
  pl_surf.reserve(plsize);
  if (feature_enabled)
  {
    for (int i = 0; i < N_SCANS; i++)
    {
      pl_buff[i].clear();
      pl_buff[i].reserve(plsize);
    }

    for (uint i = 0; i < plsize; i++)
    {
      double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y + pl_orig.points[i].z * pl_orig.points[i].z;
      if (range < (blind * blind)) continue;
      Eigen::Vector3d pt_vec;
      PointType added_pt;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.3;
      if (yaw_angle >= 180.0)
        yaw_angle -= 360.0;
      if (yaw_angle <= -180.0)
        yaw_angle += 360.0;

      added_pt.curvature = pl_orig.points[i].t * time_unit_scale;
      if(pl_orig.points[i].ring < N_SCANS)
      {
        pl_buff[pl_orig.points[i].ring].push_back(added_pt);
      }
    }

    for (int j = 0; j < N_SCANS; j++)
    {
      PointCloudXYZI &pl = pl_buff[j];
      int linesize = pl.size();
      vector<orgtype> &types = typess[j];
      types.clear();
      types.resize(linesize);
      linesize--;
      for (uint i = 0; i < linesize; i++)
      {
        types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
        vx = pl[i].x - pl[i + 1].x;
        vy = pl[i].y - pl[i + 1].y;
        vz = pl[i].z - pl[i + 1].z;
        types[i].dista = vx * vx + vy * vy + vz * vz;
      }
      types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x + pl[linesize].y * pl[linesize].y);
      give_feature(pl, types);
    }
  }
  else
  {
    double time_stamp = msg->header.stamp.toSec();
    // cout << "===================================" << endl;
    // printf("Pt size = %d, N_SCANS = %d\r\n", plsize, N_SCANS);
    for (int i = 0; i < pl_orig.points.size(); i++)
    {
      if (i % point_filter_num != 0) continue;

      double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y + pl_orig.points[i].z * pl_orig.points[i].z;
      
      if (range < (blind * blind)) continue;
      
      Eigen::Vector3d pt_vec;
      PointType added_pt;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      added_pt.curvature = pl_orig.points[i].t * time_unit_scale; // curvature unit: ms

      pl_surf.points.push_back(added_pt);
    }
  }
  // pub_func(pl_surf, pub_full, msg->header.stamp);
  // pub_func(pl_surf, pub_corn, msg->header.stamp);
}

// ═══════════════════════════════════════════════════════
// Velodyne 系列旋转 LiDAR 解析（16/32/64 线）
// 消息格式：PointCloud2 + velodyne_ros::Point（含 ring / time 字段）
// 时间戳处理（两种情况）：
//   - 若 time 字段有效（> 0）：直接用 time * time_unit_scale → curvature(ms)
//   - 若无时间戳：根据偏航角变化和旋转角速度推算偏移时间（yaw-based）
//     curvature = (yaw_fp - yaw_angle) / omega_l（ms）
//     omega_l = 0.361° * SCAN_RATE（°/ms）
// ═══════════════════════════════════════════════════════
void Preprocess::velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();

    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    if (plsize == 0) return;
    pl_surf.reserve(plsize);

    // 无时间戳时用偏航角推算所需的变量（仅当 given_offset_time=false 时有效）
    double omega_l = 0.361 * SCAN_RATE;           // 角速度（°/ms）= 0.361°/点 * SCAN_RATE(Hz)
    std::vector<bool>   is_first(N_SCANS, true);  // 每线是否是第一个点
    std::vector<double> yaw_fp(N_SCANS, 0.0);     // 每线第一个点的偏航角（°）
    std::vector<float>  yaw_last(N_SCANS, 0.0);   // 每线上一点的偏航角
    std::vector<float>  time_last(N_SCANS, 0.0);  // 每线上一点的时间偏移（ms）

    // 检测是否有每点时间戳：用最后一个点的 time 字段判断
    if (pl_orig.points[plsize - 1].time > 0)
    {
      given_offset_time = true;   // 有时间戳，直接用
    }
    else
    {
      given_offset_time = false;  // 无时间戳，用角度推算
      // 找第一条线的首尾偏航角（用于后续角度回绕检测，已弃用但保留）
      double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
      double yaw_end   = yaw_first;
      int layer_first  = pl_orig.points[0].ring;
      for (uint i = plsize - 1; i > 0; i--)
      {
        if (pl_orig.points[i].ring == layer_first)
        {
          yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
          break;
        }
      }
    }

    if(feature_enabled)
    {
      for (int i = 0; i < N_SCANS; i++)
      {
        pl_buff[i].clear();
        pl_buff[i].reserve(plsize);
      }
      
      for (int i = 0; i < plsize; i++)
      {
        PointType added_pt;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        int layer  = pl_orig.points[i].ring;
        if (layer >= N_SCANS) continue;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time * time_unit_scale; // units: ms

        if (!given_offset_time)
        {
          double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;
          if (is_first[layer])
          {
            // printf("layer: %d; is first: %d", layer, is_first[layer]);
              yaw_fp[layer]=yaw_angle;
              is_first[layer]=false;
              added_pt.curvature = 0.0;
              yaw_last[layer]=yaw_angle;
              time_last[layer]=added_pt.curvature;
              continue;
          }

          if (yaw_angle <= yaw_fp[layer])
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle) / omega_l;
          }
          else
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle+360.0) / omega_l;
          }

          if (added_pt.curvature < time_last[layer])  added_pt.curvature+=360.0/omega_l;

          yaw_last[layer] = yaw_angle;
          time_last[layer]=added_pt.curvature;
        }

        pl_buff[layer].points.push_back(added_pt);
      }

      for (int j = 0; j < N_SCANS; j++)
      {
        PointCloudXYZI &pl = pl_buff[j];
        int linesize = pl.size();
        if (linesize < 2) continue;
        vector<orgtype> &types = typess[j];
        types.clear();
        types.resize(linesize);
        linesize--;
        for (uint i = 0; i < linesize; i++)
        {
          types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
          vx = pl[i].x - pl[i + 1].x;
          vy = pl[i].y - pl[i + 1].y;
          vz = pl[i].z - pl[i + 1].z;
          types[i].dista = vx * vx + vy * vy + vz * vz;
        }
        types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x + pl[linesize].y * pl[linesize].y);
        give_feature(pl, types);
      }
    }
    else
    {
      for (int i = 0; i < plsize; i++)
      {
        PointType added_pt;
        // cout<<"!!!!!!"<<i<<" "<<plsize<<endl;
        
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time * time_unit_scale;  // curvature unit: ms // cout<<added_pt.curvature<<endl;

        if (!given_offset_time)
        {
          int layer = pl_orig.points[i].ring;
          double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

          if (is_first[layer])
          {
            // printf("layer: %d; is first: %d", layer, is_first[layer]);
              yaw_fp[layer]=yaw_angle;
              is_first[layer]=false;
              added_pt.curvature = 0.0;
              yaw_last[layer]=yaw_angle;
              time_last[layer]=added_pt.curvature;
              continue;
          }

          // compute offset time
          if (yaw_angle <= yaw_fp[layer])
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle) / omega_l;
          }
          else
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle+360.0) / omega_l;
          }

          if (added_pt.curvature < time_last[layer])  added_pt.curvature+=360.0/omega_l;

          yaw_last[layer] = yaw_angle;
          time_last[layer]=added_pt.curvature;
        }

        if (i % point_filter_num == 0)
        {
          if(added_pt.x*added_pt.x+added_pt.y*added_pt.y+added_pt.z*added_pt.z > (blind * blind))
          {
            pl_surf.points.push_back(added_pt);
          }
        }
      }
    }
}

// ═══════════════════════════════════════════════════════
// MARSIM 仿真器点云解析
// 消息格式：标准 PointCloud2 + pcl::PointXYZI（无 ring / time 字段）
// 特点：仿真点云无运动畸变，curvature 直接置 0
//   → IMU_Processing.hpp 中 UndistortPcl 对 MARSIM 跳过后向补偿
// ═══════════════════════════════════════════════════════
void Preprocess::sim_handler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    pl_surf.clear();
    pl_full.clear();
    pcl::PointCloud<pcl::PointXYZI> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.size();
    pl_surf.reserve(plsize);
    for (int i = 0; i < (int)pl_orig.points.size(); i++) {
        // 盲区过滤（距离² < blind²）
        double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                       pl_orig.points[i].z * pl_orig.points[i].z;
        if (range < blind * blind) continue;
        PointType added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.curvature = 0.0;  // 仿真点云无时间偏移（无运动畸变）
        pl_surf.points.push_back(added_pt);
    }
}

// ═══════════════════════════════════════════════════════
// 特征提取：对单条扫描线的点序列，按几何特征分类为：
//   Real_Plane  — 可靠平面点（位于连续平面段内部）
//   Poss_Plane  — 候选平面点（平面段端点，可靠性稍低）
//   Edge_Jump   — 深度跳变边缘（遮挡/反射突变）
//   Edge_Plane  — 两平面段交界处边缘
//   Wire        — 孤立点（两侧均为特殊跳变）
//
// 分三轮扫描：
//   第一轮（平面段识别）：逐点调用 plane_judge()，将连续平面段标记为 Poss/Real_Plane；
//     相邻两平面段方向夹角 > 45°（|cos|<0.707）则交界点设为 Edge_Plane
//   第二轮（边缘跳变识别）：对未分类点，检查前后点距离突变（比值 > 4×）
//     并通过 edge_jump_judge() 确认是否为遮挡边缘 Edge_Jump
//   第三轮（小平面补充）：三点夹角 < smallp_intersect(172.5°) 且距离比 < smallp_ratio(1.2)
//     的局部区域标记为 Real_Plane（短平面段补救）
//
// 最终输出：
//   pl_surf ← 平面点（Poss/Real_Plane，按 point_filter_num 抽帧或取平均）
//   pl_corn ← 边缘点（Edge_Jump / Edge_Plane）
// ═══════════════════════════════════════════════════════
void Preprocess::give_feature(pcl::PointCloud<PointType> &pl, vector<orgtype> &types)
{
  int plsize = pl.size();
  int plsize2;
  if(plsize == 0)
  {
    printf("something wrong\n");
    return;
  }
  uint head = 0;

  // 跳过盲区内的点（从第一个有效距离点开始）
  while(types[head].range < blind)
  {
    head++;
  }

  // ── 第一轮：平面段识别 ──────────────────────────────
  // 最多扫描到倒数第 group_size 个点（为 plane_judge 预留窗口）
  plsize2 = (plsize > group_size) ? (plsize - group_size) : 0;

  Eigen::Vector3d curr_direct(Eigen::Vector3d::Zero());  // 当前平面段法向量
  Eigen::Vector3d last_direct(Eigen::Vector3d::Zero());  // 上一平面段法向量

  uint i_nex = 0, i2;
  uint last_i = 0; uint last_i_nex = 0;
  int last_state = 0;   // 上一段是否为平面（1=是）
  int plane_type;

  for(uint i=head; i<plsize2; i++)
  {
    if(types[i].range < blind)
    {
      continue;
    }

    i2 = i;

    // plane_judge 返回：
    //   1 → 当前窗口是平面（curr_direct 为主方向）
    //   0 → 非平面（点间距方差大）
    //   2 → 窗口内有盲区点（无效）
    plane_type = plane_judge(pl, types, i, i_nex, curr_direct);

    if(plane_type == 1)
    {
      // 平面段内部 → Real_Plane，两端点 → Poss_Plane（可靠性略低）
      for(uint j=i; j<=i_nex; j++)
      {
        if(j!=i && j!=i_nex)
        {
          types[j].ftype = Real_Plane;
        }
        else
        {
          types[j].ftype = Poss_Plane;
        }
      }

      // 若上一段也是平面，检测两段法向量夹角
      // |cos(θ)| < 0.707 → θ ∈ (45°, 135°) → 两平面不共面 → 交界点为 Edge_Plane
      if(last_state==1 && last_direct.norm()>0.1)
      {
        double mod = last_direct.transpose() * curr_direct;
        if(mod>-0.707 && mod<0.707)
        {
          types[i].ftype = Edge_Plane;
        }
        else
        {
          types[i].ftype = Real_Plane;
        }
      }

      i = i_nex - 1;  // 跳到平面段末尾（下一循环 i++ 后从 i_nex 开始）
      last_state = 1;
    }
    else // plane_type == 0 或 2：非平面或含盲区
    {
      i = i_nex;   // 跳过整个窗口
      last_state = 0;
    }
    // else if(plane_type == 0)
    // {
    //   if(last_state == 1)
    //   {
    //     uint i_nex_tem;
    //     uint j;
    //     for(j=last_i+1; j<=last_i_nex; j++)
    //     {
    //       uint i_nex_tem2 = i_nex_tem;
    //       Eigen::Vector3d curr_direct2;

    //       uint ttem = plane_judge(pl, types, j, i_nex_tem, curr_direct2);

    //       if(ttem != 1)
    //       {
    //         i_nex_tem = i_nex_tem2;
    //         break;
    //       }
    //       curr_direct = curr_direct2;
    //     }

    //     if(j == last_i+1)
    //     {
    //       last_state = 0;
    //     }
    //     else
    //     {
    //       for(uint k=last_i_nex; k<=i_nex_tem; k++)
    //       {
    //         if(k != i_nex_tem)
    //         {
    //           types[k].ftype = Real_Plane;
    //         }
    //         else
    //         {
    //           types[k].ftype = Poss_Plane;
    //         }
    //       }
    //       i = i_nex_tem-1;
    //       i_nex = i_nex_tem;
    //       i2 = j-1;
    //       last_state = 1;
    //     }

    //   }
    // }

    last_i = i2;
    last_i_nex = i_nex;
    last_direct = curr_direct;
  }

  // ── 第二轮：边缘跳变识别 ───────────────────────────
  // 对未分类点（ftype < Real_Plane），检测深度方向突变（遮挡边缘）
  // 需要至少 3 个前置点 / 3 个后置点用于 edge_jump_judge()
  plsize2 = plsize > 3 ? plsize - 3 : 0;
  for(uint i=head+3; i<plsize2; i++)
  {
    // 跳过盲区点和已分类的平面点
    if(types[i].range<blind || types[i].ftype>=Real_Plane)
    {
      continue;
    }

    // 跳过零距离点（重叠点，无法计算角度）
    if(types[i-1].dista<1e-16 || types[i].dista<1e-16)
    {
      continue;
    }

    Eigen::Vector3d vec_a(pl[i].x, pl[i].y, pl[i].z);  // 当前点向量（从原点出发）
    Eigen::Vector3d vecs[2];  // vecs[0]=Prev方向向量, vecs[1]=Next方向向量

    // 对前后两个方向分别判断跳变类型
    for(int j=0; j<2; j++)
    {
      int m = -1;  // j=0: Prev(前一点), j=1: Next(后一点)
      if(j == 1)
      {
        m = 1;
      }

      if(types[i+m].range < blind)
      {
        // 邻点在盲区内 → 当前点是否在无穷远处？
        if(types[i].range > inf_bound)
        {
          types[i].edj[j] = Nr_inf;   // 当前点极远（遮挡前景）
        }
        else
        {
          types[i].edj[j] = Nr_blind; // 邻点在盲区（传感器噪声）
        }
        continue;
      }

      // 计算当前点到邻点的方向向量（相对于当前点）
      vecs[j] = Eigen::Vector3d(pl[i+m].x, pl[i+m].y, pl[i+m].z);
      vecs[j] = vecs[j] - vec_a;

      // angle = cos(当前点方向 与 到邻点方向 的夹角)
      // jump_up_limit   = cos(170°) ≈ -0.985 → 近乎反向 → Nr_180（遮挡边缘对侧）
      // jump_down_limit = cos(8°)   ≈  0.990 → 近乎同向 → Nr_zero（深度跳变前景侧）
      types[i].angle[j] = vec_a.dot(vecs[j]) / vec_a.norm() / vecs[j].norm();
      if(types[i].angle[j] < jump_up_limit)
      {
        types[i].edj[j] = Nr_180;
      }
      else if(types[i].angle[j] > jump_down_limit)
      {
        types[i].edj[j] = Nr_zero;
      }
    }

    // 前后向量夹角（共面性）：intersect = cos(前向量, 后向量)
    // 越接近 1（180°）表示点在中间，两侧方向相反（正常连续性）
    types[i].intersect = vecs[Prev].dot(vecs[Next]) / vecs[Prev].norm() / vecs[Next].norm();

    // ── 跳变判断逻辑（四种情况）：
    // 情况1：后侧 Nr_zero（近点）, 前侧 Nr_nor，且当前距离 > 4× 前距离
    //        → 当前点是遮挡前景边缘，用 edge_jump_judge 向 Prev 方向验证
    if(types[i].edj[Prev]==Nr_nor && types[i].edj[Next]==Nr_zero && types[i].dista>0.0225 && types[i].dista>4*types[i-1].dista)
    {
      if(types[i].intersect > cos160)  // 前后方向夹角 > 160°（接近共线，排除弯曲）
      {
        if(edge_jump_judge(pl, types, i, Prev))
        {
          types[i].ftype = Edge_Jump;
        }
      }
    }
    // 情况2：前侧 Nr_zero, 后侧 Nr_nor，且前距离 > 4× 当前距离 → 对称情况
    else if(types[i].edj[Prev]==Nr_zero && types[i].edj[Next]== Nr_nor && types[i-1].dista>0.0225 && types[i-1].dista>4*types[i].dista)
    {
      if(types[i].intersect > cos160)
      {
        if(edge_jump_judge(pl, types, i, Next))
        {
          types[i].ftype = Edge_Jump;
        }
      }
    }
    // 情况3：前侧正常, 后侧无穷远 → 当前点在遮挡背景边缘
    else if(types[i].edj[Prev]==Nr_nor && types[i].edj[Next]==Nr_inf)
    {
      if(edge_jump_judge(pl, types, i, Prev))
      {
        types[i].ftype = Edge_Jump;
      }
    }
    // 情况4：前侧无穷远, 后侧正常 → 对称情况
    else if(types[i].edj[Prev]==Nr_inf && types[i].edj[Next]==Nr_nor)
    {
      if(edge_jump_judge(pl, types, i, Next))
      {
        types[i].ftype = Edge_Jump;
      }
    }
    // 情况5：两侧均为特殊跳变（Nr_180/Nr_zero/Nr_blind/Nr_inf）→ 孤立线状物
    else if(types[i].edj[Prev]>Nr_nor && types[i].edj[Next]>Nr_nor)
    {
      if(types[i].ftype == Nor)
      {
        types[i].ftype = Wire;
      }
    }
  }

  // ── 第三轮：小平面补充 ────────────────────────────
  // 对仍未分类的点（ftype==Nor），检测三点局部共面性：
  //   intersect < cos(172.5°)（略有弯曲）且 距离比 < 1.2（均匀间距）
  //   → 视为小平面，将三点均标记为 Real_Plane
  plsize2 = plsize-1;
  double ratio;
  for(uint i=head+1; i<plsize2; i++)
  {
    if(types[i].range<blind || types[i-1].range<blind || types[i+1].range<blind)
    {
      continue;
    }

    if(types[i-1].dista<1e-8 || types[i].dista<1e-8)
    {
      continue;
    }

    if(types[i].ftype == Nor)
    {
      // ratio = max(d_{i-1}, d_i) / min(d_{i-1}, d_i)：前后间距比（均匀性）
      if(types[i-1].dista > types[i].dista)
      {
        ratio = types[i-1].dista / types[i].dista;
      }
      else
      {
        ratio = types[i].dista / types[i-1].dista;
      }

      // intersect < smallp_intersect(cos172.5°≈-0.991) → 三点夹角接近 180°（共线）
      // ratio < smallp_ratio(1.2) → 前后间距接近（均匀平面）
      if(types[i].intersect<smallp_intersect && ratio < smallp_ratio)
      {
        if(types[i-1].ftype == Nor)
        {
          types[i-1].ftype = Real_Plane;
        }
        if(types[i+1].ftype == Nor)
        {
          types[i+1].ftype = Real_Plane;
        }
        types[i].ftype = Real_Plane;
      }
    }
  }

  // ── 输出阶段：将分类结果填入 pl_surf / pl_corn ──────
  // 平面点（Poss/Real_Plane）→ pl_surf（按 point_filter_num 间隔取点，
  //   若平面段中途被打断则取已积累点的坐标均值）
  // 边缘点（Edge_Jump/Edge_Plane）→ pl_corn
  int last_surface = -1;  // 当前平面段起始索引（-1=无进行中的平面段）
  for(uint j=head; j<plsize; j++)
  {
    if(types[j].ftype==Poss_Plane || types[j].ftype==Real_Plane)
    {
      if(last_surface == -1)
      {
        last_surface = j;  // 开始一个新的平面段
      }

      // 每 point_filter_num 个平面点取一个输出（抽帧）
      if(j == uint(last_surface+point_filter_num-1))
      {
        PointType ap;
        ap.x = pl[j].x;
        ap.y = pl[j].y;
        ap.z = pl[j].z;
        ap.intensity = pl[j].intensity;
        ap.curvature = pl[j].curvature;
        pl_surf.push_back(ap);

        last_surface = -1;
      }
    }
    else
    {
      // 边缘点直接进 pl_corn
      if(types[j].ftype==Edge_Jump || types[j].ftype==Edge_Plane)
      {
        pl_corn.push_back(pl[j]);
      }
      // 若平面段被打断（遇到非平面点），取已积累点的坐标均值加入 pl_surf
      if(last_surface != -1)
      {
        PointType ap;
        for(uint k=last_surface; k<j; k++)
        {
          ap.x += pl[k].x;
          ap.y += pl[k].y;
          ap.z += pl[k].z;
          ap.intensity += pl[k].intensity;
          ap.curvature += pl[k].curvature;
        }
        ap.x /= (j-last_surface);
        ap.y /= (j-last_surface);
        ap.z /= (j-last_surface);
        ap.intensity /= (j-last_surface);
        ap.curvature /= (j-last_surface);
        pl_surf.push_back(ap);
      }
      last_surface = -1;
    }
  }
}

void Preprocess::pub_func(PointCloudXYZI &pl, const ros::Time &ct)
{
  pl.height = 1; pl.width = pl.size();
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "livox";
  output.header.stamp = ct;
}

// ═══════════════════════════════════════════════════════
// 局部平面判断：检查从 i_cur 开始的滑动窗口内的点序列是否构成平面
//
// 算法思路（LOAM 风格）：
//   1. 以 i_cur 为起点，收集一个满足"点间距 < group_dis"的窗口 [i_cur, i_nex)
//      group_dis = (disA × range + disB)² — 距离自适应窗口大小
//   2. 用"点到线距离比值"检验：
//      two_dis = |p[i_nex] - p[i_cur]|²（端到端距离平方）
//      leng_wid = max_j |v1_j × (p[i_nex]-p[i_cur])|²（最大点到线距离平方）
//      若 two_dis² / leng_wid < p2l_ratio(225)，则点云太"宽"，不是直线 → 返回 0
//   3. 对窗口内相邻点间距排序，检验均匀性（间距分布不应有突变）：
//      AVIA：dismax/dismid < limit_maxmid(6.25) 且 dismid/dismin < limit_midmin(6.25)
//      其他：dismax/dismin < limit_maxmin(3.24)
//   4. 通过所有检查 → 返回 1，curr_direct = 归一化的主方向向量
//
// 返回值：
//   1 → 平面（窗口内点序列共线，视为平面段主方向）
//   0 → 非平面（间距不均匀或点到线距离过大）
//   2 → 含盲区点（窗口内有 range < blind 的点）
//
// 注：这里"平面"实际是"直线段"的判断（激光扫描线在平面表面产生直线轨迹）
// ═══════════════════════════════════════════════════════
int Preprocess::plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct)
{
  // 距离自适应的窗口半径²：远距离点允许更大的点间距
  double group_dis = disA*types[i_cur].range + disB;
  group_dis = group_dis * group_dis;

  double two_dis;          // 窗口首尾点间距²
  vector<double> disarr;   // 窗口内所有相邻点间距（用于均匀性检验）
  disarr.reserve(20);

  // 阶段1：先收集至少 group_size 个点的初始窗口
  for(i_nex=i_cur; i_nex<i_cur+group_size; i_nex++)
  {
    if(types[i_nex].range < blind)
    {
      curr_direct.setZero();
      return 2;  // 窗口内有盲区点 → 无效
    }
    disarr.push_back(types[i_nex].dista);
  }

  // 阶段2：继续扩展窗口，直到首尾距离 >= group_dis（自适应大小）
  for(;;)
  {
    if((i_cur >= pl.size()) || (i_nex >= pl.size())) break;

    if(types[i_nex].range < blind)
    {
      curr_direct.setZero();
      return 2;
    }
    // 计算当前 i_nex 到 i_cur 的向量（用于点到线距离计算）
    vx = pl[i_nex].x - pl[i_cur].x;
    vy = pl[i_nex].y - pl[i_cur].y;
    vz = pl[i_nex].z - pl[i_cur].z;
    two_dis = vx*vx + vy*vy + vz*vz;
    if(two_dis >= group_dis)
    {
      break;  // 窗口已足够大
    }
    disarr.push_back(types[i_nex].dista);
    i_nex++;
  }

  // 阶段3：计算最大点到线距离²（检验共线性）
  // 对窗口内每个中间点 j，计算其到 (p[i_cur], p[i_nex]) 连线的距离
  // 用叉积实现：|v1 × d|²（v1=p[j]-p[i_cur], d=(vx,vy,vz)=p[i_nex]-p[i_cur]）
  double leng_wid = 0;
  double v1[3], v2[3];
  for(uint j=i_cur+1; j<i_nex; j++)
  {
    if((j >= pl.size()) || (i_cur >= pl.size())) break;
    v1[0] = pl[j].x - pl[i_cur].x;
    v1[1] = pl[j].y - pl[i_cur].y;
    v1[2] = pl[j].z - pl[i_cur].z;

    // v2 = v1 × d（叉积）→ 其模 = |v1||d|sin(θ) = 点到线距离 × |d|
    v2[0] = v1[1]*vz - vy*v1[2];
    v2[1] = v1[2]*vx - v1[0]*vz;
    v2[2] = v1[0]*vy - vx*v1[1];

    double lw = v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2];
    if(lw > leng_wid)
    {
      leng_wid = lw;  // 记录最大点到线距离²×|d|²
    }
  }

  // 线宽比检验：two_dis² / leng_wid = (|d|²)² / (max_pt2line² × |d|²) = |d|² / max_pt2line²
  // 即 (首尾距离 / 最大点到线距离)² = p2l_ratio(225) → 点到线距离 < 首尾距离/15
  if((two_dis*two_dis/leng_wid) < p2l_ratio)
  {
    curr_direct.setZero();
    return 0;  // 点云太"宽"，不是直线段
  }

  // 阶段4：对相邻点间距进行降序排序，检验均匀性
  // （重用 leng_wid 作为排序临时变量）
  uint disarrsize = disarr.size();
  for(uint j=0; j<disarrsize-1; j++)
  {
    for(uint k=j+1; k<disarrsize; k++)
    {
      if(disarr[j] < disarr[k])
      {
        leng_wid = disarr[j];
        disarr[j] = disarr[k];
        disarr[k] = leng_wid;
      }
    }
  }

  // 排除次小间距过小的情况（说明有重叠点或噪点导致间距异常）
  if(disarr[disarr.size()-2] < 1e-16)
  {
    curr_direct.setZero();
    return 0;
  }

  // 阶段5：间距均匀性检验（不同 LiDAR 类型用不同指标）
  if(lidar_type==AVIA)
  {
    // AVIA 固态 LiDAR 扫描模式特殊，用三分位比值
    double dismax_mid = disarr[0]/disarr[disarrsize/2];      // 最大/中位间距比
    double dismid_min = disarr[disarrsize/2]/disarr[disarrsize-2]; // 中位/次小间距比

    if(dismax_mid>=limit_maxmid || dismid_min>=limit_midmin)
    {
      curr_direct.setZero();
      return 0;  // 间距分布不均匀
    }
  }
  else
  {
    // 旋转 LiDAR（Velodyne/Ouster）只看最大/次小间距比
    double dismax_min = disarr[0] / disarr[disarrsize-2];
    if(dismax_min >= limit_maxmin)
    {
      curr_direct.setZero();
      return 0;
    }
  }

  // 通过所有检验：设置主方向向量（单位化）
  curr_direct << vx, vy, vz;
  curr_direct.normalize();
  return 1;
}

// ═══════════════════════════════════════════════════════
// 边缘跳变验证：在 give_feature 检测到疑似 Edge_Jump 后，
// 进一步验证"安全侧"（未发生深度跳变的方向）是否稳定。
//
// 原理：遮挡边缘处，一侧（安全侧 nor_dir）应该是连续平面，
//   即该侧相邻两对点的间距应相近（d1 ≈ d2），不应发生突变。
//   若安全侧也出现明显间距突变，则当前点可能是噪点而非真正的边缘。
//
// 参数：
//   nor_dir — 安全方向（Prev=0：向前检查 i-1,i-2；Next=1：向后检查 i+1,i+2）
//
// 索引计算：
//   nor_dir=0(Prev)：d1 = types[i-1].dista, d2 = types[i-2].dista
//     → nor_dir-1=-1 → i+(-1)=i-1；3×0-2=-2 → i+(-2)=i-2
//   nor_dir=1(Next)：d1 = types[i].dista,   d2 = types[i+1].dista
//     → nor_dir-1=0  → i+0=i；    3×1-2=1  → i+1
//
// 判断条件（若满足任一则认为安全侧不稳定 → 返回 false，非边缘）：
//   d1 > edgea(2) × d2  → 大间距是小间距的 2 倍以上（突变）
//   d1 - d2 > edgeb(0.1) → 绝对差值过大
// ═══════════════════════════════════════════════════════
bool Preprocess::edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir)
{
  // 检查安全侧的相邻两点是否在盲区内（盲区内无法判断 → 放弃判断返回 false）
  if(nor_dir == 0)  // Prev 方向：检查 i-1, i-2
  {
    if(types[i-1].range<blind || types[i-2].range<blind)
    {
      return false;
    }
  }
  else if(nor_dir == 1)  // Next 方向：检查 i+1, i+2
  {
    if(types[i+1].range<blind || types[i+2].range<blind)
    {
      return false;
    }
  }

  // 取安全侧两对相邻点的间距
  // nor_dir=0: d1=types[i-1].dista, d2=types[i-2].dista
  // nor_dir=1: d1=types[i].dista,   d2=types[i+1].dista
  double d1 = types[i+nor_dir-1].dista;
  double d2 = types[i+3*nor_dir-2].dista;
  double d;

  // 确保 d1 >= d2（便于统一比较方向）
  if(d1<d2)
  {
    d = d1;
    d1 = d2;
    d2 = d;
  }

  // 将平方距离转换为欧氏距离
  d1 = sqrt(d1);
  d2 = sqrt(d2);

  // 安全侧稳定性检验：
  //   若大间距 > edgea(2)× 小间距，或绝对差 > edgeb(0.1m)
  //   → 安全侧也不稳定 → 当前点不是可靠的遮挡边缘 → 返回 false
  if(d1>edgea*d2 || (d1-d2)>edgeb)
  {
    return false;
  }

  // 安全侧稳定 → 确认是真实的遮挡边缘跳变
  return true;
}
