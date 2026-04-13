#ifndef COMMON_LIB_H
#define COMMON_LIB_H

// FAST-LIO2 公共类型定义与工具函数
// 包含：系统常量、类型别名、数据结构（MeasureGroup/StatesGroup）、平面拟合工具

#include <so3_math.h>
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <fast_lio/Pose6D.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>

using namespace std;
using namespace Eigen;

#define USE_IKFOM  // 启用 IKFoM 流形滤波器（相对于旧版 EKF 的编译开关）

// ---------- 物理常量 ----------
#define PI_M (3.14159265358)
#define G_m_s2 (9.81)         // 重力加速度（广东/中国地区实测值 m/s²）

// ---------- 状态/噪声维度（用于旧版 StatesGroup，IKFoM 版本已不依赖） ----------
#define DIM_STATE (18)        // 旧版状态维度（SO(3)用3维表示）
#define DIM_PROC_N (12)       // 旧版过程噪声维度

// ---------- 地图管理参数 ----------
#define CUBE_LEN  (6.0)       // 地图立方体单元边长（暂未直接使用，实际由 yaml 的 cube_side_length 覆盖）
#define LIDAR_SP_LEN    (2)   // LiDAR 特殊点（X轴方向参考点）偏移量，用于 FOV 分割

// ---------- 滤波器初始化参数 ----------
#define INIT_COV   (1)        // StatesGroup 协方差初始值（对角线）

// ---------- 特征匹配参数 ----------
#define NUM_MATCH_POINTS    (5)    // ikd-Tree 最近邻搜索数量（用于平面拟合）
#define MAX_MEAS_DIM        (10000) // 单次 iEKF 更新最大测量维度（有效点上限）

// ---------- 工具宏 ----------
#define VEC_FROM_ARRAY(v)        v[0],v[1],v[2]                        // 数组→展开为3个值（用于Eigen初始化）
#define MAT_FROM_ARRAY(v)        v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]  // 数组→9个值（旋转矩阵行主序）
#define CONSTRAIN(v,min,max)     ((v>min)?((v<max)?v:max):min)         // 值域裁剪
#define ARRAY_FROM_EIGEN(mat)    mat.data(), mat.data() + mat.rows() * mat.cols()
#define STD_VEC_FROM_EIGEN(mat)  vector<decltype(mat)::Scalar> (mat.data(), mat.data() + mat.rows() * mat.cols())
#define DEBUG_FILE_DIR(name)     (string(string(ROOT_DIR) + "Log/"+ name))  // 调试日志路径（ROOT_DIR 由 CMake 注入）

// ---------- 类型别名 ----------
typedef fast_lio::Pose6D Pose6D;           // 自定义6DoF位姿消息（存储IMU轨迹中间帧）
// PointType 复用 pcl::PointXYZINormal：
//   x,y,z     → 3D 坐标
//   intensity  → 反射率
//   curvature  → 【重要】复用为点相对帧首的时间偏移（单位：ms），用于运动畸变补偿
//   normal_x/y/z → 未使用（预处理阶段置零）
typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
typedef vector<PointType, Eigen::aligned_allocator<PointType>>  PointVector;  // 对齐分配器，用于 ikd-Tree
typedef Vector3d V3D;
typedef Matrix3d M3D;
typedef Vector3f V3F;
typedef Matrix3f M3F;

// 动态矩阵/向量快捷宏
#define MD(a,b)  Matrix<double, (a), (b)>
#define VD(a)    Matrix<double, (a), 1>
#define MF(a,b)  Matrix<float, (a), (b)>
#define VF(a)    Matrix<float, (a), 1>

// 全局常用矩阵常量
M3D Eye3d(M3D::Identity());
M3F Eye3f(M3F::Identity());
V3D Zero3d(0, 0, 0);
V3F Zero3f(0, 0, 0);

// 一帧 LiDAR 扫描 + 对应时段内所有 IMU 测量的打包结构
// 由 sync_packages() 填充，传递给 ImuProcess::Process() 处理
struct MeasureGroup
{
    MeasureGroup()
    {
        lidar_beg_time = 0.0;
        this->lidar.reset(new PointCloudXYZI());
    };
    double lidar_beg_time;                    // LiDAR 帧起始时间戳（秒）
    double lidar_end_time;                    // LiDAR 帧结束时间戳（秒，由最后一个点的 curvature 推算）
    PointCloudXYZI::Ptr lidar;               // 原始（已预处理但未去畸变）点云，LiDAR 系
    deque<sensor_msgs::Imu::ConstPtr> imu;   // 覆盖 [lidar_beg_time, lidar_end_time] 的 IMU 测量序列
};

// 旧版（非 IKFoM）EKF 状态结构（当前代码已使用 IKFoM 版本 state_ikfom，此结构仅保留兼容性）
// 状态向量（18维）：[rot(3), pos(3), vel(3), bias_g(3), bias_a(3), gravity(3)]
// 定义了流形加法/减法运算符，用于 EKF 更新时的状态增量叠加
struct StatesGroup
{
    StatesGroup() {
		this->rot_end = M3D::Identity();  // 帧末 IMU 姿态（旋转矩阵）
		this->pos_end = Zero3d;           // 帧末 IMU 位置（世界系，m）
        this->vel_end = Zero3d;           // 帧末 IMU 速度（世界系，m/s）
        this->bias_g  = Zero3d;           // 陀螺偏置（rad/s）
        this->bias_a  = Zero3d;           // 加速度偏置（m/s²）
        this->gravity = Zero3d;           // 重力向量（世界系，m/s²）
        this->cov     = MD(DIM_STATE,DIM_STATE)::Identity() * INIT_COV;
        this->cov.block<9,9>(9,9) = MD(9,9)::Identity() * 0.00001; // 偏置/重力初始协方差更小
	};

    StatesGroup(const StatesGroup& b) {
		this->rot_end = b.rot_end;
		this->pos_end = b.pos_end;
        this->vel_end = b.vel_end;
        this->bias_g  = b.bias_g;
        this->bias_a  = b.bias_a;
        this->gravity = b.gravity;
        this->cov     = b.cov;
	};

    StatesGroup& operator=(const StatesGroup& b)
	{
        this->rot_end = b.rot_end;
		this->pos_end = b.pos_end;
        this->vel_end = b.vel_end;
        this->bias_g  = b.bias_g;
        this->bias_a  = b.bias_a;
        this->gravity = b.gravity;
        this->cov     = b.cov;
        return *this;
	};

    // 流形加法 x ⊞ δ：将切空间增量叠加到当前状态
    // SO(3) 分量用右乘指数映射：R ← R · Exp(δω)
    // 其余分量（R^n）直接相加
    StatesGroup operator+(const Matrix<double, DIM_STATE, 1> &state_add)
	{
        StatesGroup a;
		a.rot_end = this->rot_end * Exp(state_add(0,0), state_add(1,0), state_add(2,0)); // R·Exp(δω)
		a.pos_end = this->pos_end + state_add.block<3,1>(3,0);
        a.vel_end = this->vel_end + state_add.block<3,1>(6,0);
        a.bias_g  = this->bias_g  + state_add.block<3,1>(9,0);
        a.bias_a  = this->bias_a  + state_add.block<3,1>(12,0);
        a.gravity = this->gravity + state_add.block<3,1>(15,0);
        a.cov     = this->cov;
		return a;
	};

    StatesGroup& operator+=(const Matrix<double, DIM_STATE, 1> &state_add)
	{
        this->rot_end = this->rot_end * Exp(state_add(0,0), state_add(1,0), state_add(2,0));
		this->pos_end += state_add.block<3,1>(3,0);
        this->vel_end += state_add.block<3,1>(6,0);
        this->bias_g  += state_add.block<3,1>(9,0);
        this->bias_a  += state_add.block<3,1>(12,0);
        this->gravity += state_add.block<3,1>(15,0);
		return *this;
	};

    // 流形减法 x1 ⊖ x2：返回从 x2 到 x1 的切空间增量
    // SO(3) 分量：Log(R2^T · R1)（右扰动对应的轴角差）
    // 其余分量：直接差值
    Matrix<double, DIM_STATE, 1> operator-(const StatesGroup& b)
	{
        Matrix<double, DIM_STATE, 1> a;
        M3D rotd(b.rot_end.transpose() * this->rot_end);  // R2^T · R1
        a.block<3,1>(0,0)  = Log(rotd);                   // SO(3) 对数映射→旋转向量
        a.block<3,1>(3,0)  = this->pos_end - b.pos_end;
        a.block<3,1>(6,0)  = this->vel_end - b.vel_end;
        a.block<3,1>(9,0)  = this->bias_g  - b.bias_g;
        a.block<3,1>(12,0) = this->bias_a  - b.bias_a;
        a.block<3,1>(15,0) = this->gravity - b.gravity;
		return a;
	};

    void resetpose()
    {
        this->rot_end = M3D::Identity();
		this->pos_end = Zero3d;
        this->vel_end = Zero3d;
    }

	M3D rot_end;      // the estimated attitude (rotation matrix) at the end lidar point
    V3D pos_end;      // the estimated position at the end lidar point (world frame)
    V3D vel_end;      // the estimated velocity at the end lidar point (world frame)
    V3D bias_g;       // gyroscope bias
    V3D bias_a;       // accelerator bias
    V3D gravity;      // the estimated gravity acceleration
    Matrix<double, DIM_STATE, DIM_STATE>  cov;     // states covariance
};

template<typename T>
T rad2deg(T radians)
{
  return radians * 180.0 / PI_M;
}

template<typename T>
T deg2rad(T degrees)
{
  return degrees * PI_M / 180.0;
}

template<typename T>
auto set_pose6d(const double t, const Matrix<T, 3, 1> &a, const Matrix<T, 3, 1> &g, \
                const Matrix<T, 3, 1> &v, const Matrix<T, 3, 1> &p, const Matrix<T, 3, 3> &R)
{
    Pose6D rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++)
    {
        rot_kp.acc[i] = a(i);
        rot_kp.gyr[i] = g(i);
        rot_kp.vel[i] = v(i);
        rot_kp.pos[i] = p(i);
        for (int j = 0; j < 3; j++)  rot_kp.rot[i*3+j] = R(i,j);
    }
    return move(rot_kp);
}

// 法向量估计（辅助版本，当前主流程使用 esti_plane）
// 平面方程 Ax+By+Cz+D=0，令 D=-1，转化为线性系统：
//   [x_i, y_i, z_i] · [A/D, B/D, C/D]^T = -1
// 用列主元 QR 分解求解，然后归一化得法向量
// 返回 false 表示任意点到拟合平面的距离超过 threshold（平面质量不佳）
template<typename T>
bool esti_normvector(Matrix<T, 3, 1> &normvec, const PointVector &point, const T &threshold, const int &point_num)
{
    MatrixXf A(point_num, 3);
    MatrixXf b(point_num, 1);
    b.setOnes();
    b *= -1.0f;  // 右端向量全为 -1

    for (int j = 0; j < point_num; j++)
    {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }
    normvec = A.colPivHouseholderQr().solve(b);  // 最小二乘求解法向量（未归一化）

    // 检验每个点到平面的距离（未归一化距离）是否在阈值内
    for (int j = 0; j < point_num; j++)
    {
        if (fabs(normvec(0) * point[j].x + normvec(1) * point[j].y + normvec(2) * point[j].z + 1.0f) > threshold)
        {
            return false;  // 拟合质量差，该平面无效
        }
    }

    normvec.normalize();  // 归一化为单位法向量
    return true;
}

// 两点间距离平方（避免开方，用于地图增量插入时的距离比较）
float calc_dist(PointType p1, PointType p2){
    float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    return d;
}

// 主流程平面拟合函数（h_share_model 中调用）
// 用 NUM_MATCH_POINTS=5 个最近邻点拟合平面 n^T·p + d = 0
// 输出 pca_result = [nx, ny, nz, d]（单位法向量 + 原点距离）
// 求解方法与 esti_normvector 相同（令 D=-1 转线性系统，QR 分解）
// threshold：任意最近邻点到拟合平面距离超过此值则返回 false（平面无效）
// 该阈值在 laserMapping.cpp 中设为 0.1（米）
template<typename T>
bool esti_plane(Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold)
{
    Matrix<T, NUM_MATCH_POINTS, 3> A;
    Matrix<T, NUM_MATCH_POINTS, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;  // 右端向量全为 -1

    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }

    // 最小二乘求解未归一化法向量
    Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

    // 归一化：n̂ = normvec/||normvec||，d = 1/||normvec||
    T n = normvec.norm();
    pca_result(0) = normvec(0) / n;  // nx
    pca_result(1) = normvec(1) / n;  // ny
    pca_result(2) = normvec(2) / n;  // nz
    pca_result(3) = 1.0 / n;         // d（原点到平面的有符号距离）

    // 验证所有最近邻点到平面的距离均在阈值内（验证平面拟合质量）
    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold)
        {
            return false;  // 点不在平面上（可能是边缘/角点，非平面特征）
        }
    }
    return true;
}

#endif