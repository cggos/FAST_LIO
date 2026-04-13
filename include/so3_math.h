#ifndef SO3_MATH_H
#define SO3_MATH_H

// SO(3) 李群数学工具库
// 提供旋转矩阵的指数映射(Exp)、对数映射(Log)和反对称矩阵工具
// 所有函数均支持模板类型（float/double）

#include <math.h>
#include <Eigen/Core>

// 反对称矩阵宏：将向量 v=[v0,v1,v2] 展开为 3x3 反对称矩阵的行主序元素
// 结果矩阵:  [  0  -v2   v1 ]
//            [ v2    0  -v0 ]
//            [-v1   v0    0 ]
// 用法: M << SKEW_SYM_MATRX(v);
#define SKEW_SYM_MATRX(v) 0.0,-v[2],v[1],v[2],0.0,-v[0],-v[1],v[0],0.0

// 构造反对称（叉积）矩阵 [v]×，即 v× p = v.cross(p)
template<typename T>
Eigen::Matrix<T, 3, 3> skew_sym_mat(const Eigen::Matrix<T, 3, 1> &v)
{
    Eigen::Matrix<T, 3, 3> skew_sym_mat;
    skew_sym_mat<<0.0,-v[2],v[1],v[2],0.0,-v[0],-v[1],v[0],0.0;
    return skew_sym_mat;
}

// SO(3) 指数映射（重载1）：旋转向量 → 旋转矩阵
// 输入: ang — 旋转向量（方向=旋转轴，模长=旋转角 θ，单位 rad）
// 输出: R ∈ SO(3)
// 公式（Rodrigues）: R = I + sin(θ)/θ·[ω]× + (1-cos(θ))/θ²·[ω]×²
// 当 ||ang|| ≈ 0 时退化为 I（避免除零）
template<typename T>
Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1> &&ang)
{
    T ang_norm = ang.norm();
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();
    if (ang_norm > 0.0000001)
    {
        Eigen::Matrix<T, 3, 1> r_axis = ang / ang_norm;  // 单位旋转轴
        Eigen::Matrix<T, 3, 3> K;
        K << SKEW_SYM_MATRX(r_axis);
        /// Roderigous Tranformation
        return Eye3 + std::sin(ang_norm) * K + (1.0 - std::cos(ang_norm)) * K * K;
    }
    else
    {
        return Eye3;  // 旋转角近零，返回单位矩阵
    }
}

// SO(3) 指数映射（重载2）：角速度 × 时间步 → 旋转矩阵
// 输入: ang_vel — 角速度向量（rad/s），dt — 时间步长（s）
// 用于 IMU 前向传播：R_{k+1} = R_k · Exp(ω * dt)
template<typename T, typename Ts>
Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1> &ang_vel, const Ts &dt)
{
    T ang_vel_norm = ang_vel.norm();
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();

    if (ang_vel_norm > 0.0000001)
    {
        Eigen::Matrix<T, 3, 1> r_axis = ang_vel / ang_vel_norm;  // 单位旋转轴
        Eigen::Matrix<T, 3, 3> K;

        K << SKEW_SYM_MATRX(r_axis);

        T r_ang = ang_vel_norm * dt;  // 旋转角 θ = ||ω|| * dt

        /// Roderigous Tranformation
        return Eye3 + std::sin(r_ang) * K + (1.0 - std::cos(r_ang)) * K * K;
    }
    else
    {
        return Eye3;
    }
}

// SO(3) 指数映射（重载3）：直接传入旋转向量三分量
// 用于 common_lib.h 中的 StatesGroup::operator+ （状态流形加法）
template<typename T>
Eigen::Matrix<T, 3, 3> Exp(const T &v1, const T &v2, const T &v3)
{
    T &&norm = sqrt(v1 * v1 + v2 * v2 + v3 * v3);
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();
    if (norm > 0.00001)
    {
        T r_ang[3] = {v1 / norm, v2 / norm, v3 / norm};
        Eigen::Matrix<T, 3, 3> K;
        K << SKEW_SYM_MATRX(r_ang);

        /// Roderigous Tranformation
        return Eye3 + std::sin(norm) * K + (1.0 - std::cos(norm)) * K * K;
    }
    else
    {
        return Eye3;
    }
}

// SO(3) 对数映射：旋转矩阵 → 旋转向量（李代数 so(3)）
// 公式: θ = arccos((tr(R)-1)/2)，ω = θ/(2sinθ) · [R32-R23, R13-R31, R21-R12]^T
// 当 θ ≈ 0 时用一阶近似 ω ≈ 0.5 * [R32-R23, ...]（避免 sin(θ)/θ 的奇异）
// 用于计算状态差 x1 ⊖ x2 中的旋转分量（iEKF 迭代更新）
template<typename T>
Eigen::Matrix<T,3,1> Log(const Eigen::Matrix<T, 3, 3> &R)
{
    // 旋转角 θ，tr(R) > 3-ε 时视为零旋转
    T theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
    // 反对称部分的轴向量 K = [R32-R23, R13-R31, R21-R12]
    Eigen::Matrix<T,3,1> K(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1));
    // |θ| < 0.001 时用一阶近似；否则用完整公式
    return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}

// 旋转矩阵转欧拉角（ZYX 顺序：yaw-pitch-roll）
// 返回 [roll, pitch, yaw]（单位 rad）
// 奇异点（gimbal lock）检测：sy < 1e-6 时 pitch ≈ ±90°
template<typename T>
Eigen::Matrix<T, 3, 1> RotMtoEuler(const Eigen::Matrix<T, 3, 3> &rot)
{
    T sy = sqrt(rot(0,0)*rot(0,0) + rot(1,0)*rot(1,0));
    bool singular = sy < 1e-6;  // sy 接近 0 表示万向锁奇异
    T x, y, z;
    if(!singular)
    {
        x = atan2(rot(2, 1), rot(2, 2));   // roll
        y = atan2(-rot(2, 0), sy);          // pitch
        z = atan2(rot(1, 0), rot(0, 0));    // yaw
    }
    else
    {
        x = atan2(-rot(1, 2), rot(1, 1));   // 万向锁退化情形
        y = atan2(-rot(2, 0), sy);
        z = 0;
    }
    Eigen::Matrix<T, 3, 1> ang(x, y, z);
    return ang;
}

#endif
