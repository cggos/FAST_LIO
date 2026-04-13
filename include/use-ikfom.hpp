#ifndef USE_IKFOM_H
#define USE_IKFOM_H

// FAST-LIO2 流形状态定义与 IMU 过程模型
// 基于 IKFoM（Kalman Filters on Differentiable Manifolds）框架
// 定义：状态流形、输入、过程噪声、连续时间动力学及其雅可比

#include <IKFoM_toolkit/esekfom/esekfom.hpp>

// ---------- 基础流形类型别名 ----------
typedef MTK::vect<3, double> vect3;    // R^3 向量（平移、速度、偏置等）
typedef MTK::SO3<double>     SO3;      // 旋转群 SO(3)，内部用四元数表示
// S2：单位球面，用于表示重力方向（仅方向，模长固定为 g）
// 模板参数 <double, 98090, 10000, 1>：数值类型，球半径分子/分母（9.809/1），切映射阶数
typedef MTK::S2<double, 98090, 10000, 1> S2;
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

// ============================================================
// 系统状态流形定义（切空间总维度 = 23）
// 坐标系说明：W=世界系，I=IMU体坐标系，L=LiDAR坐标系
// ============================================================
MTK_BUILD_MANIFOLD(state_ikfom,
  ((vect3, pos))          // p_WI：IMU 在世界系的位置          (3维)
  ((SO3,   rot))          // R_WI：IMU 在世界系的姿态           (3维切空间)
  ((SO3,   offset_R_L_I)) // R_IL：LiDAR 相对 IMU 的旋转外参   (3维切空间)
  ((vect3, offset_T_L_I)) // p_IL：LiDAR 相对 IMU 的平移外参   (3维)
  ((vect3, vel))          // v_WI：IMU 在世界系的速度           (3维)
  ((vect3, bg))           // b_g：陀螺仪偏置                    (3维)
  ((vect3, ba))           // b_a：加速度计偏置                  (3维)
  ((S2,    grav))         // g_W：重力向量方向（球面流形S2）     (2维切空间)
);
// 注：外参 offset_R_L_I / offset_T_L_I 当 extrinsic_est_en=true 时被在线估计

// IMU 输入（测量值）
MTK_BUILD_MANIFOLD(input_ikfom,
  ((vect3, acc))   // a_m：加速度计测量值（含重力，m/s²）
  ((vect3, gyro))  // ω_m：陀螺仪测量值（rad/s）
);

// 过程噪声（12维：陀螺噪声 ng + 加速度噪声 na + 陀螺偏置随机游走 nbg + 加速度偏置随机游走 nba）
MTK_BUILD_MANIFOLD(process_noise_ikfom,
  ((vect3, ng))    // 陀螺仪测量噪声
  ((vect3, na))    // 加速度计测量噪声
  ((vect3, nbg))   // 陀螺偏置随机游走噪声
  ((vect3, nba))   // 加速度偏置随机游走噪声
);

// 构造过程噪声协方差矩阵 Q（12×12 对角矩阵）
// 值均为初始默认值，实际运行时由 yaml 参数覆盖
// Q = diag(σ_ng²·I, σ_na²·I, σ_nbg²·I, σ_nba²·I)
MTK::get_cov<process_noise_ikfom>::type process_noise_cov()
{
	MTK::get_cov<process_noise_ikfom>::type cov = MTK::get_cov<process_noise_ikfom>::type::Zero();
	MTK::setDiagonal<process_noise_ikfom, vect3, 0>(cov, &process_noise_ikfom::ng,  0.0001); // 陀螺仪噪声方差（对角）
	MTK::setDiagonal<process_noise_ikfom, vect3, 3>(cov, &process_noise_ikfom::na,  0.0001); // 加速度计噪声方差
	MTK::setDiagonal<process_noise_ikfom, vect3, 6>(cov, &process_noise_ikfom::nbg, 0.00001); // 陀螺偏置随机游走方差
	MTK::setDiagonal<process_noise_ikfom, vect3, 9>(cov, &process_noise_ikfom::nba, 0.00001); // 加速度偏置随机游走方差
	return cov;
}

// ============================================================
// 连续时间过程模型 f(x, u)
// 返回状态切向量（24维，对应流形各分量的微分）
// 对应 IMU 运动学方程：
//   ṗ = v
//   Ṙ = R · [ω_m - b_g]×        （SO(3) 切向量 = 偏置修正角速度）
//   v̇ = R·(a_m - b_a) + g       （世界系加速度）
//   ḃ_g = 0，ḃ_a = 0，ġ = 0，外参不变
// ============================================================
Eigen::Matrix<double, 24, 1> get_f(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
	// 偏置修正角速度：ω = ω_m - b_g（boxminus 是流形减法，这里 vect3 即普通减法）
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	// 世界系线加速度：R·(a_m - b_a)
	vect3 a_inertial = s.rot * (in.acc-s.ba);
	for(int i = 0; i < 3; i++ ){
		res(i)      = s.vel[i];              // ṗ = v（位置导数 = 速度）
		res(i + 3)  = omega[i];              // Ṙ 的切空间分量 = ω_corrected
		res(i + 12) = a_inertial[i] + s.grav[i]; // v̇ = R·(a-b_a) + g
	}
	// res(6~11): 外参 offset_R/T 的导数 = 0（静态外参假设）
	// res(15~17): b_g 导数 = 0（随机游走，噪声在 Q 中体现）
	// res(18~20): b_a 导数 = 0
	// res(21~23): grav 的切空间导数 = 0（重力方向不变）
	return res;
}

// ============================================================
// 过程模型对状态的雅可比 F = ∂f/∂x（24×23）
// 用于协方差传播：P̂ = F·P·Fᵀ + G·Q·Gᵀ
// 仅列出非零块（其余为零）：
//   ∂ṗ/∂v    = I₃              (行0~2, 列12~14)
//   ∂v̇/∂R    = -R·[a_m-b_a]×   (行12~14, 列3~5)
//   ∂v̇/∂b_a  = -R               (行12~14, 列18~20)
//   ∂v̇/∂g    = ∂g/∂ξ (S2切映射) (行12~14, 列21~22)
//   ∂Ṙ/∂b_g  = -I₃              (行3~5, 列15~17)
// ============================================================
Eigen::Matrix<double, 24, 23> df_dx(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 23> cov = Eigen::Matrix<double, 24, 23>::Zero();
	// ∂ṗ/∂v = I：位置对速度的偏导
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
	// 偏置修正加速度和角速度（用于后续雅可比计算）
	vect3 acc_;
	in.acc.boxminus(acc_, s.ba);   // a_corrected = a_m - b_a
	vect3 omega;
	in.gyro.boxminus(omega, s.bg); // ω_corrected = ω_m - b_g
	// ∂v̇/∂R = -R·[a_m-b_a]×：速度对旋转的偏导（SO(3)切空间）
	cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix()*MTK::hat(acc_);
	// ∂v̇/∂b_a = -R：速度对加速度偏置的偏导
	cov.template block<3, 3>(12, 18) = -s.rot.toRotationMatrix();
	// ∂v̇/∂g：速度对重力的偏导，需要 S2 流形的切映射矩阵（3×2）
	// S2_Mx 返回 grav 分量在全局坐标系下的切基底，列索引 21 对应 state 中 grav 的起始切空间列
	Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	s.S2_Mx(grav_matrix, vec, 21);
	cov.template block<3, 2>(12, 21) = grav_matrix;
	// ∂Ṙ/∂b_g = -I：旋转对陀螺偏置的偏导（SO(3)切空间）
	cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity();
	return cov;
}

// ============================================================
// 过程模型对噪声的雅可比 G = ∂f/∂w（24×12）
// 用于协方差传播：P̂ = F·P·Fᵀ + G·Q·Gᵀ
// 非零块：
//   ∂v̇/∂na   = -R               (行12~14, 列3~5)
//   ∂Ṙ/∂ng   = -I₃              (行3~5,   列0~2)
//   ∂ḃ_g/∂nbg = I₃              (行15~17, 列6~8)
//   ∂ḃ_a/∂nba = I₃              (行18~20, 列9~11)
// ============================================================
Eigen::Matrix<double, 24, 12> df_dw(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
	// ∂v̇/∂na = -R：加速度测量噪声对速度导数的影响
	cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix();
	// ∂Ṙ/∂ng = -I：陀螺噪声对旋转导数的影响
	cov.template block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
	// ∂ḃ_g/∂nbg = I：陀螺偏置随机游走
	cov.template block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();
	// ∂ḃ_a/∂nba = I：加速度偏置随机游走
	cov.template block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();
	return cov;
}

// SO(3) → 欧拉角（ZYX 顺序），返回单位为度
// 内部用四元数 q=[x,y,z,w] 计算，处理万向锁奇异（pitch=±90°）
// 返回 [roll, pitch, yaw]（度），仅用于调试日志输出（laserMapping.cpp）
vect3 SO3ToEuler(const SO3 &orient)
{
	Eigen::Matrix<double, 3, 1> _ang;
	Eigen::Vector4d q_data = orient.coeffs().transpose(); // [x,y,z,w]
	double sqw = q_data[3]*q_data[3];
	double sqx = q_data[0]*q_data[0];
	double sqy = q_data[1]*q_data[1];
	double sqz = q_data[2]*q_data[2];
	double unit = sqx + sqy + sqz + sqw; // 归一化四元数时 unit=1，否则作修正因子
	double test = q_data[3]*q_data[1] - q_data[2]*q_data[0]; // 判断奇异性的测试值

	if (test > 0.49999*unit) { // 万向锁奇异：北极（pitch = +90°）
		_ang << 2 * std::atan2(q_data[0], q_data[3]), M_PI/2, 0;
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		vect3 euler_ang(temp, 3);
		return euler_ang;
	}
	if (test < -0.49999*unit) { // 万向锁奇异：南极（pitch = -90°）
		_ang << -2 * std::atan2(q_data[0], q_data[3]), -M_PI/2, 0;
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		vect3 euler_ang(temp, 3);
		return euler_ang;
	}
	// 正常情况：ZYX 欧拉角
	_ang <<
			std::atan2(2*q_data[0]*q_data[3]+2*q_data[1]*q_data[2] , -sqx - sqy + sqz + sqw), // roll
			std::asin (2*test/unit),                                                              // pitch
			std::atan2(2*q_data[2]*q_data[3]+2*q_data[1]*q_data[0] , sqx - sqy - sqz + sqw);  // yaw
	double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3}; // rad→deg
	vect3 euler_ang(temp, 3);
		// euler_ang[0] = roll, euler_ang[1] = pitch, euler_ang[2] = yaw
	return euler_ang;
}

#endif