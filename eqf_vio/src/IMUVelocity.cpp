#include "eqf_vio/IMUVelocity.h"

IMUVelocity IMUVelocity::Zero() {
    IMUVelocity result;
    result.stamp = 0;
    result.omega.setZero();
    result.accel.setZero();
    return result;
}

IMUVelocity::IMUVelocity(const Eigen::Matrix<double, 6, 1>& vec) {
    stamp = 0;
    omega = vec.block<3, 1>(0, 0);
    accel = vec.block<3, 1>(3, 0);
}

IMUVelocity IMUVelocity::operator+(const IMUVelocity& other) const {
    IMUVelocity result;
    result.stamp = (this->stamp > 0) ? this->stamp : other.stamp;
    result.omega = this->omega + other.omega;
    result.accel = this->accel + other.accel;
    return result;
}

IMUVelocity IMUVelocity::operator+(const Eigen::Matrix<double, 6, 1>& vec) const {
    IMUVelocity result;
    result.stamp = this->stamp;
    result.omega = this->omega + vec.block<3, 1>(0, 0);
    result.accel = this->accel + vec.block<3, 1>(3, 0);
    return result;
}

IMUVelocity IMUVelocity::operator-(const Eigen::Matrix<double, 6, 1>& vec) const { return *this + (-vec); }

IMUVelocity IMUVelocity::operator*(const double& c) const {
    IMUVelocity result;
    result.stamp = this->stamp;
    result.omega = this->omega * c;
    result.accel = this->accel * c;
    return result;
}