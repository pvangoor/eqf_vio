#pragma once

#include "eigen3/Eigen/Dense"
#include <memory> // PImpl

#include "SO3.h"

using se3vector = Eigen::Matrix<double, 6, 1>;

class SE3 {
  public:
    static Eigen::Matrix4d wedge(const se3vector& u);
    static se3vector vee(const Eigen::Matrix4d& U);
    static SE3 SE3Exp(const se3vector& u);
    static se3vector SE3Log(const SE3& P);
    static SE3 Identity();

    SE3();
    SE3(const SE3& other);
    SE3(const Eigen::Matrix4d& mat);
    ~SE3();

    void setIdentity();
    Eigen::Vector3d operator*(const Eigen::Vector3d& point) const;
    SE3 operator*(const SE3& other) const;
    SE3& operator=(const SE3& other);

    void invert();
    SE3 inverse() const;
    Eigen::Matrix<double, 6, 6> Adjoint() const;

    // Set and get
    Eigen::Matrix4d asMatrix() const;
    void fromMatrix(const Eigen::Matrix4d& mat);
    SO3& R() const;
    Eigen::Vector3d& x() const;

  private:
    class SE3Impl;
    std::unique_ptr<SE3Impl> pimpl;
};