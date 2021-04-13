// Copyright (C) 2021 Pieter van Goor
// 
// This file is part of EqF VIO.
// 
// EqF VIO is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// EqF VIO is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with EqF VIO.  If not, see <http://www.gnu.org/licenses/>.

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