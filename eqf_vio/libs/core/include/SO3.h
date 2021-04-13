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

class SO3 {
  public:
    static Eigen::Matrix3d skew(const Eigen::Vector3d& omega);
    static Eigen::Vector3d vex(const Eigen::Matrix3d& Omega);
    static SO3 SO3Exp(const Eigen::Vector3d& omega);
    static Eigen::Vector3d SO3Log(const SO3& R);
    static SO3 SO3FromVectors(const Eigen::Vector3d& origin, const Eigen::Vector3d& dest);
    static SO3 Identity();

    SO3();
    SO3(const SO3& other);
    ~SO3();

    SO3(const Eigen::Matrix3d& mat);
    SO3(const Eigen::Quaterniond& quat);
    SO3 inverse() const;

    void setIdentity();
    Eigen::Vector3d operator*(const Eigen::Vector3d& point) const;
    SO3 operator*(const SO3& other) const;
    SO3& operator=(const SO3& other);
    Eigen::Vector3d applyInverse(const Eigen::Vector3d& point) const;

    void invert();

    // Set and get
    Eigen::Matrix3d asMatrix() const;
    Eigen::Quaterniond asQuaternion() const;
    void fromMatrix(const Eigen::Matrix3d& mat);
    void fromQuaternion(const Eigen::Quaterniond& quat);

  private:
    class SO3Impl;
    std::unique_ptr<SO3Impl> pimpl;
};