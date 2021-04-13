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

class SOT3 {
  public:
    static Eigen::Matrix4d wedge(const Eigen::Vector4d& u);
    static Eigen::Vector4d vee(const Eigen::Matrix4d& U);
    static SOT3 SOT3Exp(const Eigen::Vector4d& u);
    static Eigen::Vector4d SOT3Log(const SOT3& T);
    static SOT3 Identity();

    SOT3();
    SOT3(const SOT3& other);
    ~SOT3();

    SOT3(const Eigen::Matrix4d& mat);
    void setIdentity();
    Eigen::Vector3d operator*(const Eigen::Vector3d& point) const;
    SOT3 operator*(const SOT3& other) const;
    SOT3& operator=(const SOT3& other);
    Eigen::Vector3d applyInverse(const Eigen::Vector3d& p) const;

    void invert();
    SOT3 inverse() const;

    // Set and get
    Eigen::Matrix4d asMatrix() const;
    Eigen::Matrix3d asMatrix3d() const;
    void fromMatrix(const Eigen::Matrix4d& mat);
    SO3& R() const;
    double& a() const;

  private:
    class SOT3Impl;
    std::unique_ptr<SOT3Impl> pimpl;
};