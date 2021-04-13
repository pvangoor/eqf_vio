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

#include "eigen3/Eigen/Eigen"

constexpr double GRAVITY_CONSTANT = 9.81;

struct IMUVelocity {
    double stamp;
    Eigen::Vector3d omega;
    Eigen::Vector3d accel;

    static IMUVelocity Zero();

    IMUVelocity() = default;
    IMUVelocity(const Eigen::Matrix<double, 6, 1>& vec);

    IMUVelocity operator+(const IMUVelocity& other) const;
    IMUVelocity operator+(const Eigen::Matrix<double, 6, 1>& vec) const;
    IMUVelocity operator-(const Eigen::Matrix<double, 6, 1>& vec) const;
    IMUVelocity operator*(const double& c) const;
};