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

#include "SE3.h"
#include "SOT3.h"
#include "eqf_vio/VIOState.h"

struct VIOGroup {
    SE3 A;
    Eigen::Vector3d w;
    std::vector<SOT3> Q;
    std::vector<int> id;

    [[nodiscard]] VIOGroup operator*(const VIOGroup& other) const;
    [[nodiscard]] static VIOGroup Identity(const std::vector<int>& id = {});
    [[nodiscard]] VIOGroup inverse() const;
};

struct VIOAlgebra {
    Eigen::Matrix<double, 6, 1> U;
    Eigen::Vector3d u;
    std::vector<Eigen::Vector4d> W;
    std::vector<int> id;

    [[nodiscard]] VIOAlgebra operator*(const double& c) const;
    [[nodiscard]] VIOAlgebra operator-() const;
    [[nodiscard]] VIOAlgebra operator+(const VIOAlgebra& other) const;
    [[nodiscard]] VIOAlgebra operator-(const VIOAlgebra& other) const;
};
[[nodiscard]] VIOAlgebra operator*(const double& c, const VIOAlgebra& lambda);

[[nodiscard]] VIOState stateGroupAction(const VIOGroup& X, const VIOState& state);
[[nodiscard]] VIOManifoldState stateGroupAction(const VIOGroup& X, const VIOManifoldState& state);
[[nodiscard]] VisionMeasurement outputGroupAction(const VIOGroup& X, const VisionMeasurement& measurement);

[[nodiscard]] VIOAlgebra liftVelocity(const VIOManifoldState& state, const IMUVelocity& velocity);

[[nodiscard]] VIOGroup liftVelocityDiscrete(
    const VIOManifoldState& state, const IMUVelocity& velocity, const double& dt);

[[nodiscard]] VIOGroup VIOExp(const VIOAlgebra& lambda);