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
#include "eqf_vio/IMUVelocity.h"
#include "eqf_vio/VIOGroup.h"
#include "eqf_vio/VIOState.h"

VIOGroup randomGroupElement(const std::vector<int>& ids);
VIOState randomStateElement(const std::vector<int>& ids);
IMUVelocity randomVelocityElement();
Eigen::VectorXd stateVecDiff(const VIOState& xi1, const VIOState& xi2);
double logNorm(const VIOGroup& X);

VisionMeasurement randomVisionMeasurement(const std::vector<int>& ids);
double stateDistance(const VIOState& xi1, const VIOState& xi2);
double stateDistance(const VIOManifoldState& xi1, const VIOManifoldState& xi2);
double measurementDistance(const VisionMeasurement& y1, const VisionMeasurement& y2);