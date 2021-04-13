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
#include <vector>

struct Point3d;
struct VisionMeasurement {
    double stamp;
    int numberOfBearings;
    std::vector<Point3d> bearings;
};

Eigen::VectorXd outputCoordinateChart(const VisionMeasurement& y, const VisionMeasurement& y0);
VisionMeasurement outputCoordinateChartInv(const Eigen::VectorXd& delta, const VisionMeasurement& y0);