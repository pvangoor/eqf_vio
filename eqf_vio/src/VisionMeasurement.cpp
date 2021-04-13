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

#include "eqf_vio/VisionMeasurement.h"
#include "eqf_vio/VIOState.h"

using namespace std;
using namespace Eigen;

VectorXd outputCoordinateChart(const VisionMeasurement& y, const VisionMeasurement& y0) {
    const int N = y.numberOfBearings;
    assert(y.bearings.size() == y0.bearings.size());
    VectorXd delta(2 * N);
    for (int i = 0; i < N; ++i) {
        assert(y.bearings[i].id == y0.bearings[i].id);
        const Vector2d yiCoords = stereoSphereChart(y.bearings[i].p, y0.bearings[i].p);
        delta.block<2, 1>(2 * i, 0) = yiCoords;
    }
    return delta;
}

VisionMeasurement outputCoordinateChartInv(const Eigen::VectorXd& delta, const VisionMeasurement& y0) {
    const int N = y0.numberOfBearings;
    assert(delta.size() == 2 * N);
    VisionMeasurement y;
    y.stamp = y0.stamp;
    y.numberOfBearings = y0.numberOfBearings;
    y.bearings.resize(N);

    for (int i = 0; i < N; ++i) {
        y.bearings[i].p = stereoSphereChartInv(delta.block<2, 1>(2 * i, 0), y0.bearings[i].p);
        y.bearings[i].id = y0.bearings[i].id;
    }

    return y;
}
