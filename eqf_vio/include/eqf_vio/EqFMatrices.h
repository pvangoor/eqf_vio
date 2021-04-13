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

#include "eqf_vio/VIOGroup.h"
#include <functional>

struct EqFStateMatrixA {
    const std::function<Eigen::MatrixXd(const VIOGroup&, const VIOManifoldState&, const IMUVelocity&)> stateMatrixA;
    const std::function<Eigen::VectorXd(const VIOManifoldState&, const VIOManifoldState&)>& coordinateChart;
    const std::function<VIOManifoldState(const Eigen::VectorXd&, const VIOManifoldState&)>& coordinateChartInv;

    Eigen::MatrixXd operator()(const VIOGroup& X, const VIOManifoldState& xi0, const IMUVelocity& imuVel) const {
        return this->stateMatrixA(X, xi0, imuVel);
    };
};
struct EqFInputMatrixB {
    const std::function<Eigen::MatrixXd(const VIOGroup&, const VIOManifoldState&)> inputMatrixB;
    const std::function<Eigen::VectorXd(const VIOManifoldState&, const VIOManifoldState&)>& coordinateChart;

    Eigen::MatrixXd operator()(const VIOGroup& X, const VIOManifoldState& xi0) const {
        return this->inputMatrixB(X, xi0);
    };
};

struct EqFOutputMatrixC {
    const std::function<Eigen::MatrixXd(const VIOManifoldState&)> outputMatrixC;
    const std::function<VIOManifoldState(const Eigen::VectorXd&, const VIOManifoldState&)>& stateChartInv;
    const std::function<Eigen::VectorXd(const VisionMeasurement&, const VisionMeasurement&)>& outputChart;

    Eigen::MatrixXd operator()(const VIOManifoldState& xi0) const { return this->outputMatrixC(xi0); };
};

extern const EqFStateMatrixA EqFStateMatrixA_euclid;
extern const EqFInputMatrixB EqFInputMatrixB_euclid;
extern const EqFOutputMatrixC EqFOutputMatrixC_euclid;

Eigen::MatrixXd EqFStateMatrixA_invdepth(const VIOGroup& X, const VIOManifoldState& xi0, const IMUVelocity& imuVel);
Eigen::MatrixXd EqFStateMatrixA_invdepth(const VIOGroup& X, const VIOState& xi0, const IMUVelocity& imuVel);

VIOAlgebra liftInnovation(const Eigen::VectorXd& baseInnovation, const VIOManifoldState& xi0);
VIOAlgebra liftTotalSpaceInnovation(const Eigen::VectorXd& totalInnovation, const VIOState& xi0);
VIOGroup liftTotalSpaceInnovationDiscrete(const Eigen::VectorXd& totalInnovation, const VIOState& xi0);

VIOAlgebra liftInnovation(
    const Eigen::VectorXd& baseInnovation, const VIOState& xi0, const VIOGroup& X, const Eigen::MatrixXd& Sigma);

Eigen::VectorXd bundleLift(
    const Eigen::VectorXd& baseInnovation, const VIOState& xi0, const VIOGroup& X, const Eigen::MatrixXd& Sigma);