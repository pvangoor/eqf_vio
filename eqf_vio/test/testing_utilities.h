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