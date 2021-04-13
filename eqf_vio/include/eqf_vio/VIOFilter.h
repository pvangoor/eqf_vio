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

#include <memory>
#include <ostream>

#include "IMUVelocity.h"
#include "VIOGroup.h"
#include "VIOState.h"
#include "VisionMeasurement.h"

#define SIGMA_BASE_SIZE 11

struct AuxiliaryFilterData {
    Eigen::Quaterniond initialAttitude;
    Eigen::Vector3d initialPosition;
    double initialTime;
    double measurementVariance = 0.1;
    double processVariance = 1.0;
    double omegaVariance = 0.1;
    double accelVariance = 0.1;
    SE3 cameraOffset = SE3::Identity();
};

class VIOFilter {
  protected:
    AuxiliaryFilterData auxData;

    Eigen::Matrix<double, 6, 1> inputBias = Eigen::Matrix<double, 6, 1>::Zero();
    VIOState xi0;
    VIOGroup X = VIOGroup::Identity();
    Eigen::MatrixXd Sigma = Eigen::MatrixXd::Identity(SIGMA_BASE_SIZE, SIGMA_BASE_SIZE);

    bool initialisedFlag = false;
    double currentTime = -1;
    IMUVelocity currentVelocity = IMUVelocity::Zero();

    IMUVelocity accumulatedVelocity = IMUVelocity::Zero();
    double accumulatedTime = 0.0;

    bool integrateUpToTime(const double& newTime, const bool doRiccati = true);
    void addNewLandmarks(const VisionMeasurement& measurement);
    void removeOldLandmarks(const VisionMeasurement& measurement);
    void removeOutliers(VisionMeasurement& measurement);
    void removeLandmarkAtIndex(const int& idx);
    VisionMeasurement matchMeasurementsToState(const VisionMeasurement& measurement) const;

  public:
    // Settings
    struct Settings;
    std::unique_ptr<VIOFilter::Settings> settings;

    // Setup
    VIOFilter() = default;
    VIOFilter(const AuxiliaryFilterData& auxiliaryData);
    VIOFilter(const AuxiliaryFilterData& auxiliaryData, const VIOFilter::Settings& settings);
    VIOFilter(const VIOFilter::Settings& settings);
    void initialiseFromIMUData(const IMUVelocity& imuVelocity);
    void reset();

    // Input
    void setAuxiliaryData(const AuxiliaryFilterData& auxiliaryData);
    void setInertialPoints(const std::vector<Point3d>& inertialPoints);
    void processIMUData(const IMUVelocity& imuVelocity);
    void processVisionData(const VisionMeasurement& measurement);

    // Output
    double getTime() const;
    VIOState stateEstimate() const;
    Eigen::MatrixXd stateCovariance() const;
    friend std::ostream& operator<<(std::ostream& os, const VIOFilter& filter);
};

std::ostream& operator<<(std::ostream& os, const VIOFilter& filter);