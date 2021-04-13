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

#include "yaml-cpp/yaml.h"

#include "VIOFilter.h"
#include "common.h"

struct VIOFilter::Settings {
    double biasOmegaProcessVariance = 0.001;
    double biasAccelProcessVariance = 0.001;
    double gravityProcessVariance = 0.001;
    double velocityProcessVariance = 0.001;
    double pointProcessVariance = 0.001;
    double velOmegaVariance = 0.1;
    double velAccelVariance = 0.1;
    double measurementVariance = 0.1;
    double initialGravityVariance = 1.0;
    double initialVelocityVariance = 1.0;
    double initialPointVariance = 1.0;
    double initialBiasOmegaVariance = 1.0;
    double initialBiasAccelVariance = 1.0;
    double initialSceneDepth = 1.0;
    double outlierThreshold = 0.01;
    bool useInnovationLift = true;
    bool useDiscreteInnovationLift = true;
    bool useDiscreteVelocityLift = true;
    bool fastRiccati = false;
    Eigen::Vector3d initialAccelBias = Eigen::Vector3d::Zero();
    Eigen::Vector3d initialOmegaBias = Eigen::Vector3d::Zero();
    SE3 cameraOffset = SE3::Identity();

    Settings() = default;
    Settings(const YAML::Node& configNode);
};

inline VIOFilter::Settings::Settings(const YAML::Node& configNode) {
    // Configure gain matrices
    safeConfig(configNode["biasOmegaProcessVariance"], biasOmegaProcessVariance);
    safeConfig(configNode["biasOmegaProcessVariance"], biasOmegaProcessVariance);
    safeConfig(configNode["biasAccelProcessVariance"], biasAccelProcessVariance);
    safeConfig(configNode["gravityProcessVariance"], gravityProcessVariance);
    safeConfig(configNode["velocityProcessVariance"], velocityProcessVariance);
    safeConfig(configNode["pointProcessVariance"], pointProcessVariance);
    safeConfig(configNode["measurementVariance"], measurementVariance);
    safeConfig(configNode["velOmegaVariance"], velOmegaVariance);
    safeConfig(configNode["velAccelVariance"], velAccelVariance);

    // Configure initial gains
    safeConfig(configNode["initialGravityVariance"], initialGravityVariance);
    safeConfig(configNode["initialVelocityVariance"], initialVelocityVariance);
    safeConfig(configNode["initialPointVariance"], initialPointVariance);
    safeConfig(configNode["initialBiasOmegaVariance"], initialBiasOmegaVariance);
    safeConfig(configNode["initialBiasAccelVariance"], initialBiasAccelVariance);

    // Configure computation methods
    safeConfig(configNode["useInnovationLift"], useInnovationLift);
    safeConfig(configNode["useDiscreteInnovationLift"], useDiscreteInnovationLift);
    safeConfig(configNode["useDiscreteVelocityLift"], useDiscreteVelocityLift);
    safeConfig(configNode["fastRiccati"], fastRiccati);

    // Configure extra settings
    safeConfig(configNode["outlierThreshold"], outlierThreshold);
    safeConfig(configNode["initialSceneDepth"], initialSceneDepth);

    // Configure vector settings
    if (configNode["initialAccelBias"]) {
        initialAccelBias.block<3, 1>(0, 0) << configNode["initialAccelBias"][0].as<double>(),
            configNode["initialAccelBias"][1].as<double>(), configNode["initialAccelBias"][2].as<double>();
    }
    if (configNode["initialOmegaBias"]) {
        initialOmegaBias.block<3, 1>(0, 0) << configNode["initialOmegaBias"][0].as<double>(),
            configNode["initialOmegaBias"][1].as<double>(), configNode["initialOmegaBias"][2].as<double>();
    }

    if (configNode["cameraOffset"]) {
        assert(configNode["cameraOffset"][0].as<std::string>() == "xw");
        Eigen::Vector3d cameraPosition;
        cameraPosition << configNode["cameraOffset"][1].as<double>(), configNode["cameraOffset"][2].as<double>(),
            configNode["cameraOffset"][3].as<double>();
        Eigen::Quaterniond cameraAttitude;
        cameraAttitude.w() = configNode["cameraOffset"][4].as<double>();
        cameraAttitude.x() = configNode["cameraOffset"][5].as<double>();
        cameraAttitude.y() = configNode["cameraOffset"][6].as<double>();
        cameraAttitude.z() = configNode["cameraOffset"][7].as<double>();

        cameraOffset.R().fromQuaternion(cameraAttitude);
        cameraOffset.x() = cameraPosition;
    }
}