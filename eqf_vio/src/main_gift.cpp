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

#include "eqf_vio/CSVReader.h"
#include "eqf_vio/IMUVelocity.h"
#include "eqf_vio/VIOFilter.h"
#include "eqf_vio/VIOFilterSettings.h"
#include "eqf_vio/VisionMeasurement.h"

#include "GIFT/PointFeatureTracker.h"
#include "GIFT/Visualisation.h"
#include "opencv2/highgui/highgui.hpp"

#include "yaml-cpp/yaml.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <exception>

#if BUILD_VISUALISATION
#include "Plotter.h"
#endif

IMUVelocity readIMUData(const CSVLine& row);
double readVideoStamp(const CSVLine& row);
VisionMeasurement convertGIFTFeatures(const std::vector<GIFT::Feature>& GIFTFeatures, const double& stamp);

int main(int argc, char const* argv[]) {

    // Read file names
    if (argc != 4 && argc != 5) {
        std::cout << "Not enough arguments were provided.\n";
        std::cout << "Usage: eqf_vio_gift IMU_file video_file video_stamps (config_file)." << std::endl;
        return 1;
    }
    std::string IMUFileName(argv[1]), VideoFileName(argv[2]), VideoStampsFileName(argv[3]),
        ConfigFileName(DEFAULT_CONFIG_FILE);
    if (argc == 5) {
        ConfigFileName = std::string(argv[4]);
    }

    // Set up the data iterator for IMU and video and video stamps
    std::ifstream IMUFile = std::ifstream(IMUFileName);
    if (!IMUFile.good()) {
            std::stringstream ess;
        ess << "Could not open the IMU data: " << IMUFileName;
        throw std::runtime_error(ess.str());
    }
    CSVReader IMUFileIter(IMUFile);
    ++IMUFileIter; // skip the header
    IMUVelocity imuData = readIMUData(*IMUFileIter);

    if (!std::ifstream(VideoFileName).good()) {
        std::stringstream ess;
        ess << "Could not open the video file: " << VideoFileName;
        throw std::runtime_error(ess.str());
    }

    cv::VideoCapture cap(VideoFileName);
    cv::Mat currentFrame;

    std::ifstream VideoStampsFile = std::ifstream(VideoStampsFileName);
        if (!VideoStampsFile.good()) {
        std::stringstream ess;
        ess << "Could not open the video stamps file: " << VideoStampsFileName;
        throw std::runtime_error(ess.str());
    }
    CSVReader VideoStampsIter(VideoStampsFile);
    ++VideoStampsIter; // skip the header
    double videoStamp = readVideoStamp(*VideoStampsIter);

    // Read I/O settings
    if (!std::ifstream(ConfigFileName).good()) {
        std::stringstream ess;
        ess << "Could not open the configuration file: " << ConfigFileName;
        throw std::runtime_error(ess.str());
    }
    const YAML::Node eqf_vioConfig = YAML::LoadFile(ConfigFileName);
    const double startTime = eqf_vioConfig["main"]["startTime"].as<double>();
    const bool writeStateFlag = eqf_vioConfig["main"]["writeState"].as<bool>();
    const bool writeFilterFlag = eqf_vioConfig["main"]["writeFilter"].as<bool>();

    const bool showVisualisationFlag = eqf_vioConfig["main"]["showVisualisation"].as<bool>();
    if (showVisualisationFlag && !BUILD_VISUALISATION) {
        std::cout << "Visualisations have been requested, but the necessary module is not set to be built."
                  << std::endl;
    }
    const double limitRateSetting = eqf_vioConfig["main"]["limitRate"].as<double>();
    std::chrono::steady_clock::time_point loopTimer = std::chrono::steady_clock::now();

#if BUILD_VISUALISATION
    std::unique_ptr<Plotter> plotter;
    if (showVisualisationFlag)
        plotter = std::make_unique<Plotter>();
#endif

    // Initialise the filter
    VIOFilter::Settings filterSettings(eqf_vioConfig["eqf"]);
    VIOFilter filter(filterSettings);

    // Set up the feature tracker
    const std::string cameraIntrinsicsFname = eqf_vioConfig["GIFT"]["intrinsicsFile"].as<std::string>();
    if (!std::ifstream(cameraIntrinsicsFname).good()) {
        std::stringstream ess;
        ess << "Could not open the GIFT camera intrinsics: " << cameraIntrinsicsFname;
        throw std::runtime_error(ess.str());
    }
    GIFT::PinholeCamera camera = GIFT::PinholeCamera(cv::String(cameraIntrinsicsFname));
    GIFT::PointFeatureTracker featureTracker = GIFT::PointFeatureTracker(camera);
    safeConfig(eqf_vioConfig["GIFT"]["maxFeatures"], featureTracker.maxFeatures);
    safeConfig(eqf_vioConfig["GIFT"]["featureDist"], featureTracker.featureDist);
    safeConfig(eqf_vioConfig["GIFT"]["minHarrisQuality"], featureTracker.minHarrisQuality);
    safeConfig(eqf_vioConfig["GIFT"]["featureSearchThreshold"], featureTracker.featureSearchThreshold);
    safeConfig(eqf_vioConfig["GIFT"]["maxError"], featureTracker.maxError);
    safeConfig(eqf_vioConfig["GIFT"]["winSize"], featureTracker.winSize);
    safeConfig(eqf_vioConfig["GIFT"]["maxLevel"], featureTracker.maxLevel);

    // Set up output files
    std::time_t t0 = std::time(nullptr);
    std::stringstream outputFileNameStream, internalFileNameStream;
    outputFileNameStream << "EQF_VIO_output_" << std::put_time(std::localtime(&t0), "%F_%T") << ".csv";
    internalFileNameStream << "EQF_VIO_internal_" << std::put_time(std::localtime(&t0), "%F_%T") << ".csv";
    std::ofstream internalFile, outputFile;
    if (writeStateFlag) {
        outputFile = std::ofstream(outputFileNameStream.str());
        outputFile << "time, tx, ty, tz, qw, qx, qy, qz, vx, vy, vz, N, "
                   << "p1id, p1x, p1y, p1z, ..., ..., ..., ..., pNid, pNx, pNy, pNz" << std::endl;
    }
    if (writeFilterFlag) {
        internalFile = std::ofstream(internalFileNameStream.str());
        internalFile
            << "time, t0x, t0y, t0z, q0w, q0x, q0y, q0z, v0x, v0y, v0z, tAx, tAy, tAz, qAw, qAx, qAy, qAz, wx, wy, wz, "
               "N, "
               "p1id, p1x, p1y, p1z, qQ1w, qQ1x, qQ1y, qQ1z, aQ1, ..., ..., ..., ..., ..., ..., ..., ..., ..., "
               "pNid, pNx, pNy, pNz, qQNw, qQNx, qQNy, qQNz, aQN, Sigma(1,1), Sigma(1,2), ..., Sigma(5+3N, 5+3N)"
            << std::endl;
    }

    // Read in all the data
    int imuDataCounter = 0, visionDataCounter = 0;
    std::chrono::steady_clock::time_point loopStartTime = std::chrono::steady_clock::now();
    while (true) {
        // Treat data in turns
        if (imuData.stamp < videoStamp) {
            // Pass IMU data to the filter
            if (imuData.stamp > startTime) {
                filter.processIMUData(imuData);
                ++imuDataCounter;
            }

            // Increment the iterator
            ++IMUFileIter;
            if (IMUFileIter == CSVReader())
                break;
            imuData = readIMUData(*IMUFileIter);

        } else {
            // Pass measurement data to the filter
            if (!cap.read(currentFrame))
                break;

            if (videoStamp > startTime) {
                // Run GIFT
                featureTracker.processImage(currentFrame);
                const std::vector<GIFT::Feature> features = featureTracker.outputFeatures();
                const VisionMeasurement measData = convertGIFTFeatures(features, videoStamp);

                filter.processVisionData(measData);
                ++visionDataCounter;

                // Output filter data
                VIOState estimatedState = filter.stateEstimate();
                if (writeStateFlag)
                    outputFile << std::setprecision(20) << filter.getTime() << std::setprecision(5) << ", "
                               << estimatedState << std::endl;
                if (writeFilterFlag)
                    internalFile << std::setprecision(20) << filter.getTime() << std::setprecision(5) << ", " << filter
                                 << std::endl;

                // Optionally visualise the filter data
                if (showVisualisationFlag) {
#if BUILD_VISUALISATION
                    std::vector<Eigen::Vector3d> inertialPoints(estimatedState.bodyLandmarks.size());
                    std::transform(estimatedState.bodyLandmarks.begin(), estimatedState.bodyLandmarks.end(),
                        inertialPoints.begin(), [&estimatedState](const Point3d& pt) {
                            return estimatedState.pose * estimatedState.cameraOffset * pt.p;
                        });
                    plotter->drawPoints(inertialPoints, Eigen::Vector4d(0, 0, 1, 1), 5);
                    plotter->hold = true;
                    plotter->drawAxes(estimatedState.pose.asMatrix(), 1.0, 5);
                    plotter->hold = false;
                    plotter->lockOrigin(estimatedState.pose.x());
#endif
                    const cv::Mat featureImage = GIFT::drawFeatureImage(currentFrame, featureTracker.outputFeatures());
                    // const cv::Mat featureImage = featureTracker.drawFeatureImage(cv::Scalar(0, 255, 0), 3, 2);
                    cv::imshow("Features", featureImage);
                    cv::waitKey(1);
                }

                // Limit the loop rate
                if (limitRateSetting > 0) {
                    std::this_thread::sleep_until(loopTimer + std::chrono::duration<double>(1.0 / limitRateSetting));
                    loopTimer = std::chrono::steady_clock::now();
                }
            }

            // Increment the iterator
            ++VideoStampsIter;
            if (VideoStampsIter == CSVReader())
                break;
            videoStamp = readVideoStamp(*VideoStampsIter);
        }
    }

    const auto elapsedTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - loopStartTime);
    std::cout << "Processed " << imuDataCounter << " IMU and " << visionDataCounter << " vision measurements.\n"
              << "Time taken: " << elapsedTime.count() * 1e-3 << " seconds." << std::endl;

#if BUILD_VISUALISATION
    if (showVisualisationFlag) {
        plotter->unlockOrigin();
        plotter->maintain();
    }
#endif
    return 0;
}

IMUVelocity readIMUData(const CSVLine& row) {
    IMUVelocity imuData;
    if (row.size() < 7) {
        throw std::length_error("Each line of IMU data must contain at least 7 entries.");
    }
    imuData.stamp = stod(row[0]);
    imuData.omega << stod(row[1]), stod(row[2]), stod(row[3]);
    imuData.accel << stod(row[4]), stod(row[5]), stod(row[6]);
    return imuData;
}

double readVideoStamp(const CSVLine& row) { 
    if (row.size() < 1) {
        throw std::length_error("Each line of video stamp data must contain at least 1 entry.");
    }
    return stod(row[0]); }

VisionMeasurement convertGIFTFeatures(const std::vector<GIFT::Feature>& GIFTFeatures, const double& stamp) {
    VisionMeasurement measurement;
    measurement.stamp = stamp;
    measurement.numberOfBearings = GIFTFeatures.size();
    measurement.bearings.resize(GIFTFeatures.size());
    std::transform(GIFTFeatures.begin(), GIFTFeatures.end(), measurement.bearings.begin(), [](const GIFT::Feature& f) {
        Point3d pt;
        pt.p = f.sphereCoordinates();
        pt.id = f.idNumber;
        return pt;
    });
    return measurement;
}