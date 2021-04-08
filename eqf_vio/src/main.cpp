#include "eqf_vio/CSVReader.h"
#include "eqf_vio/IMUVelocity.h"
#include "eqf_vio/VIOFilter.h"
#include "eqf_vio/VIOFilterSettings.h"
#include "eqf_vio/VisionMeasurement.h"

#include "yaml-cpp/yaml.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#if BUILD_VISUALISATION
#include "Plotter.h"
#endif

IMUVelocity readIMUData(const CSVLine& row);
VisionMeasurement readMeasData(const CSVLine& row);

int main(int argc, char const* argv[]) {

    // Read file names
    if (argc != 3 && argc != 4) {
        std::cout << "Not enough arguments were provided.\n";
        std::cout << "Usage: eqf_vio IMU_file meas_file (config_file)." << std::endl;
        return 1;
    }
    std::string IMUFileName(argv[1]), MeasFileName(argv[2]), ConfigFileName(DEFAULT_CONFIG_FILE);
    if (argc == 4) {
        ConfigFileName = std::string(argv[3]);
    }

    // Set up the data iterators for IMU and measurements
    std::ifstream IMUFile = std::ifstream(IMUFileName);
    CSVReader IMUFileIter(IMUFile);
    ++IMUFileIter; // skip the header
    IMUVelocity imuData = readIMUData(*IMUFileIter);

    std::ifstream MeasFile = std::ifstream(MeasFileName);
    CSVReader MeasFileIter(MeasFile);
    ++MeasFileIter; // skip the header
    VisionMeasurement measData = readMeasData(*MeasFileIter);

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
    Plotter plotter;
#endif

    // Initialise the filter
    VIOFilter::Settings filterSettings(eqf_vioConfig["eqf"]);
    VIOFilter filter(filterSettings);

    // Set up output files
    std::time_t t = std::time(nullptr);
    std::stringstream outputFileNameStream, internalFileNameStream;
    outputFileNameStream << "EQF_VIO_output_" << std::put_time(std::localtime(&t), "%F_%T") << ".csv";
    internalFileNameStream << "EQF_VIO_internal_" << std::put_time(std::localtime(&t), "%F_%T") << ".csv";
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
    while (true) {
        // Treat data in turns
        if (imuData.stamp < measData.stamp) {
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
            if (measData.stamp > startTime) {
                filter.processVisionData(measData);
                ++visionDataCounter;
            }

            // Output filter data
            VIOState estimatedState = filter.stateEstimate();
            if (writeStateFlag)
                outputFile << std::setprecision(20) << filter.getTime() << std::setprecision(5) << ", "
                           << estimatedState << std::endl;
            if (writeFilterFlag)
                internalFile << std::setprecision(20) << filter.getTime() << std::setprecision(5) << ", " << filter
                             << std::endl;

#if BUILD_VISUALISATION
            // Optionally visualise the filter data
            if (showVisualisationFlag) {
                std::vector<Eigen::Vector3d> inertialPoints(estimatedState.bodyLandmarks.size());
                std::transform(estimatedState.bodyLandmarks.begin(), estimatedState.bodyLandmarks.end(),
                    inertialPoints.begin(), [&estimatedState](const Point3d& pt) {
                        return estimatedState.pose * estimatedState.cameraOffset * pt.p;
                    });
                plotter.drawPoints(inertialPoints, Eigen::Vector4d(0, 0, 1, 1), 5);
                plotter.hold = true;
                plotter.drawAxes(estimatedState.pose.asMatrix(), 1.0, 5);
                plotter.hold = false;
                plotter.lockOrigin(estimatedState.pose.x());
            }
#endif

            // Increment the iterator
            ++MeasFileIter;
            if (MeasFileIter == CSVReader())
                break;
            measData = readMeasData(*MeasFileIter);

            // Limit the loop rate
            if (limitRateSetting > 0) {
                std::this_thread::sleep_until(loopTimer + std::chrono::duration<double>(1.0 / limitRateSetting));
                loopTimer = std::chrono::steady_clock::now();
            }
        }
    }

    std::cout << "Processed " << imuDataCounter << " IMU and " << visionDataCounter << " vision measurements."
              << std::endl;

#if BUILD_VISUALISATION
    if (showVisualisationFlag) {
        plotter.unlockOrigin();
        plotter.maintain();
    }
#endif
    return 0;
}

IMUVelocity readIMUData(const CSVLine& row) {
    IMUVelocity imuData;
    imuData.stamp = stod(row[0]);
    imuData.omega << stod(row[1]), stod(row[2]), stod(row[3]);
    imuData.accel << stod(row[4]), stod(row[5]), stod(row[6]);
    return imuData;
}

VisionMeasurement readMeasData(const CSVLine& row) {
    VisionMeasurement measData;
    measData.stamp = stod(row[0]);
    measData.numberOfBearings = stoi(row[1]);
    measData.bearings.resize(measData.numberOfBearings);
    for (int i = 0; i < measData.numberOfBearings; ++i) {
        int j = 2 + 4 * i;
        measData.bearings[i].id = stoi(row[j]);
        measData.bearings[i].p << stod(row[j + 1]), stod(row[j + 2]), stod(row[j + 3]);
    }
    return measData;
}