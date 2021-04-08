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