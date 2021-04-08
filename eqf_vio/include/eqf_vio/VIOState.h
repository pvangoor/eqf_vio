#pragma once

#include "SE3.h"

#include "eigen3/Eigen/Eigen"
#include "eqf_vio/IMUVelocity.h"
#include "eqf_vio/VisionMeasurement.h"

#include <functional>
#include <memory>
#include <ostream>
#include <vector>

template <class Manifold> struct TangentBundle {
    TangentBundle(const Manifold& xi) : basePoint(xi) {}
    TangentBundle(const Manifold& xi, const Eigen::VectorXd& vec) : basePoint(xi), tVector(vec) {}
    Eigen::VectorXd tVector;
    const Manifold& basePoint;
};

struct Point3d {
    Eigen::Vector3d p;
    int id = -1;
};

struct VIOManifoldState {
    Eigen::Vector3d gravityDir;
    Eigen::Vector3d velocity;
    std::vector<Point3d> bodyLandmarks;

    SE3 cameraOffset;
};

struct VIOState {
    SE3 pose;
    Eigen::Vector3d velocity;
    std::vector<Point3d> bodyLandmarks;

    SE3 cameraOffset;
    operator VIOManifoldState() const;

    typedef TangentBundle<VIOState> Tangent;
};

VIOManifoldState projectToManifold(const VIOState& Xi);

Eigen::VectorXd euclidCoordinateChart(const VIOManifoldState& xi, const VIOManifoldState& xi0);
VIOManifoldState euclidCoordinateChartInv(const Eigen::VectorXd& eps, const VIOManifoldState& xi0);
std::function<Eigen::VectorXd(const VIOManifoldState& xi)> euclidCoordinateChartAt(const VIOManifoldState& xi0);
std::function<VIOManifoldState(const Eigen::VectorXd& eps)> euclidCoordinateChartAtInv(const VIOManifoldState& xi0);

Eigen::VectorXd invdepthCoordinateChart(const VIOManifoldState& xi, const VIOManifoldState& xi0);
VIOManifoldState invdepthCoordinateChartInv(const Eigen::VectorXd& eps, const VIOManifoldState& xi0);
std::function<Eigen::VectorXd(const VIOManifoldState& xi)> invdepthCoordinateChartAt(const VIOManifoldState& xi0);
std::function<VIOManifoldState(const Eigen::VectorXd& eps)> invdepthCoordinateChartAtInv(const VIOManifoldState& xi0);

Eigen::Vector2d e3ProjectSphere(const Eigen::Vector3d& eta);
Eigen::Vector3d e3ProjectSphereInv(const Eigen::Vector2d& y);
Eigen::Matrix<double, 2, 3> e3ProjectSphereDiff(const Eigen::Vector3d& eta);
Eigen::Matrix<double, 3, 2> e3ProjectSphereInvDiff(const Eigen::Vector2d& y);

Eigen::Vector2d stereoSphereChart(const Eigen::Vector3d& eta, const Eigen::Vector3d& pole);
Eigen::Vector3d stereoSphereChartInv(const Eigen::Vector2d& y, const Eigen::Vector3d& pole);
Eigen::Matrix<double, 2, 3> stereoSphereChartDiff(const Eigen::Vector3d& eta, const Eigen::Vector3d& pole);
Eigen::Matrix<double, 3, 2> stereoSphereChartInvDiff(const Eigen::Vector2d& y, const Eigen::Vector3d& pole);

std::ostream& operator<<(std::ostream& os, const VIOState& state);

[[nodiscard]] VIOState integrateSystemFunction(const VIOState& state, const IMUVelocity& velocity, const double& dt);
[[nodiscard]] VisionMeasurement measureSystemState(const VIOManifoldState& state);