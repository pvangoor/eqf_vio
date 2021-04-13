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

#include "eqf_vio/VIOState.h"
#include <cmath>

using namespace std;
using namespace Eigen;

Matrix3d skew(const Vector3d& vec);

VIOState integrateSystemFunction(const VIOState& state, const IMUVelocity& velocity, const double& dt) {
    VIOState newState;

    // Integrate the pose
    se3vector poseVel = se3vector::Zero();
    poseVel.block<3, 1>(0, 0) = velocity.omega;
    poseVel.block<3, 1>(3, 0) = state.velocity;
    newState.pose = state.pose * SE3::SE3Exp(dt * poseVel);

    // Integrate the velocity
    newState.velocity = state.velocity + dt * (-SO3::skew(velocity.omega) * state.velocity + velocity.accel -
                                                  (state.pose.R().inverse() * Vector3d(0, 0, GRAVITY_CONSTANT)));

    // Landmarks are transformed in the body fixed frame
    const Matrix4d cameraPoseVel =
        state.cameraOffset.inverse().asMatrix() * SE3::wedge(poseVel) * state.cameraOffset.asMatrix();
    const SE3 cameraPoseChangeInv = SE3::SE3Exp(SE3::vee(-dt * cameraPoseVel));
    newState.bodyLandmarks.resize(state.bodyLandmarks.size());
    transform(state.bodyLandmarks.begin(), state.bodyLandmarks.end(), newState.bodyLandmarks.begin(),
        [&cameraPoseChangeInv](const Point3d& blm) {
            Point3d result;
            result.p = cameraPoseChangeInv * blm.p;
            result.id = blm.id;
            return result;
        });

    // Camera offset is constant
    newState.cameraOffset = state.cameraOffset;

    return newState;
}

VisionMeasurement measureSystemState(const VIOManifoldState& state) {
    VisionMeasurement result;
    result.numberOfBearings = state.bodyLandmarks.size();
    result.bearings.resize(state.bodyLandmarks.size());
    transform(
        state.bodyLandmarks.begin(), state.bodyLandmarks.end(), result.bearings.begin(), [](const Point3d& point) {
            Point3d result;
            result.p = point.p.normalized();
            result.id = point.id;
            return result;
        });
    return result;
}

std::ostream& operator<<(std::ostream& os, const VIOState& state) {
    const Vector3d position = state.pose.x();
    const Quaterniond attitude = state.pose.R().asQuaternion();
    os << position.x() << ", " << position.y() << ", " << position.z() << ", ";
    os << attitude.w() << ", " << attitude.x() << ", " << attitude.y() << ", " << attitude.z() << ", ";
    os << state.velocity.x() << ", " << state.velocity.y() << ", " << state.velocity.z() << ", ";

    os << state.bodyLandmarks.size();
    for (const Point3d& blm : state.bodyLandmarks) {
        os << ", " << blm.id << ", " << blm.p.x() << ", " << blm.p.y() << ", " << blm.p.z();
    }
    return os;
}

VIOState::operator VIOManifoldState() const { return projectToManifold(*this); }

VIOManifoldState projectToManifold(const VIOState& Xi) {
    VIOManifoldState xi;
    xi.gravityDir = Xi.pose.R().inverse() * Vector3d(0, 0, 1);
    xi.velocity = Xi.velocity;
    xi.bodyLandmarks = Xi.bodyLandmarks;
    xi.cameraOffset = Xi.cameraOffset;
    return xi;
}

VectorXd euclidCoordinateChart(const VIOManifoldState& xi, const VIOManifoldState& xi0) {
    assert(xi0.bodyLandmarks.size() == xi.bodyLandmarks.size());
    const int N = xi0.bodyLandmarks.size();
    VectorXd eps(5 + 3 * N);

    // eps.block<2, 1>(0, 0) = e3ProjectSphere(xi.gravityDir) - e3ProjectSphere(xi0.gravityDir);
    eps.block<2, 1>(0, 0) = stereoSphereChart(xi.gravityDir, xi0.gravityDir);
    eps.block<3, 1>(2, 0) = xi.velocity - xi0.velocity;
    for (int i = 0; i < N; ++i) {
        assert(xi.bodyLandmarks[i].id == xi0.bodyLandmarks[i].id);
        eps.block<3, 1>(5 + 3 * i, 0) = xi.bodyLandmarks[i].p - xi0.bodyLandmarks[i].p;
    }
    return eps;
}

VIOManifoldState euclidCoordinateChartInv(const VectorXd& eps, const VIOManifoldState& xi0) {
    VIOManifoldState xi;
    // xi.gravityDir = e3ProjectSphereInv(e3ProjectSphere(xi0.gravityDir) + eps.block<2, 1>(0, 0));
    xi.gravityDir = stereoSphereChartInv(eps.block<2, 1>(0, 0), xi0.gravityDir);

    xi.cameraOffset = xi0.cameraOffset;
    xi.velocity = xi0.velocity + eps.block<3, 1>(2, 0);

    assert(5 + 3 * xi0.bodyLandmarks.size() == eps.size());
    const int N = xi0.bodyLandmarks.size();
    xi.bodyLandmarks.resize(N);
    for (int i = 0; i < N; ++i) {
        xi.bodyLandmarks[i].p = xi0.bodyLandmarks[i].p + eps.block<3, 1>(5 + 3 * i, 0);
        xi.bodyLandmarks[i].id = xi0.bodyLandmarks[i].id;
    }
    return xi;
}

Eigen::VectorXd invdepthCoordinateChart(const VIOManifoldState& xi, const VIOManifoldState& xi0) {
    assert(xi0.bodyLandmarks.size() == xi.bodyLandmarks.size());
    const int N = xi0.bodyLandmarks.size();
    VectorXd eps(5 + 3 * N);

    eps.block<2, 1>(0, 0) = stereoSphereChart(xi.gravityDir, xi0.gravityDir);
    eps.block<3, 1>(2, 0) = xi.velocity - xi0.velocity;
    for (int i = 0; i < N; ++i) {
        assert(xi.bodyLandmarks[i].id == xi0.bodyLandmarks[i].id);
        // Compute the inverse depth and bearing
        const double rho_i = 1.0 / xi.bodyLandmarks[i].p.norm();
        const double rho0_i = 1.0 / xi0.bodyLandmarks[i].p.norm();
        const Vector3d y_i = xi.bodyLandmarks[i].p * rho_i;
        const Vector3d y0_i = xi0.bodyLandmarks[i].p * rho0_i;

        // Store the bearing and then the inverse depth
        eps.block<3, 1>(5 + 3 * i, 0) << stereoSphereChart(y_i, y0_i), rho0_i * log(rho_i / rho0_i);
    }
    return eps;
}

std::function<Eigen::VectorXd(const VIOManifoldState& xi)> euclidCoordinateChartAt(const VIOManifoldState& xi0) {
    std::function<Eigen::VectorXd(const VIOManifoldState& xi)> euclidCoordinateChart_xi0 =
        [xi0](const VIOManifoldState& xi) { return euclidCoordinateChart(xi, xi0); };
    return euclidCoordinateChart_xi0;
}

std::function<VIOManifoldState(const Eigen::VectorXd& eps)> euclidCoordinateChartAtInv(const VIOManifoldState& xi0) {
    std::function<VIOManifoldState(const Eigen::VectorXd& eps)> euclidCoordinateChartInv_xi0 =
        [xi0](const Eigen::VectorXd& eps) { return euclidCoordinateChartInv(eps, xi0); };
    return euclidCoordinateChartInv_xi0;
}

VIOManifoldState invdepthCoordinateChartInv(const Eigen::VectorXd& eps, const VIOManifoldState& xi0) {
    VIOManifoldState xi;
    // xi.gravityDir = e3ProjectSphereInv(e3ProjectSphere(xi0.gravityDir) + eps.block<2, 1>(0, 0));
    xi.gravityDir = stereoSphereChartInv(eps.block<2, 1>(0, 0), xi0.gravityDir);

    xi.cameraOffset = xi0.cameraOffset;
    xi.velocity = xi0.velocity + eps.block<3, 1>(2, 0);

    assert(5 + 3 * xi0.bodyLandmarks.size() == eps.size());
    const int N = xi0.bodyLandmarks.size();
    xi.bodyLandmarks.resize(N);
    for (int i = 0; i < N; ++i) {
        const double rho0_i = 1.0 / xi0.bodyLandmarks[i].p.norm();
        const Vector3d y0_i = xi0.bodyLandmarks[i].p * rho0_i;

        // Retrieve bearing and inverse depth
        const Vector3d y_i = stereoSphereChartInv(eps.block<2, 1>(5 + 3 * i, 0), y0_i);
        const double rho_i = exp(eps(5 + 3 * i + 2, 0) / rho0_i) * rho0_i;

        xi.bodyLandmarks[i].p = y_i / rho_i;
        xi.bodyLandmarks[i].id = xi0.bodyLandmarks[i].id;
    }
    return xi;
}

std::function<Eigen::VectorXd(const VIOManifoldState& xi)> invdepthCoordinateChartAt(const VIOManifoldState& xi0) {
    std::function<Eigen::VectorXd(const VIOManifoldState& xi)> invdepthCoordinateChart_xi0 =
        [xi0](const VIOManifoldState& xi) { return invdepthCoordinateChart(xi, xi0); };
    return invdepthCoordinateChart_xi0;
}
std::function<VIOManifoldState(const Eigen::VectorXd& eps)> invdepthCoordinateChartAtInv(const VIOManifoldState& xi0) {
    std::function<VIOManifoldState(const Eigen::VectorXd& eps)> invdepthCoordinateChartInv_xi0 =
        [xi0](const Eigen::VectorXd& eps) { return invdepthCoordinateChartInv(eps, xi0); };
    return invdepthCoordinateChartInv_xi0;
}

Eigen::Vector2d e3ProjectSphere(const Eigen::Vector3d& eta) {
    static const Matrix<double, 2, 3> I23 = Matrix<double, 2, 3>::Identity();
    static const Vector3d e3 = Vector3d(0, 0, 1);
    const Vector2d y = I23 * (eta - e3) / (1 - e3.dot(eta));
    return y;
}

Eigen::Vector3d e3ProjectSphereInv(const Eigen::Vector2d& y) {
    static const Vector3d e3 = Vector3d(0, 0, 1);
    const Vector3d yBar = (Vector3d() << y, 0).finished();
    const Vector3d eta = e3 + 2.0 / (yBar.squaredNorm() + 1) * (yBar - e3);
    return eta;
}

Eigen::Matrix<double, 2, 3> e3ProjectSphereDiff(const Eigen::Vector3d& eta) {
    static const Matrix<double, 2, 3> I23 = Matrix<double, 2, 3>::Identity();
    static const Vector3d e3 = Vector3d(0, 0, 1);
    Eigen::Matrix<double, 2, 3> Diff;
    Diff = I23 * (Matrix3d::Identity() * (1 - eta.z()) + (eta - e3) * e3.transpose());
    Diff = pow((1 - e3.dot(eta)), -2.0) * Diff;
    return Diff;
}

Eigen::Matrix<double, 3, 2> e3ProjectSphereInvDiff(const Eigen::Vector2d& y) {
    Eigen::Matrix<double, 3, 2> Diff;
    Diff.block<2, 2>(0, 0) = Matrix2d::Identity() * (y.squaredNorm() + 1.0) - 2 * y * y.transpose();
    Diff.block<1, 2>(2, 0) = 2 * y.transpose();
    Diff = 2.0 * pow((y.squaredNorm() + 1.0), -2.0) * Diff;
    return Diff;
}

Eigen::Vector2d stereoSphereChart(const Eigen::Vector3d& eta, const Eigen::Vector3d& pole) {
    const SO3 sphereRot = SO3::SO3FromVectors(-pole, Eigen::Vector3d::Unit(2));
    const Vector3d etaRotated = sphereRot * eta;
    return e3ProjectSphere(etaRotated);
}

Eigen::Vector3d stereoSphereChartInv(const Eigen::Vector2d& y, const Eigen::Vector3d& pole) {
    const Vector3d etaRotated = e3ProjectSphereInv(y);
    const SO3 sphereRot = SO3::SO3FromVectors(-pole, Eigen::Vector3d::Unit(2));
    return sphereRot.inverse() * etaRotated;
}

Eigen::Matrix<double, 2, 3> stereoSphereChartDiff(const Eigen::Vector3d& eta, const Eigen::Vector3d& pole) {
    const SO3 sphereRot = SO3::SO3FromVectors(-pole, Eigen::Vector3d::Unit(2));
    const Vector3d etaRotated = sphereRot * eta;
    return e3ProjectSphereDiff(etaRotated) * sphereRot.asMatrix();
}

Eigen::Matrix<double, 3, 2> stereoSphereChartInvDiff(const Eigen::Vector2d& y, const Eigen::Vector3d& pole) {
    const SO3 sphereRot = SO3::SO3FromVectors(-pole, Eigen::Vector3d::Unit(2));
    return sphereRot.inverse().asMatrix() * e3ProjectSphereInvDiff(y);
}