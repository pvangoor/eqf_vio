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

#include "eigen3/Eigen/Dense"
#include "eqf_vio/VIOFilter.h"
#include "testing_utilities.h"
#include "gtest/gtest.h"

using namespace Eigen;
using namespace std;

TEST(CoordinateChartTest, SphereChartE3) {
    // Test the sphere coordinate charts
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const Vector3d eta = Vector3d::Random().normalized();
        const Vector2d y1 = e3ProjectSphere(eta);
        const Vector3d eta1 = e3ProjectSphereInv(y1);
        const double dist_eta1 = (eta - eta1).norm();
        EXPECT_LE(dist_eta1, NEAR_ZERO);

        const Vector2d y = Vector2d::Random();
        const Vector3d eta2 = e3ProjectSphereInv(y);
        const Vector2d y2 = e3ProjectSphere(eta2);
        const double dist_y2 = (y2 - y).norm();
        EXPECT_LE(dist_y2, NEAR_ZERO);
    }
}

TEST(CoordinateChartTest, SphereChartPole) {
    // Test the sphere coordinate charts with poles
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const Vector3d pole = Vector3d::Random().normalized();

        const Vector2d poleCoords = stereoSphereChart(pole, pole);
        EXPECT_LE(poleCoords.norm(), NEAR_ZERO);

        const Vector3d eta = Vector3d::Random().normalized();
        const Vector2d y1 = stereoSphereChart(eta, pole);
        const Vector3d eta1 = stereoSphereChartInv(y1, pole);
        const double dist_eta1 = (eta - eta1).norm();
        EXPECT_LE(dist_eta1, NEAR_ZERO);

        const Vector2d y = Vector2d::Random();
        const Vector3d eta2 = stereoSphereChartInv(y, pole);
        const Vector2d y2 = stereoSphereChart(eta2, pole);
        const double dist_y2 = (y2 - y).norm();
        EXPECT_LE(dist_y2, NEAR_ZERO);
    }
}

TEST(CoordinateChartTest, SphereChartE3Differential) {
    // Test the sphere chart differential
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const Vector3d eta = Vector3d::Random().normalized();
        const Vector2d y = e3ProjectSphere(eta);
        // Create a tangent vector at eta
        const Vector3d eta_offset = (Matrix3d::Identity() - eta * eta.transpose()) * Vector3d::Random();
        const Vector2d computedTangent = e3ProjectSphereDiff(eta) * eta_offset;
        double previousDist = 1e8;
        for (int i = 1; i < 7; ++i) {
            const double dt = pow(10.0, -i);
            const Vector3d eta1 = (eta + dt * eta_offset).normalized();
            const Vector2d trueTangent = (e3ProjectSphere(eta1) - y) / dt;

            double tangentDist = (trueTangent - computedTangent).norm();
            EXPECT_LE(tangentDist, previousDist);
            previousDist = tangentDist;
        }
    }

    // Test the sphere chart inverse differential
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const Vector2d y = Vector2d::Random();
        const Vector3d eta = e3ProjectSphereInv(y);
        const Matrix3d etaProjector = (Matrix3d::Identity() - eta * eta.transpose());

        // Create a tangent vector at y
        const Vector2d y_offset = Vector2d::Random();
        const Vector3d computedTangent = e3ProjectSphereInvDiff(y) * y_offset;
        double previousDist = 1e8;
        for (int i = 1; i < 7; ++i) {
            const double dt = pow(10.0, -i);
            const Vector2d y1 = y + dt * y_offset;
            const Vector3d trueTangent = etaProjector * (e3ProjectSphereInv(y1) - eta) / dt;

            double tangentDist = (trueTangent - computedTangent).norm();
            EXPECT_LE(tangentDist, previousDist);
            previousDist = tangentDist;
        }
    }
}

TEST(CoordinateChartTest, SphereChartPoleDifferential) {
    // Test the sphere chart differential with poles
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const Vector3d pole = Vector3d::Random().normalized();

        const Vector3d eta = Vector3d::Random().normalized();
        const Vector2d y = stereoSphereChart(eta, pole);
        // Create a tangent vector at eta
        const Vector3d eta_offset = (Matrix3d::Identity() - eta * eta.transpose()) * Vector3d::Random();
        const Vector2d computedTangent = stereoSphereChartDiff(eta, pole) * eta_offset;
        double previousDist = 1e8;
        for (int i = 1; i < 7; ++i) {
            const double dt = pow(10.0, -i);
            const Vector3d eta1 = (eta + dt * eta_offset).normalized();
            const Vector2d trueTangent = (stereoSphereChart(eta1, pole) - y) / dt;

            double tangentDist = (trueTangent - computedTangent).norm();
            EXPECT_LE(tangentDist, previousDist);
            previousDist = tangentDist;
        }
    }

    // Test the sphere chart inverse differential
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const Vector3d pole = Vector3d::Random().normalized();

        const Vector2d y = Vector2d::Random();
        const Vector3d eta = stereoSphereChartInv(y, pole);
        const Matrix3d etaProjector = (Matrix3d::Identity() - eta * eta.transpose());

        // Create a tangent vector at y
        const Vector2d y_offset = Vector2d::Random();
        const Vector3d computedTangent = stereoSphereChartInvDiff(y, pole) * y_offset;
        double previousDist = 1e8;
        for (int i = 1; i < 7; ++i) {
            const double dt = pow(10.0, -i);
            const Vector2d y1 = y + dt * y_offset;
            const Vector3d trueTangent = etaProjector * (stereoSphereChartInv(y1, pole) - eta) / dt;

            double tangentDist = (trueTangent - computedTangent).norm();
            EXPECT_LE(tangentDist, previousDist);
            previousDist = tangentDist;
        }
    }
}

TEST(CoordinateChartTest, VIOChart_euclid) {
    vector<int> ids = {0, 1, 2, 3, 4};
    // Test the VIO manifold coordinate chart
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOManifoldState xi0 = projectToManifold(randomStateElement(ids));
        const VIOManifoldState xi = projectToManifold(randomStateElement(ids));
        const VectorXd eps = euclidCoordinateChart(xi, xi0);
        const VIOManifoldState xi1 = euclidCoordinateChartInv(eps, xi0);

        double dist1 = stateDistance(xi, xi1);
        EXPECT_LE(dist1, NEAR_ZERO);
    }

    // Test the functional VIO manifold coordinate chart
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOManifoldState xi0 = projectToManifold(randomStateElement(ids));
        const VIOManifoldState xi = projectToManifold(randomStateElement(ids));
        const auto epsMap = euclidCoordinateChartAt(xi0);
        const auto epsMapInv = euclidCoordinateChartAtInv(xi0);

        const VIOManifoldState xi1 = epsMapInv(epsMap(xi));

        double dist1 = stateDistance(xi, xi1);
        EXPECT_LE(dist1, NEAR_ZERO);
    }
}

TEST(CoordinateChartTest, VIOChart_invdepth) {
    vector<int> ids = {0, 1, 2, 3, 4};
    // Test the VIO manifold coordinate chart
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOManifoldState xi0 = projectToManifold(randomStateElement(ids));
        const VIOManifoldState xi = projectToManifold(randomStateElement(ids));
        const VectorXd eps = invdepthCoordinateChart(xi, xi0);
        const VIOManifoldState xi1 = invdepthCoordinateChartInv(eps, xi0);

        double dist1 = stateDistance(xi, xi1);
        EXPECT_LE(dist1, NEAR_ZERO);
    }

    // Test the functional VIO manifold coordinate chart
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOManifoldState xi0 = projectToManifold(randomStateElement(ids));
        const VIOManifoldState xi = projectToManifold(randomStateElement(ids));
        const auto epsMap = invdepthCoordinateChartAt(xi0);
        const auto epsMapInv = invdepthCoordinateChartAtInv(xi0);

        const VIOManifoldState xi1 = epsMapInv(epsMap(xi));

        double dist1 = stateDistance(xi, xi1);
        EXPECT_LE(dist1, NEAR_ZERO);
    }
}

TEST(CoordinateChartTest, VisionChart) {
    const vector<int> ids = {0, 1, 2, 3, 4};
    const int N = ids.size();
    // Test the VIO output coordinate chart
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VisionMeasurement y0 = randomVisionMeasurement(ids);
        const VectorXd delta = VectorXd::Random(2 * N);
        const VisionMeasurement y = outputCoordinateChartInv(delta, y0);
        const VectorXd delta1 = outputCoordinateChart(y, y0);

        const double dist = (delta - delta1).norm();
        EXPECT_LE(dist, NEAR_ZERO);
    }
}
