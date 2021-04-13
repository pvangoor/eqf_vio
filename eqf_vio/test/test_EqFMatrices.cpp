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
#include "eqf_vio/EqFMatrices.h"
#include "testing_utilities.h"
#include "gtest/gtest.h"

using namespace Eigen;
using namespace std;

using Vector6d = Matrix<double, 6, 1>;

TEST(EqFMatricesTest, StateMatrixA) {
    vector<int> ids = {0, 1, 2, 3, 4};
    const int N = ids.size();

    // Set some random conditions
    const VIOGroup X_hat = randomGroupElement(ids);
    const VIOManifoldState xi0 = projectToManifold(randomStateElement(ids));
    const IMUVelocity vel = randomVelocityElement();
    const MatrixXd A0t = EqFStateMatrixA_euclid(X_hat, xi0, vel);

    // Compare the A matrix to the lemma from which it is derived
    // The lemma says
    // A0 = Deps . Dphi_{X^-1} . Dphi_{xi} . DLambda_v . Dphi_X . Deps^{-1} (0)
    // Let a0(delta) = eps o phi_{X^-1} o phi_xi o exp o LambdaTilde_v o phi_X o eps^{-1} (delta)
    // Then we expect that
    //  (a0(dt * epsilon) - a0(0)) / dt --> A0 epsilon
    // as dt --> 0, for any vector epsilon

    auto a0 = [&](const VectorXd& epsilon) {
        const auto xi_hat = stateGroupAction(X_hat, xi0);
        const auto xi_e = EqFStateMatrixA_euclid.coordinateChartInv(epsilon, xi0);
        const auto xi = stateGroupAction(X_hat, xi_e);
        const auto LambdaTilde = liftVelocity(xi, vel) - liftVelocity(xi_hat, vel);
        const auto xi_hat1 = stateGroupAction(VIOExp(LambdaTilde), xi_hat);
        const auto xi_e1 = stateGroupAction(X_hat.inverse(), xi_hat1);
        const VectorXd epsilon1 = EqFStateMatrixA_euclid.coordinateChart(xi_e1, xi0);
        return epsilon1;
    };

    // Check the function at zero
    const VectorXd a0AtZero = a0(VectorXd::Zero(5 + 3 * N));
    EXPECT_LE(a0AtZero.norm(), NEAR_ZERO);

    // Check individual directions of the differential
    for (int j = 0; j < 5 + 3 * N; ++j) {
        const VectorXd ej = VectorXd::Unit(5 + 3 * N, j);
        const VectorXd computedDiff = A0t * ej;

        double previousDist = 1e8;
        for (int i = 1; i < 8; ++i) {
            const double dt = pow(10.0, -i);
            const VectorXd trueDiff = a0(dt * ej) / dt;

            double dist = (trueDiff - computedDiff).norm();
            EXPECT_LE(dist, previousDist);
            previousDist = dist;
        }
    }

    // Check some random directions
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VectorXd epsVec = VectorXd::Random(5 + 3 * N);
        const VectorXd computedEpsDot = A0t * epsVec;
        double previousDist = 1e8;
        for (int i = 1; i < 8; ++i) {
            const double dt = pow(10.0, -i);
            const VectorXd trueEpsDot = a0(dt * epsVec) / dt;

            double dist = (trueEpsDot - computedEpsDot).norm();
            EXPECT_LE(dist, previousDist);
            previousDist = dist;
        }
    }
}

TEST(EqFMatricesTest, InputMatrixB) {
    vector<int> ids = {0, 1, 2, 3, 4};
    const int N = ids.size();

    // Set some random conditions
    const VIOGroup X_hat = randomGroupElement(ids);
    const VIOManifoldState xi0 = projectToManifold(randomStateElement(ids));
    const MatrixXd Bt = EqFInputMatrixB_euclid(X_hat, xi0);
    const IMUVelocity vel = randomVelocityElement();

    // Compare the A matrix to the lemma from which it is derived
    // The lemma says
    // A0 = Deps . Dphi_{X^-1} . Dphi_{xi} . DLambda_v . Dphi_X . Deps^{-1} (0)
    // Let a0(delta) = eps o phi_{X^-1} o phi_xi o exp o LambdaTilde_v o phi_X o eps^{-1} (delta)
    // Then we expect that
    //  (a0(dt * epsilon) - a0(0)) / dt --> A0 epsilon
    // as dt --> 0, for any vector epsilon

    auto b0 = [&](const Vector6d& vel_err_vec) {
        const auto xi_hat = stateGroupAction(X_hat, xi0);
        const auto vel_err = IMUVelocity(vel_err_vec);
        const auto LambdaTilde = liftVelocity(xi_hat, vel + vel_err) - liftVelocity(xi_hat, vel);
        const auto xi_hat1 = stateGroupAction(VIOExp(LambdaTilde), xi_hat);
        const auto xi_e1 = stateGroupAction(X_hat.inverse(), xi_hat1);
        const VectorXd epsilon = euclidCoordinateChart(xi_e1, xi0);
        return epsilon;
    };

    // Check the function at zero
    const VectorXd b0AtZero = b0(Vector6d::Zero());
    EXPECT_LE(b0AtZero.norm(), NEAR_ZERO);

    // Check individual directions of the differential
    for (int j = 0; j < 6; ++j) {
        const Vector6d ej = Vector6d::Unit(j);
        const VectorXd computedDiff = Bt * ej;

        double previousDist = 1e8;
        for (int i = 1; i < 6; ++i) {
            const double dt = pow(10.0, -i);
            const VectorXd trueDiff = b0(dt * ej) / dt;

            double dist = (trueDiff - computedDiff).norm();
            if (dist > 1e-8)
                EXPECT_LE(dist, previousDist);
            previousDist = dist;
        }
    }

    // Check some random directions
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const Vector6d epsVec = Vector6d::Random();
        const VectorXd computedEpsDot = Bt * epsVec;
        double previousDist = 1e8;
        for (int i = 1; i < 6; ++i) {
            const double dt = pow(10.0, -i);
            const VectorXd trueEpsDot = b0(dt * epsVec) / dt;

            double dist = (trueEpsDot - computedEpsDot).norm();
            if (dist > 1e-8)
                EXPECT_LE(dist, previousDist);
            previousDist = dist;
        }
    }
}

TEST(EqFMatricesTest, OutputMatrixC) {
    vector<int> ids = {0, 1, 2, 3, 4};
    const int N = ids.size();

    // Set some random conditions
    const VIOManifoldState xi0 = projectToManifold(randomStateElement(ids));
    const MatrixXd C0 = EqFOutputMatrixC_euclid(xi0);

    // Compare the C matrix to the expression from which it is derived
    // C0 = Ddelta(y0) . Dh(xi0) . Deps^{-1} (0)
    // Let c0(epsilon) = delta o h o eps^{-1} (epsilon)
    // Then we expect that
    // (c0(dt * epsilon) - c0(0)) / dt --> C0 epsilon
    // as dt --> 0, for any vector epsilon

    auto c0 = [&](const VectorXd& epsilon) {
        const auto y0 = measureSystemState(xi0);
        const auto xi_e = EqFOutputMatrixC_euclid.stateChartInv(epsilon, xi0);
        const auto y_e = measureSystemState(xi_e);
        const VectorXd delta = EqFOutputMatrixC_euclid.outputChart(y_e, y0);
        return delta;
    };

    // Check the function at zero
    const VectorXd c0AtZero = c0(VectorXd::Zero(5 + 3 * N));
    EXPECT_LE(c0AtZero.norm(), NEAR_ZERO);

    // Check individual directions of the differential
    for (int j = 0; j < 5 + 3 * N; ++j) {
        const VectorXd ej = VectorXd::Unit(5 + 3 * N, j);
        const VectorXd computedDiff = C0 * ej;

        double previousDist = 1e8;
        for (int i = 1; i < 8; ++i) {
            const double dt = pow(10.0, -i);
            const VectorXd trueDiff = c0(dt * ej) / dt;

            double dist = (trueDiff - computedDiff).norm();
            if (dist > 1e-8)
                EXPECT_LE(dist, previousDist);
            previousDist = dist;
        }
    }

    // Check some random directions
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VectorXd epsVec = VectorXd::Random(5 + 3 * N);
        const VectorXd computedEpsDot = C0 * epsVec;
        double previousDist = 1e8;
        for (int i = 1; i < 7; ++i) {
            const double dt = pow(10.0, -i);
            const VectorXd trueEpsDot = c0(dt * epsVec) / dt;

            double dist = (trueEpsDot - computedEpsDot).norm();
            EXPECT_LE(dist, previousDist);
            previousDist = dist;
        }
    }
}