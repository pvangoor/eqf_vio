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
#include "eqf_vio/VIOGroup.h"
#include "eqf_vio/VIOState.h"
#include "testing_utilities.h"
#include "gtest/gtest.h"

using namespace Eigen;
using namespace std;

TEST(VIOLiftTest, Lift) {
    vector<int> ids = {0, 1, 2, 3, 4};
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOState xi0 = randomStateElement(ids);
        const IMUVelocity velocity = randomVelocityElement();

        // Check the convergence of the derivative
        double previousDist = 1e8;
        for (int i = 0; i < 8; ++i) {
            // Integrate the system normally
            const double dt = pow(10.0, -i);
            const VIOState xi1 = integrateSystemFunction(xi0, velocity, dt);

            // Lift the velocity and apply as a group action
            const VIOAlgebra lambda = liftVelocity(xi0, velocity);
            const VIOGroup lambdaExp = VIOExp(dt * lambda);
            const VIOState xi2 = stateGroupAction(lambdaExp, xi0);

            // Check the error has decreased
            const VectorXd dxi1 = stateVecDiff(xi0, xi1) / dt;
            const VectorXd dxi2 = stateVecDiff(xi0, xi2) / dt;
            const VectorXd dxiErr = (dxi1 - dxi2);
            const double diffDist = dxiErr.norm();
            EXPECT_LE(diffDist, previousDist);
            previousDist = diffDist;
        }
    }
}

TEST(VIOLiftTest, DiscreteLift) {
    vector<int> ids = {0, 1, 2, 3, 4};
    const double dt = 0.1;
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOState Xi0 = randomStateElement(ids);
        const IMUVelocity velocity = randomVelocityElement();

        // Check the discrete result
        const VIOState Xi1 = integrateSystemFunction(Xi0, velocity, dt);

        const VIOManifoldState xi0 = projectToManifold(Xi0);
        const VIOManifoldState xi1 = projectToManifold(Xi1);

        const VIOGroup X = liftVelocityDiscrete(xi0, velocity, dt);
        const VIOManifoldState xi2 = stateGroupAction(X, xi0);

        const double dist12 = stateDistance(xi1, xi2);
        EXPECT_LE(dist12, NEAR_ZERO);
    }
}

TEST(VIOLiftTest, InnovationLift) {
    vector<int> ids = {0, 1, 2, 3, 4};
    const int N = ids.size();
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOManifoldState xi0 = projectToManifold(randomStateElement(ids));
        const VectorXd baseInnovation = VectorXd::Random(5 + 3 * N);
        const VIOAlgebra liftedInnovation = liftInnovation(baseInnovation, xi0);

        // Check the convergence of the derivative
        double previousDist = 1e8;
        for (int i = 0; i < 8; ++i) {
            const double dt = pow(10.0, -i);
            // Apply the lifted innovation

            const VIOGroup Delta = VIOExp(dt * liftedInnovation);
            const VIOManifoldState xi1 = stateGroupAction(Delta, xi0);
            const VectorXd reprojectedInnovation = euclidCoordinateChart(xi1, xi0) / dt;

            const double innovationDist = (reprojectedInnovation - baseInnovation).norm();
            EXPECT_LE(innovationDist, previousDist);
            previousDist = innovationDist;
        }
    }
}

TEST(VIOLiftTest, InnovationLiftUnitDirs) {
    vector<int> ids = {0, 1, 2, 3, 4};
    const int N = ids.size();
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOManifoldState xi0 = projectToManifold(randomStateElement(ids));
        for (int j = 0; j < 5 + 3 * N; ++j) {
            const VectorXd baseInnovation = VectorXd::Unit(5 + 3 * N, j);
            const VIOAlgebra liftedInnovation = liftInnovation(baseInnovation, xi0);

            // Check the convergence of the derivative
            double previousDist = 1e8;
            for (int i = 1; i < 5; ++i) {
                const double dt = pow(10.0, -i);
                // Apply the lifted innovation

                const VIOGroup Delta = VIOExp(dt * liftedInnovation);
                const VIOManifoldState xi1 = stateGroupAction(Delta, xi0);
                const VectorXd reprojectedInnovation = euclidCoordinateChart(xi1, xi0) / dt;

                const double innovationDist = (reprojectedInnovation - baseInnovation).norm();
                if (innovationDist < NEAR_ZERO)
                    break;
                EXPECT_LE(innovationDist, previousDist);
                previousDist = innovationDist;
            }
        }
    }
}

TEST(VIOLiftTest, FullInnovationLiftUnitDirs) {
    vector<int> ids = {0, 1, 2, 3, 4};
    const int N = ids.size();
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOState Xi0 = randomStateElement(ids);
        const VIOManifoldState xi0 = projectToManifold(Xi0);
        const VIOGroup X = randomGroupElement(ids);
        MatrixXd Sigma = MatrixXd::Random(5 + 3 * N, 5 + 3 * N);
        Sigma = Sigma * Sigma.transpose(); // Makes Sigma positive (semi) definite
        for (int j = 0; j < 5 + 3 * N; ++j) {
            const VectorXd baseInnovation = VectorXd::Unit(5 + 3 * N, j);
            const VIOAlgebra liftedInnovation = liftInnovation(baseInnovation, Xi0, X, Sigma);

            // Check the convergence of the derivative
            double previousDist = 1e8;
            for (int i = 1; i < 8; ++i) {
                const double dt = pow(10.0, -i);
                // Apply the lifted innovation

                const VIOGroup Delta = VIOExp(dt * liftedInnovation);
                const VIOManifoldState xi1 = stateGroupAction(Delta, xi0);
                const VectorXd reprojectedInnovation = euclidCoordinateChart(xi1, xi0) / dt;

                const double innovationDist = (reprojectedInnovation - baseInnovation).norm();
                if (innovationDist < NEAR_ZERO)
                    break;
                EXPECT_LE(innovationDist, previousDist);
                previousDist = innovationDist;
            }
        }
    }
}

TEST(VIOLiftTest, TotalInnovationLiftUnitDirs) {
    vector<int> ids = {0, 1, 2, 3, 4};
    const int N = ids.size();
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOState Xi0 = randomStateElement(ids);
        const VIOManifoldState xi0 = projectToManifold(Xi0);
        const VIOGroup X = randomGroupElement(ids);
        MatrixXd Sigma = MatrixXd::Random(5 + 3 * N, 5 + 3 * N);
        Sigma = Sigma * Sigma.transpose(); // Makes Sigma positive (semi) definite
        for (int j = 0; j < 5 + 3 * N; ++j) {
            const VectorXd baseInnovation = VectorXd::Unit(5 + 3 * N, j);
            const VectorXd totalInnovation = bundleLift(baseInnovation, Xi0, X, Sigma);
            const VIOAlgebra liftedInnovation = liftTotalSpaceInnovation(totalInnovation, Xi0);

            // Check the convergence of the derivative
            double previousDist = 1e8;
            for (int i = 1; i < 8; ++i) {
                const double dt = pow(10.0, -i);
                // Apply the lifted innovation

                const VIOGroup Delta = VIOExp(dt * liftedInnovation);
                const VIOManifoldState xi1 = stateGroupAction(Delta, xi0);
                const VectorXd reprojectedInnovation = euclidCoordinateChart(xi1, xi0) / dt;

                const double innovationDist = (reprojectedInnovation - baseInnovation).norm();
                if (innovationDist < NEAR_ZERO)
                    break;
                EXPECT_LE(innovationDist, previousDist);
                previousDist = innovationDist;
            }
        }
    }
}

TEST(VIOLiftTest, TotalInnovationLiftUnitDirsComparison) {
    vector<int> ids = {0, 1, 2, 3, 4};
    const int N = ids.size();
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOState Xi0 = randomStateElement(ids);
        const VIOManifoldState xi0 = projectToManifold(Xi0);
        const VIOGroup X = randomGroupElement(ids);
        MatrixXd Sigma = MatrixXd::Random(5 + 3 * N, 5 + 3 * N);
        Sigma = Sigma * Sigma.transpose(); // Makes Sigma positive (semi) definite
        for (int j = 0; j < 5 + 3 * N; ++j) {
            const VectorXd baseInnovation = VectorXd::Unit(5 + 3 * N, j) / 10;
            const VectorXd totalInnovation = bundleLift(baseInnovation, Xi0, X, Sigma);
            const VIOAlgebra liftedInnovation1 = liftTotalSpaceInnovation(totalInnovation, Xi0);
            const VIOAlgebra liftedInnovation2 = liftInnovation(baseInnovation, Xi0, X, Sigma);

            const VIOGroup groupError = VIOExp(liftedInnovation1 - liftedInnovation2);
            const double liftError = logNorm(groupError);
            EXPECT_LE(liftError, NEAR_ZERO * 1e2);
        }
    }
}

TEST(VIOLiftTest, DiscreteLiftUnitDirsComparison) {
    vector<int> ids = {0, 1, 2, 3, 4};
    const int N = ids.size();
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOState Xi0 = randomStateElement(ids);
        const VIOManifoldState xi0 = projectToManifold(Xi0);
        const VIOGroup X = randomGroupElement(ids);
        MatrixXd Sigma = MatrixXd::Random(5 + 3 * N, 5 + 3 * N);
        Sigma = Sigma * Sigma.transpose(); // Makes Sigma positive (semi) definite
        for (int j = 0; j < 5 + 3 * N; ++j) {
            const VectorXd baseInnovation = VectorXd::Unit(5 + 3 * N, j) / 10;
            const VectorXd totalInnovation = bundleLift(baseInnovation, Xi0, X, Sigma);

            // Check the convergence of the derivative
            double previousDist = 1e8;
            for (int i = 1; i < 8; ++i) {
                const double dt = pow(10.0, -i);

                // Compare the innovation lifts
                const VIOGroup discreteInn1 = liftTotalSpaceInnovationDiscrete(dt * totalInnovation, Xi0);
                const VIOAlgebra continuousInn = liftTotalSpaceInnovation(dt * totalInnovation, Xi0);
                const VIOGroup discreteInn2 = VIOExp(continuousInn);

                const double liftDist = logNorm(discreteInn1.inverse() * discreteInn2);
                if (liftDist < NEAR_ZERO)
                    break;
                EXPECT_LE(liftDist, previousDist);
                previousDist = liftDist;
            }
        }
    }
}
