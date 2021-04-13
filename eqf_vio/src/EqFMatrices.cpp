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

#include "eqf_vio/EqFMatrices.h"

using namespace Eigen;
using namespace std;

Eigen::MatrixXd EqFStateMatrixA_euclid_impl(const VIOGroup& X, const VIOManifoldState& xi0, const IMUVelocity& imuVel);
Eigen::MatrixXd EqFInputMatrixB_euclid_impl(const VIOGroup& X, const VIOManifoldState& xi0);
Eigen::MatrixXd EqFOutputMatrixC_euclid_impl(const VIOManifoldState& xi0);

const EqFStateMatrixA EqFStateMatrixA_euclid{
    EqFStateMatrixA_euclid_impl, euclidCoordinateChart, euclidCoordinateChartInv};

const EqFInputMatrixB EqFInputMatrixB_euclid{EqFInputMatrixB_euclid_impl, euclidCoordinateChart};

const EqFOutputMatrixC EqFOutputMatrixC_euclid{
    EqFOutputMatrixC_euclid_impl, euclidCoordinateChartInv, outputCoordinateChart};

VIOAlgebra liftInnovation(const Eigen::VectorXd& baseInnovation, const VIOManifoldState& xi0) {
    VIOAlgebra Delta;

    // Delta_A
    // Omega part
    const Vector2d& gamma_gravity = baseInnovation.block<2, 1>(0, 0);
    Delta.U.block<3, 1>(0, 0) =
        -SO3::skew(xi0.gravityDir) * stereoSphereChartInvDiff(Vector2d::Zero(), xi0.gravityDir) * gamma_gravity;
    // V part
    Delta.U.block<3, 1>(3, 0) = Vector3d::Zero();

    // Delta w
    const Vector3d& gamma_v = baseInnovation.block<3, 1>(2, 0);
    Delta.u = -gamma_v - SO3::skew(Delta.U.block<3, 1>(0, 0)) * xi0.velocity;

    // Delta q_i
    const int N = xi0.bodyLandmarks.size();
    Delta.id.resize(N);
    Delta.W.resize(N);
    for (int i = 0; i < N; ++i) {
        const Vector3d& gamma_qi0 = baseInnovation.block<3, 1>(5 + 3 * i, 0);
        const Vector3d& qi0 = xi0.bodyLandmarks[i].p;

        // Rotation part
        Delta.W[i].block<3, 1>(0, 0) = -qi0.cross(gamma_qi0) / qi0.squaredNorm();
        // scale part
        Delta.W[i](3) = -qi0.dot(gamma_qi0) / qi0.squaredNorm();
        // id number
        Delta.id[i] = xi0.bodyLandmarks[i].id;
    }

    return Delta;
}

VIOAlgebra liftTotalSpaceInnovation(const Eigen::VectorXd& totalInnovation, const VIOState& xi0) {
    VIOAlgebra Delta;

    // Delta_A
    Delta.U = totalInnovation.block<6, 1>(0, 0);

    // Delta w
    const Vector3d& gamma_v = totalInnovation.block<3, 1>(6, 0);
    Delta.u = -gamma_v - SO3::skew(Delta.U.block<3, 1>(0, 0)) * xi0.velocity;

    // Delta q_i
    const int N = xi0.bodyLandmarks.size();
    Delta.id.resize(N);
    Delta.W.resize(N);
    for (int i = 0; i < N; ++i) {
        const Vector3d& gamma_qi0 = totalInnovation.block<3, 1>(9 + 3 * i, 0);
        const Vector3d& qi0 = xi0.bodyLandmarks[i].p;

        // Rotation part
        Delta.W[i].block<3, 1>(0, 0) = -qi0.cross(gamma_qi0) / qi0.squaredNorm();
        // scale part
        Delta.W[i](3) = -qi0.dot(gamma_qi0) / qi0.squaredNorm();
        // id number
        Delta.id[i] = xi0.bodyLandmarks[i].id;
    }

    return Delta;
}

VIOAlgebra liftInnovation(
    const Eigen::VectorXd& baseInnovation, const VIOState& xi0, const VIOGroup& X, const MatrixXd& Sigma) {
    // Lift the innovation using weighted least-squares

    // First we determine the known parts of Delta as before
    VIOAlgebra Delta = liftInnovation(baseInnovation, xi0);
    const VIOState xiHat = stateGroupAction(X, xi0);
    const VIOManifoldState xi0Projected = projectToManifold(xi0);
    const Vector3d& eta0 = xi0Projected.gravityDir.normalized();

    // The unknown parts of Delta are those corresponding to the vehicle yaw and position

    // Create some lambdas for constructing const matrices
    auto constructKPara = [](const Vector3d& eta) {
        Matrix<double, 6, 4> KPara = Matrix<double, 6, 4>::Zero();
        KPara.block<3, 1>(0, 0) = eta;
        KPara.block<3, 3>(3, 1) = Matrix3d::Identity();
        return KPara;
    };
    auto constructKPerp = [](const Vector3d& eta) {
        Matrix<double, 6, 6> KPerp = Matrix<double, 6, 6>::Zero();
        KPerp.block<3, 3>(0, 0) = Matrix3d::Identity() - eta * eta.transpose();
        KPerp.block<3, 3>(3, 1) = Matrix3d::Zero();
        return KPerp;
    };

    // Use some sensible variable names
    const int N = xi0.bodyLandmarks.size();
    const SO3 R_C = xiHat.pose.R() * xiHat.cameraOffset.R();
    const Matrix3d R_CTransMat = R_C.inverse().asMatrix();
    const Matrix<double, 6, 6> AdP0 = xi0.pose.Adjoint();
    const Matrix<double, 6, 4> KPara = constructKPara(eta0);
    const Matrix<double, 6, 6> KPerp = constructKPerp(eta0);
    const Matrix<double, 6, 1> DeltaUFixed = KPerp * Delta.U;

    // Set up the components of a least squares problem
    Matrix<double, Dynamic, 4> coeffMat = Matrix<double, Dynamic, 4>(3 * N, 4);
    VectorXd observationVec(3 * N);
    MatrixXd weightingTransferD = MatrixXd::Zero(5 + 3 * N, 3 * N);

    // Populate the least squares components
    for (int i = 0; i < N; ++i) {
        const Vector3d& gamma_qi0 = baseInnovation.block<3, 1>(5 + 3 * i, 0);
        const Vector3d& pHat_i = xiHat.pose * xiHat.cameraOffset * xiHat.bodyLandmarks[i].p;

        // Populate the observation vector
        const Vector3d alpha = -(R_C * (X.Q[i].inverse() * gamma_qi0));
        Matrix<double, 3, 6> pHatMat;
        pHatMat.block<3, 3>(0, 0) = -SO3::skew(pHat_i);
        pHatMat.block<3, 3>(0, 3) = Matrix3d::Identity();
        const Vector3d obsVecBlock = alpha - pHatMat * AdP0 * DeltaUFixed;
        observationVec.block<3, 1>(3 * i, 0) = obsVecBlock;

        // Populate the coefficient matrix
        Matrix<double, 3, 4> coeffMatBlock = pHatMat * AdP0 * KPara;
        coeffMat.block<3, 4>(3 * i, 0) = coeffMatBlock;

        // Populate the weighting transfer matrix
        weightingTransferD.block<3, 3>(5 + 3 * i, 3 * i) = X.Q[i].asMatrix3d() * R_CTransMat;
    }

    // Compute the weighted least-squares solution
    const MatrixXd weightMat = weightingTransferD.transpose() * Sigma.inverse() * weightingTransferD;
    const Vector4d WLSSolution = (coeffMat.transpose() * weightMat * coeffMat)
                                     .householderQr()
                                     .solve(coeffMat.transpose() * weightMat * observationVec);
    Delta.U = DeltaUFixed + KPara * WLSSolution;

    // We must re-compute the velocity component of Delta
    const Vector3d& gamma_v = baseInnovation.block<3, 1>(2, 0);
    Delta.u = -gamma_v - SO3::skew(Delta.U.block<3, 1>(0, 0)) * xi0.velocity;

    return Delta;
}

Eigen::VectorXd bundleLift(
    const Eigen::VectorXd& baseInnovation, const VIOState& xi0, const VIOGroup& X, const MatrixXd& Sigma) {
    // Lift the innovation to the total space using weighted least-squares

    const VIOState xiHat = stateGroupAction(X, xi0);
    const VIOManifoldState xi0Projected = projectToManifold(xi0);
    const int N = xi0.bodyLandmarks.size();
    const Vector3d& eta0 = xi0Projected.gravityDir.normalized();

    // Construct the default  Delta
    const Vector2d& gamma_gravity = baseInnovation.block<2, 1>(0, 0);
    se3vector DeltaU;
    DeltaU.block<3, 1>(0, 0) = -SO3::skew(eta0) * stereoSphereChartInvDiff(Vector2d::Zero(), eta0) * gamma_gravity;
    DeltaU.block<3, 1>(3, 0) = Vector3d::Zero();

    // The unknown parts of the innovation are those corresponding to the vehicle yaw and position

    // Create some lambdas for constructing const matrices
    auto constructKPara = [](const Vector3d& eta) {
        Matrix<double, 6, 4> KPara = Matrix<double, 6, 4>::Zero();
        KPara.block<3, 1>(0, 0) = eta;
        KPara.block<3, 3>(3, 1) = Matrix3d::Identity();
        return KPara;
    };
    auto constructKPerp = [](const Vector3d& eta) {
        Matrix<double, 6, 6> KPerp = Matrix<double, 6, 6>::Zero();
        KPerp.block<3, 3>(0, 0) = Matrix3d::Identity() - eta * eta.transpose();
        KPerp.block<3, 3>(3, 1) = Matrix3d::Zero();
        return KPerp;
    };

    // Use some sensible variable names
    const SO3 R_C = xiHat.pose.R() * xiHat.cameraOffset.R();
    const Matrix3d R_CTransMat = R_C.inverse().asMatrix();
    const Matrix<double, 6, 6> AdP0 = xi0.pose.Adjoint();
    const Matrix<double, 6, 4> KPara = constructKPara(eta0);
    const Matrix<double, 6, 6> KPerp = constructKPerp(eta0);
    const Matrix<double, 6, 1> DeltaUFixed = KPerp * DeltaU;

    // Set up the components of a least squares problem
    Matrix<double, Dynamic, 4> coeffMat = Matrix<double, Dynamic, 4>(3 * N, 4);
    VectorXd observationVec(3 * N);
    MatrixXd weightingTransferD = MatrixXd::Zero(5 + 3 * N, 3 * N);

    // Populate the least squares components
    for (int i = 0; i < N; ++i) {
        const Vector3d& gamma_qi0 = baseInnovation.block<3, 1>(5 + 3 * i, 0);
        const Vector3d& pHat_i = xiHat.pose * xiHat.cameraOffset * xiHat.bodyLandmarks[i].p;

        // Populate the observation vector
        const Vector3d alpha = -(R_C * (X.Q[i].inverse() * gamma_qi0));
        Matrix<double, 3, 6> pHatMat;
        pHatMat.block<3, 3>(0, 0) = -SO3::skew(pHat_i);
        pHatMat.block<3, 3>(0, 3) = Matrix3d::Identity();
        const Vector3d obsVecBlock = alpha - pHatMat * AdP0 * DeltaUFixed;
        observationVec.block<3, 1>(3 * i, 0) = obsVecBlock;

        // Populate the coefficient matrix
        Matrix<double, 3, 4> coeffMatBlock = pHatMat * AdP0 * KPara;
        coeffMat.block<3, 4>(3 * i, 0) = coeffMatBlock;

        // Populate the weighting transfer matrix
        weightingTransferD.block<3, 3>(5 + 3 * i, 3 * i) = X.Q[i].asMatrix3d() * R_CTransMat;
    }

    // Compute the weighted least-squares solution
    const MatrixXd weightMat = weightingTransferD.transpose() * Sigma.inverse() * weightingTransferD;
    const Vector4d WLSSolution = (coeffMat.transpose() * weightMat * coeffMat)
                                     .householderQr()
                                     .solve(coeffMat.transpose() * weightMat * observationVec);
    DeltaU = DeltaUFixed + KPara * WLSSolution;

    // We must re-compute the velocity component of Delta
    VectorXd liftedInnovation = VectorXd(9 + 3 * N);
    liftedInnovation.block(6, 0, 3 + 3 * N, 1) = baseInnovation.block(2, 0, 3 + 3 * N, 1);
    liftedInnovation.block(6, 0, 3 + 3 * N, 1) = baseInnovation.block(2, 0, 3 + 3 * N, 1);
    liftedInnovation.block<6, 1>(0, 0) = DeltaU;

    return liftedInnovation;
}

VIOGroup liftTotalSpaceInnovationDiscrete(const Eigen::VectorXd& totalInnovation, const VIOState& xi0) {
    // Lift the innovation discretely
    VIOGroup lift;
    lift.A = SE3::SE3Exp(totalInnovation.block<6, 1>(0, 0));
    lift.w = xi0.velocity - lift.A.R() * (xi0.velocity + totalInnovation.block<3, 1>(6, 0));

    // Lift for each of the points
    const int N = xi0.bodyLandmarks.size();
    lift.id.resize(N);
    lift.Q.resize(N);
    for (int i = 0; i < N; ++i) {
        const Vector3d& qi = xi0.bodyLandmarks[i].p;
        const Vector3d& Gamma_qi = totalInnovation.block<3, 1>(9 + 3 * i, 0);
        const Vector3d qi1 = (qi + Gamma_qi);
        lift.Q[i].R() = SO3::SO3FromVectors(qi1.normalized(), qi.normalized());
        lift.Q[i].a() = qi.norm() / qi1.norm();

        lift.id[i] = xi0.bodyLandmarks[i].id;
    }

    return lift;
}

Eigen::MatrixXd EqFStateMatrixA_euclid_impl(const VIOGroup& X, const VIOManifoldState& xi0, const IMUVelocity& imuVel) {
    const int N = xi0.bodyLandmarks.size();
    MatrixXd A0t = MatrixXd::Zero(5 + 3 * N, 5 + 3 * N);

    // Rows / Cols and their corresponding state components
    // I am using zero indexing and half open ranges
    // [0,2): Gravity vector (deviation from e3)
    // [2,5) Body-fixed velocity
    // [5+3i,5+3(i+1)): Body-fixed landmark i

    // Effect of gravity cov on velocity cov
    A0t.block<3, 2>(2, 0) = -Matrix<double, 3, 2>::Identity() * GRAVITY_CONSTANT;
    A0t.block<3, 2>(2, 0) = -stereoSphereChartInvDiff(Vector2d::Zero(), xi0.gravityDir) * GRAVITY_CONSTANT;

    // Effect of velocity cov on landmarks cov
    const Matrix3d R_IC = xi0.cameraOffset.R().asMatrix();
    const Matrix3d R_Ahat = X.A.R().asMatrix();
    for (int i = 0; i < N; ++i) {
        const Matrix3d Qhat_i = X.Q[i].R().asMatrix() * X.Q[i].a();
        A0t.block<3, 3>(5 + 3 * i, 2) = -Qhat_i * R_IC.transpose() * R_Ahat.transpose();
    }

    // Effect of landmark cov on landmark cov
    const VIOManifoldState xi_hat = stateGroupAction(X, xi0);

    const se3vector U_I = (se3vector() << imuVel.omega, xi_hat.velocity).finished();
    const se3vector U_C = xi0.cameraOffset.inverse().Adjoint() * U_I;
    const Vector3d v_C = U_C.block<3, 1>(3, 0);
    for (int i = 0; i < N; ++i) {
        const Matrix3d Qhat_i = X.Q[i].R().asMatrix() * X.Q[i].a();
        const Vector3d& qhat_i = xi_hat.bodyLandmarks[i].p;
        const Matrix3d A_qi =
            -Qhat_i * (SO3::skew(qhat_i) * SO3::skew(v_C) - 2 * v_C * qhat_i.transpose() + qhat_i * v_C.transpose()) *
            Qhat_i.inverse() * (1 / qhat_i.squaredNorm());
        A0t.block<3, 3>(5 + 3 * i, 5 + 3 * i) = A_qi;
    }

    assert(!A0t.hasNaN());

    return A0t;
}

Eigen::MatrixXd EqFOutputMatrixC_euclid_impl(const VIOManifoldState& xi0) {
    const int N = xi0.bodyLandmarks.size();
    MatrixXd C0 = MatrixXd::Zero(2 * N, 5 + 3 * N);

    // Rows and their corresponding output components
    // [2i, 2i+2): Landmark measurement i

    // Cols and their corresponding state components
    // I am using zero indexing and half open ranges
    // [0,2): Gravity vector (deviation from e3)
    // [2,5) Body-fixed velocity
    // [5+3i,5+3(i+1)): Body-fixed landmark i

    for (int i = 0; i < N; ++i) {
        const Vector3d& qi0 = xi0.bodyLandmarks[i].p;
        const Vector3d yi0 = qi0.normalized();

        const Matrix<double, 2, 3> C0i =
            1 / qi0.norm() * stereoSphereChartDiff(yi0, yi0) * (Matrix3d::Identity() - yi0 * yi0.transpose());
        C0.block<2, 3>(2 * i, 5 + 3 * i) = C0i;
    }

    assert(!C0.hasNaN());

    return C0;
}

Eigen::MatrixXd EqFInputMatrixB_euclid_impl(const VIOGroup& X, const VIOManifoldState& xi0) {
    const int N = xi0.bodyLandmarks.size();
    MatrixXd Bt = MatrixXd::Zero(5 + 3 * N, 6);

    // Rows and their corresponding state components
    // I am using zero indexing and half open ranges
    // [0,2): Gravity vector (deviation from e3)
    // [2,5) Body-fixed velocity
    // [5+3i,5+3(i+1)): Body-fixed landmark i

    // Cols and their corresponding output components
    // [0, 3): Angular velocity omega
    // [3, 6): Linear acceleration accel

    const VIOManifoldState xi_hat = stateGroupAction(X, xi0);

    // Gravity vector
    const Matrix3d R_A = X.A.R().asMatrix();
    Bt.block<2, 3>(0, 0) = stereoSphereChartDiff(xi0.gravityDir, xi0.gravityDir) * R_A * SO3::skew(xi_hat.gravityDir);

    // Body fixed velocity
    Bt.block<3, 3>(2, 0) = R_A * SO3::skew(xi_hat.velocity);
    Bt.block<3, 3>(2, 3) = R_A;

    // Landmarks
    const Matrix3d RT_IC = xi0.cameraOffset.R().inverse().asMatrix();
    const Vector3d x_IC = xi0.cameraOffset.x();
    for (int i = 0; i < N; ++i) {
        const Matrix3d Qhat_i = X.Q[i].R().asMatrix() * X.Q[i].a();
        const Vector3d& qhat_i = xi_hat.bodyLandmarks[i].p;
        Bt.block<3, 3>(5 + 3 * i, 0) = Qhat_i * (SO3::skew(qhat_i) * RT_IC + RT_IC * SO3::skew(x_IC));
    }

    assert(!Bt.hasNaN());

    return Bt;
}
