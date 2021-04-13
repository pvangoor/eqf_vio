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

#include "SE3.h"
#include "SO3.h"
#include "eigen3/unsupported/Eigen/MatrixFunctions"
#include "gtest/gtest.h"
// #include <iostream>

using namespace std;
using namespace Eigen;

TEST(TestCommon, SkewVex) {
    for (int i = 0; i < 100; ++i) {
        Vector3d v = Vector3d::Random();
        Vector3d w = Vector3d::Random();
        Vector3d r1 = v.cross(w);
        Vector3d r2 = SO3::skew(v) * w;

        double error = (r1 - r2).norm();
        EXPECT_LE(error, 1e-8);
        EXPECT_EQ(SO3::vex(SO3::skew(v)), v);
    }
}

TEST(TestCommon, SO3) {
    // Test the SO(3) exponential
    for (int i = 0; i < 100; ++i) {
        Vector3d v = Vector3d::Random();

        Matrix3d R1 = SO3::skew(v).exp();
        Matrix3d R2 = SO3::SO3Exp(v).asMatrix();

        double error = (R1 - R2).norm();
        EXPECT_LE(error, 1e-8);

        double errorR = (R1.transpose() * R1 - Matrix3d::Identity()).norm();
        EXPECT_LE(errorR, 1e-8);
    }

    // Test generating an SO(3) matrix between two vectors
    for (int i = 0; i < 100; ++i) {
        Vector3d v = Vector3d::Random();
        Vector3d w = Vector3d::Random();

        v = v.normalized();
        w = w.normalized();

        Matrix3d R = SO3::SO3FromVectors(v, w).asMatrix();
        Vector3d w2 = R * v;
        double errorW = (w - w2).norm();
        EXPECT_LE(errorW, 1e-8);

        double errorR = (R.transpose() * R - Matrix3d::Identity()).norm();
        EXPECT_LE(errorR, 1e-8);
    }

    // Test the SO(3) logarithm
    for (int i = 0; i < 100; ++i) {
        Matrix3d Omega = SO3::skew(Vector3d::Random());

        Matrix3d R1 = Omega.exp();
        Matrix3d R2 = SO3::SO3Exp(SO3::vex(Omega)).asMatrix();

        Matrix3d Omega11 = R1.log();
        Matrix3d Omega12 = SO3::skew(SO3::SO3Log(R1));
        Matrix3d Omega21 = R2.log();
        Matrix3d Omega22 = SO3::skew(SO3::SO3Log(R2));

        EXPECT_LE((Omega - Omega11).norm(), 1e-8);
        EXPECT_LE((Omega - Omega12).norm(), 1e-8);
        EXPECT_LE((Omega - Omega21).norm(), 1e-8);
        EXPECT_LE((Omega - Omega22).norm(), 1e-8);
    }
}

TEST(TestCommon, SE3) {
    // Test the SE(3) exponential and logarithm
    for (int i = 0; i < 100; ++i) {
        Vector3d omega = Vector3d::Random();
        Vector3d v = Vector3d::Random();
        Matrix4d U = Matrix4d::Zero();
        U.block<3, 3>(0, 0) = SO3::skew(omega);
        U.block<3, 1>(0, 3) = v;

        Matrix4d X1 = U.exp();
        Matrix4d X2 = SE3::SE3Exp(SE3::vee(U)).asMatrix();

        double error = (X1 - X2).norm();
        EXPECT_LE(error, 1e-8);

        Matrix4d U11 = X1.log();
        Matrix4d U12 = SE3::wedge(SE3::SE3Log(X1));
        Matrix4d U21 = X2.log();
        Matrix4d U22 = SE3::wedge(SE3::SE3Log(X2));

        EXPECT_LE((U - U11).norm(), 1e-8);
        EXPECT_LE((U - U12).norm(), 1e-8);
        EXPECT_LE((U - U21).norm(), 1e-8);
        EXPECT_LE((U - U22).norm(), 1e-8);
    }
}

TEST(TestCommon, SE3Drift) {
    // Test the SE(3) exponential and logarithm
    SE3 drifter1 = SE3::Identity();
    SE3 drifter2 = SE3::Identity();
    SE3 drifter3 = SE3::Identity();
    SE3 drifter4 = SE3::Identity();

    for (int i = 0; i < 1000; ++i) {
        Vector3d omega = Vector3d::Random() * 1000;
        Vector3d v = Vector3d::Random() * 100;
        Matrix4d U = Matrix4d::Zero();
        U.block<3, 3>(0, 0) = SO3::skew(omega);
        U.block<3, 1>(0, 3) = v;
        Matrix<double, 6, 1> stepVec;
        stepVec.block<3, 1>(0, 0) = omega;
        stepVec.block<3, 1>(3, 0) = v;

        SE3 X1;
        X1.fromMatrix(U.exp());
        SE3 X2 = SE3::SE3Exp(SE3::vee(U)).asMatrix();

        drifter1 = drifter1 * SE3(X1);
        drifter2 = drifter2 * SE3(X2);
        drifter3 = SE3(X1) * drifter3;
        drifter4 = SE3(X2) * drifter4;

        double error1 = (drifter1.R().asMatrix() * drifter1.R().asMatrix().transpose() - Matrix3d::Identity()).norm();
        double error2 = (drifter2.R().asMatrix() * drifter2.R().asMatrix().transpose() - Matrix3d::Identity()).norm();
        double error3 = (drifter3.R().asMatrix() * drifter3.R().asMatrix().transpose() - Matrix3d::Identity()).norm();
        double error4 = (drifter4.R().asMatrix() * drifter4.R().asMatrix().transpose() - Matrix3d::Identity()).norm();

        // std::cout << "rotation drift 1: " << error1 << std::endl;
        // std::cout << "rotation drift 2: " << error2 << std::endl;
        // std::cout << "rotation drift 3: " << error3 << std::endl;
        // std::cout << "rotation drift 4: " << error4 << std::endl;

        EXPECT_LE(error1, 1e-8);
        EXPECT_LE(error2, 1e-8);
        EXPECT_LE(error3, 1e-8);
        EXPECT_LE(error4, 1e-8);
    }
}
