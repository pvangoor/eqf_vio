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

using namespace Eigen;
using namespace std;

class SE3::SE3Impl {
  public:
    SO3 R;
    Vector3d x;

    SE3Impl(){};
    SE3Impl(const Matrix4d& mat);

    void setIdentity();
    Vector3d operator*(const Vector3d& point) const;
    SE3Impl operator*(const SE3Impl& other) const;

    void invert();
    Matrix<double, 6, 6> Adjoint() const;

    // Set and get
    Matrix4d asMatrix() const;
    void fromMatrix(const Matrix4d& mat);
};

SE3::SE3Impl::SE3Impl(const Matrix4d& mat) { this->fromMatrix(mat); }

SE3& SE3::operator=(const SE3& other) {
    (*this->pimpl) = (*other.pimpl);
    return *this;
}

SE3 SE3::Identity() {
    SE3 result;
    result.setIdentity();
    return result;
}

void SE3::setIdentity() { pimpl->setIdentity(); }

void SE3::SE3Impl::setIdentity() {
    R.setIdentity();
    x.setZero();
}

Vector3d SE3::operator*(const Vector3d& point) const { return (*pimpl) * point; }

Vector3d SE3::SE3Impl::operator*(const Vector3d& point) const { return R * point + x; }

SE3 SE3::operator*(const SE3& other) const {
    SE3 result;
    *result.pimpl = (*this->pimpl) * (*other.pimpl);
    return result;
}

SE3::SE3Impl SE3::SE3Impl::operator*(const SE3Impl& other) const {
    SE3Impl result;
    result.R = R * other.R;
    result.x = x + R * other.x;
    return result;
}

void SE3::invert() { pimpl->invert(); }

void SE3::SE3Impl::invert() {
    R.invert();
    x = -(R * x);
}

SE3 SE3::inverse() const {
    SE3 result(*this);
    result.invert();
    return result;
}

Matrix<double, 6, 6> SE3::Adjoint() const { return pimpl->Adjoint(); }

Matrix<double, 6, 6> SE3::SE3Impl::Adjoint() const {
    Matrix<double, 6, 6> AdMat;
    Matrix3d Rmat = R.asMatrix();
    AdMat.block<3, 3>(0, 0) = Rmat;
    AdMat.block<3, 3>(0, 3) = Matrix3d::Zero();
    AdMat.block<3, 3>(3, 0) = SO3::skew(x) * Rmat;
    AdMat.block<3, 3>(3, 3) = Rmat;
    return AdMat;
}

Matrix4d SE3::asMatrix() const { return pimpl->asMatrix(); }

Matrix4d SE3::SE3Impl::asMatrix() const {
    Matrix4d result;
    result.setIdentity();
    result.block<3, 3>(0, 0) = R.asMatrix();
    result.block<3, 1>(0, 3) = x;
    return result;
}

void SE3::fromMatrix(const Matrix4d& mat) { pimpl->fromMatrix(mat); }

void SE3::SE3Impl::fromMatrix(const Matrix4d& mat) {
    R.fromMatrix(mat.block<3, 3>(0, 0));
    x = mat.block<3, 1>(0, 3);
}

Matrix4d SE3::wedge(const se3vector& u) {
    // u is in the format (omega, v)
    Matrix4d result;
    result.block<3, 3>(0, 0) = SO3::skew(u.block<3, 1>(0, 0));
    result.block<3, 1>(0, 3) = u.block<3, 1>(3, 0);
    result.block<1, 4>(3, 0).setZero();
    return result;
}

se3vector SE3::vee(const Matrix4d& U) {
    // u is in the format (omega, v)
    se3vector result;
    result.block<3, 1>(0, 0) = SO3::vex(U.block<3, 3>(0, 0));
    result.block<3, 1>(3, 0) = U.block<3, 1>(0, 3);
    return result;
}

SE3 SE3::SE3Exp(const se3vector& u) {
    Vector3d w = u.block<3, 1>(0, 0);
    Vector3d v = u.block<3, 1>(3, 0);

    double th = w.norm();
    double A, B, C;
    if (abs(th) >= 1e-12) {
        A = sin(th) / th;
        B = (1 - cos(th)) / pow(th, 2);
        C = (1 - A) / pow(th, 2);
    } else {
        A = 1.0;
        B = 1.0 / 2.0;
        C = 1.0 / 6.0;
    }

    Matrix3d wx = SO3::skew(w);
    Matrix3d R = Matrix3d::Identity() + A * wx + B * wx * wx;
    Matrix3d V = Matrix3d::Identity() + B * wx + C * wx * wx;

    Matrix4d expMat = Matrix4d::Identity();
    expMat.block<3, 3>(0, 0) = R;
    expMat.block<3, 1>(0, 3) = V * v;

    return SE3(expMat);
}

se3vector SE3::SE3Log(const SE3& P) {
    SO3 R = P.pimpl->R;
    Vector3d x = P.pimpl->x;

    Matrix3d Omega = SO3::skew(SO3::SO3Log(R));

    double theta = SO3::vex(Omega).norm();
    double coefficient = 1.0 / 12.0;
    if (abs(theta) > 1e-8) {
        coefficient = 1 / (theta * theta) * (1 - (theta * sin(theta)) / (2 * (1 - cos(theta))));
    }

    Matrix3d VInv = Matrix3d::Identity() - 0.5 * Omega + coefficient * Omega * Omega;
    Vector3d v = VInv * x;

    Matrix4d U = Matrix4d::Zero();
    U.block<3, 3>(0, 0) = Omega;
    U.block<3, 1>(0, 3) = v;

    return SE3::vee(U);
}

SO3& SE3::R() const { return pimpl->R; }

Vector3d& SE3::x() const { return pimpl->x; }

SE3::SE3() : pimpl(make_unique<SE3Impl>()) {}
SE3::SE3(SE3 const& old) : pimpl(make_unique<SE3Impl>(*old.pimpl)) {}
SE3::SE3(const Matrix4d& mat) : pimpl(make_unique<SE3Impl>(mat)) {}
SE3::~SE3() = default;