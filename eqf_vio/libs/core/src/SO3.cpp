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

#include "SO3.h"

using namespace std;
using namespace Eigen;

class SO3::SO3Impl {
  public:
    Quaterniond quat;

    void setIdentity();
    Vector3d operator*(const Vector3d& point) const;
    SO3Impl operator*(const SO3Impl& other) const;

    SO3Impl(){};
    SO3Impl(const Matrix3d& mat);
    SO3Impl(const Quaterniond& quat);

    void invert();
    Vector3d applyInverse(const Vector3d& point) const;

    // Set and get
    Matrix3d asMatrix() const;
    Quaterniond asQuaternion() const;
    void fromMatrix(const Matrix3d& mat);
    void fromQuaternion(const Quaterniond& quat);
};

SO3::SO3Impl::SO3Impl(const Matrix3d& mat) { this->fromMatrix(mat); }

SO3::SO3Impl::SO3Impl(const Quaterniond& quat) { this->fromQuaternion(quat); }

SO3& SO3::operator=(const SO3& other) {
    (*this->pimpl) = (*other.pimpl);
    return *this;
}

SO3 SO3::Identity() {
    SO3 R;
    R.setIdentity();
    return R;
}

void SO3::setIdentity() { pimpl->setIdentity(); }

void SO3::SO3Impl::setIdentity() { quat.setIdentity(); }

Vector3d SO3::operator*(const Vector3d& point) const { return (*pimpl) * point; }

Vector3d SO3::SO3Impl::operator*(const Vector3d& point) const { return quat * point; }

SO3 SO3::operator*(const SO3& other) const {
    SO3 result;
    *result.pimpl = (*this->pimpl) * (*other.pimpl);
    return result;
}

SO3::SO3Impl SO3::SO3Impl::operator*(const SO3::SO3Impl& other) const {
    SO3::SO3Impl result;
    result.quat = quat * other.quat;
    return result;
}

void SO3::invert() { pimpl->invert(); }

void SO3::SO3Impl::invert() { quat = quat.inverse(); }

SO3 SO3::inverse() const {
    SO3 result(*this);
    result.invert();
    return result;
}

Matrix3d SO3::asMatrix() const { return pimpl->asMatrix(); }

Matrix3d SO3::SO3Impl::asMatrix() const { return quat.matrix(); }

Quaterniond SO3::asQuaternion() const { return pimpl->asQuaternion(); }

Quaterniond SO3::SO3Impl::asQuaternion() const { return quat; }

void SO3::fromMatrix(const Matrix3d& mat) { pimpl->fromMatrix(mat); }

void SO3::SO3Impl::fromMatrix(const Matrix3d& mat) { quat = Quaterniond(mat); }

void SO3::fromQuaternion(const Quaterniond& quat) { pimpl->fromQuaternion(quat); }

void SO3::SO3Impl::fromQuaternion(const Quaterniond& q) { this->quat = q; }

Vector3d SO3::applyInverse(const Vector3d& point) const { return pimpl->applyInverse(point); }

Vector3d SO3::SO3Impl::applyInverse(const Vector3d& point) const { return quat.inverse() * point; }

Matrix3d SO3::skew(const Vector3d& v) {
    Matrix3d m;
    m << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
    return m;
}

Vector3d SO3::vex(const Matrix3d& Omega) {
    Vector3d v;
    v << Omega(2, 1), Omega(0, 2), Omega(1, 0);
    return v;
}

SO3 SO3::SO3Exp(const Vector3d& w) {
    double th = w.norm();

    double A, B;
    if (abs(th) >= 1e-8) {
        A = sin(th) / th;
        B = (1 - cos(th)) / pow(th, 2);
    } else {
        A = 1.0;
        B = 1.0 / 2.0;
    }

    Matrix3d wx = skew(w);
    Matrix3d R = Matrix3d::Identity() + A * wx + B * wx * wx;

    SO3 result;
    result.fromMatrix(R);
    return result;
}

Vector3d SO3::SO3Log(const SO3& rotation) {
    Matrix3d R = rotation.asMatrix();
    double theta = acos((R.trace() - 1.0) / 2.0);
    double coefficient = 0.5;
    if (abs(theta) >= 1e-6) {
        coefficient = theta / (2.0 * sin(theta));
    }

    Matrix3d Omega = coefficient * (R - R.transpose());

    return vex(Omega);
}

SO3 SO3::SO3FromVectors(const Vector3d& origin, const Vector3d& dest) {
    const Vector3d v = origin.normalized().cross(dest.normalized());
    const double c = origin.normalized().dot(dest.normalized());

    const Matrix3d mat = Matrix3d::Identity() + (skew(v) + 1 / (1 + c) * skew(v) * skew(v));
    if (abs(1 + c) <= 1e-8)
        throw(domain_error("The vectors cannot be exactly opposing."));

    SO3 result;
    result.fromMatrix(mat);

    return result;
}

SO3::SO3() : pimpl(make_unique<SO3Impl>()){};
SO3::SO3(const Matrix3d& mat) : pimpl(make_unique<SO3Impl>(mat)){};
SO3::SO3(const Quaterniond& quat) : pimpl(make_unique<SO3Impl>(quat)){};
SO3::SO3(SO3 const& old) : pimpl(make_unique<SO3Impl>(*old.pimpl)) {}
SO3::~SO3() = default;