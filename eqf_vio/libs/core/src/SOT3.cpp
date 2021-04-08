#include "SOT3.h"

using namespace std;
using namespace Eigen;

class SOT3::SOT3Impl {
  public:
    SO3 Q;
    double a;

    SOT3Impl(){};
    SOT3Impl(const Matrix4d& mat);

    void setIdentity();
    Vector3d operator*(const Vector3d& point) const;
    SOT3Impl operator*(const SOT3Impl& other) const;

    Vector3d applyInverse(const Vector3d& p) const;
    void invert();
    SOT3Impl inverse() const;

    // Set and get
    Matrix4d asMatrix() const;
    Matrix3d asMatrix3d() const;
    void fromMatrix(const Matrix4d& mat);
};

SOT3::SOT3(const Matrix4d& mat) { *pimpl = SOT3Impl(mat); }

SOT3::SOT3Impl::SOT3Impl(const Matrix4d& mat) { this->fromMatrix(mat); }

SOT3& SOT3::operator=(const SOT3& other) {
    (*this->pimpl) = (*other.pimpl);
    return *this;
}

SOT3 SOT3::Identity() {
    SOT3 R;
    R.setIdentity();
    return R;
}

void SOT3::setIdentity() { pimpl->setIdentity(); }

void SOT3::SOT3Impl::setIdentity() {
    Q.setIdentity();
    a = 1.0;
}

Vector3d SOT3::operator*(const Vector3d& point) const { return (*pimpl) * point; }

Vector3d SOT3::SOT3Impl::operator*(const Vector3d& point) const { return a * (Q * point); }

SOT3 SOT3::operator*(const SOT3& other) const {
    SOT3 result;
    *result.pimpl = (*this->pimpl) * (*other.pimpl);
    return result;
}

SOT3::SOT3Impl SOT3::SOT3Impl::operator*(const SOT3::SOT3Impl& other) const {
    SOT3::SOT3Impl result;
    result.Q = Q * other.Q;
    result.a = a * other.a;
    return result;
}

void SOT3::invert() { pimpl->invert(); }

void SOT3::SOT3Impl::invert() {
    Q.invert();
    a = 1.0 / a;
}

SOT3 SOT3::inverse() const {
    SOT3 result(*this);
    result.invert();
    return result;
}

Matrix4d SOT3::asMatrix() const { return pimpl->asMatrix(); }
Matrix3d SOT3::asMatrix3d() const { return pimpl->asMatrix3d(); }

Matrix4d SOT3::SOT3Impl::asMatrix() const {
    Matrix4d result = Matrix4d::Identity();
    result.block<3, 3>(0, 0) = Q.asMatrix();
    result(3, 3) = a;
    return result;
}

Matrix3d SOT3::SOT3Impl::asMatrix3d() const {
    Matrix3d result = a * Q.asMatrix();
    return result;
}

void SOT3::fromMatrix(const Matrix4d& mat) { pimpl->fromMatrix(mat); }

void SOT3::SOT3Impl::fromMatrix(const Matrix4d& mat) {
    Q.fromMatrix(mat.block<3, 3>(0, 0));
    a = mat(3, 3);
}

Vector3d SOT3::applyInverse(const Vector3d& p) const { return pimpl->applyInverse(p); }

Vector3d SOT3::SOT3Impl::applyInverse(const Vector3d& p) const { return (1 / a) * Q.applyInverse(p); }

SO3& SOT3::R() const { return pimpl->Q; }

double& SOT3::a() const { return pimpl->a; }

SOT3 SOT3::SOT3Exp(const Vector4d& w) {
    SOT3 result;
    result.R() = SO3::SO3Exp(w.block<3, 1>(0, 0));
    result.a() = exp(w(3));
    return result;
}

Vector4d SOT3::SOT3Log(const SOT3& T) {
    Vector4d result;
    result.block<3, 1>(0, 0) = SO3::SO3Log(T.R());
    result(3) = log(T.a());
    return result;
}

SOT3::SOT3() : pimpl(std::make_unique<SOT3Impl>()) {}
SOT3::SOT3(SOT3 const& old) : pimpl(std::make_unique<SOT3Impl>(*old.pimpl)) {}
SOT3::~SOT3() = default;