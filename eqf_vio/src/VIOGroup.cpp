#include "eqf_vio/VIOGroup.h"

using namespace Eigen;
using namespace std;

VIOState stateGroupAction(const VIOGroup& X, const VIOState& state) {
    VIOState newState;
    newState.pose = state.pose * X.A;
    newState.velocity = X.A.R().inverse() * (state.velocity - X.w);
    newState.cameraOffset = state.cameraOffset;

    // Check the landmarks and transforms are aligned.
    assert(X.Q.size() == state.bodyLandmarks.size());
    for (size_t i = 0; i < X.Q.size(); ++i)
        assert(X.id[i] == state.bodyLandmarks[i].id);

    // Transform the body-fixed landmarks
    newState.bodyLandmarks.resize(state.bodyLandmarks.size());
    transform(state.bodyLandmarks.begin(), state.bodyLandmarks.end(), X.Q.begin(), newState.bodyLandmarks.begin(),
        [](const Point3d& lm, const SOT3& Q) {
            Point3d result;
            result.p = Q.inverse() * lm.p;
            result.id = lm.id;
            return result;
        });

    return newState;
}

VIOManifoldState stateGroupAction(const VIOGroup& X, const VIOManifoldState& state) {
    VIOManifoldState newState;
    newState.gravityDir = X.A.R().inverse() * state.gravityDir;
    newState.velocity = X.A.R().inverse() * (state.velocity - X.w);
    newState.cameraOffset = state.cameraOffset;

    // Check the landmarks and transforms are aligned.
    assert(X.Q.size() == state.bodyLandmarks.size());
    for (size_t i = 0; i < X.Q.size(); ++i)
        assert(X.id[i] == state.bodyLandmarks[i].id);

    // Transform the body-fixed landmarks
    newState.bodyLandmarks.resize(state.bodyLandmarks.size());
    transform(state.bodyLandmarks.begin(), state.bodyLandmarks.end(), X.Q.begin(), newState.bodyLandmarks.begin(),
        [](const Point3d& lm, const SOT3& Q) {
            Point3d result;
            result.p = Q.inverse() * lm.p;
            result.id = lm.id;
            return result;
        });

    return newState;
}

VisionMeasurement outputGroupAction(const VIOGroup& X, const VisionMeasurement& measurement) {
    // Check the measurements and transforms are aligned.
    assert(X.Q.size() == measurement.numberOfBearings);
    for (size_t i = 0; i < X.Q.size(); ++i)
        assert(X.id[i] == measurement.bearings[i].id);

    // Transform the measurements
    VisionMeasurement newMeasurements;
    newMeasurements.bearings.resize(measurement.bearings.size());
    newMeasurements.numberOfBearings = measurement.bearings.size();
    transform(measurement.bearings.begin(), measurement.bearings.end(), X.Q.begin(), newMeasurements.bearings.begin(),
        [](const Point3d& y, const SOT3& Q) {
            Point3d result;
            result.p = Q.R().inverse() * y.p;
            result.id = y.id;
            return result;
        });

    return newMeasurements;
}

VIOGroup VIOGroup::operator*(const VIOGroup& other) const {
    VIOGroup result;

    result.A = this->A * other.A;
    result.w = this->w + this->A.R() * other.w;

    // Check the transforms are aligned.
    assert(this->Q.size() == other.Q.size());
    assert(this->id.size() == other.id.size());
    for (size_t i = 0; i < this->Q.size(); ++i)
        assert(this->id[i] == other.id[i]);

    result.Q.resize(this->Q.size());
    transform(this->Q.begin(), this->Q.end(), other.Q.begin(), result.Q.begin(),
        [](const SOT3& Qi1, const SOT3& Qi2) { return Qi1 * Qi2; });
    result.id = this->id;

    return result;
}

VIOGroup VIOGroup::Identity(const vector<int>& id) {
    VIOGroup result;
    result.A = SE3::Identity();
    result.w = Vector3d::Zero();
    result.id = id;
    result.Q.resize(id.size());
    for (SOT3& Qi : result.Q) {
        Qi.setIdentity();
    }
    return result;
}

VIOGroup VIOGroup::inverse() const {
    VIOGroup result;
    result.A = A.inverse();
    result.w = -(A.R().inverse() * w);

    result.Q = Q;
    for_each(result.Q.begin(), result.Q.end(), [](SOT3& Qi) { Qi = Qi.inverse(); });
    result.id = this->id;

    return result;
}

VIOAlgebra VIOAlgebra::operator*(const double& c) const {
    VIOAlgebra result;
    result.U = this->U * c;
    result.u = this->u * c;
    result.W.resize(this->W.size());
    transform(W.begin(), W.end(), result.W.begin(), [&c](const Vector4d& Wi) { return c * Wi; });
    result.id = this->id;

    return result;
}

VIOAlgebra VIOAlgebra::operator-() const {
    VIOAlgebra result;
    result.U = -this->U;
    result.u = -this->u;
    result.W.resize(this->W.size());
    transform(W.begin(), W.end(), result.W.begin(), [](const Vector4d& Wi) { return -Wi; });
    result.id = this->id;

    return result;
}

VIOAlgebra VIOAlgebra::operator+(const VIOAlgebra& other) const {
    VIOAlgebra result;
    result.U = this->U + other.U;
    result.u = this->u + other.u;

    assert(this->id.size() == other.id.size());
    for (size_t i = 0; i < this->id.size(); ++i) {
        assert(this->id[i] == other.id[i]);
    }

    result.W.resize(this->W.size());
    transform(this->W.begin(), this->W.end(), other.W.begin(), result.W.begin(),
        [](const Vector4d& Wi1, const Vector4d& Wi2) { return Wi1 + Wi2; });
    result.id = this->id;

    return result;
}

VIOAlgebra VIOAlgebra::operator-(const VIOAlgebra& other) const { return *this + (-other); }

[[nodiscard]] VIOAlgebra liftVelocity(const VIOManifoldState& state, const IMUVelocity& velocity) {
    VIOAlgebra lift;

    // Set the SE(3) velocity
    lift.U.setZero();
    lift.U.block<3, 1>(0, 0) = velocity.omega;
    lift.U.block<3, 1>(3, 0) = state.velocity;

    // Set the R3 velocity
    lift.u = -velocity.accel + state.gravityDir * GRAVITY_CONSTANT;

    // Set the landmark transform velocities
    const se3vector U_C = state.cameraOffset.inverse().Adjoint() * lift.U;
    lift.W.resize(state.bodyLandmarks.size());
    transform(state.bodyLandmarks.begin(), state.bodyLandmarks.end(), lift.W.begin(), [&U_C](const Point3d& blm) {
        Vector4d result;
        const Vector3d& omega_C = U_C.block<3, 1>(0, 0);
        const Vector3d& v_C = U_C.block<3, 1>(3, 0);
        result.block<3, 1>(0, 0) = omega_C + SO3::skew(blm.p) * v_C / blm.p.squaredNorm();
        result(3) = blm.p.dot(v_C) / blm.p.squaredNorm();
        return result;
    });

    // Set the lift ids
    lift.id.resize(state.bodyLandmarks.size());
    transform(state.bodyLandmarks.begin(), state.bodyLandmarks.end(), lift.id.begin(),
        [](const Point3d& blm) { return blm.id; });

    return lift;
}

[[nodiscard]] VIOGroup liftVelocityDiscrete(
    const VIOManifoldState& state, const IMUVelocity& velocity, const double& dt) {
    // Lift the velocity discretely
    VIOGroup lift;

    // Set the SE(3) velocity
    Matrix<double, 6, 1> AVel;
    AVel << velocity.omega, state.velocity;
    lift.A = SE3::SE3Exp(dt * AVel);

    // Set the R3 velocity
    lift.w =
        state.velocity - lift.A.R() * (state.velocity + dt * (-SO3::skew(velocity.omega) * state.velocity +
                                                                 velocity.accel - state.gravityDir * GRAVITY_CONSTANT));

    // Set the landmark transform velocities
    const int N = state.bodyLandmarks.size();
    const se3vector U_C = state.cameraOffset.inverse().Adjoint() * AVel;
    const SE3 cameraPoseChangeInv = SE3::SE3Exp(-dt * U_C);
    lift.Q.resize(N);
    lift.id.resize(N);
    for (int i = 0; i < N; ++i) {
        const Point3d& blm0 = state.bodyLandmarks[i];
        Point3d blm1;
        blm1.id = blm0.id;
        blm1.p = cameraPoseChangeInv * blm0.p;

        // Find the transform to take blm1 to blm0
        lift.Q[i].R() = SO3::SO3FromVectors(blm1.p.normalized(), blm0.p.normalized());
        lift.Q[i].a() = blm0.p.norm() / blm1.p.norm();
        lift.id[i] = blm1.id;
    }

    return lift;
}

[[nodiscard]] VIOGroup VIOExp(const VIOAlgebra& lambda) {
    VIOGroup result;
    result.A = SE3::SE3Exp(lambda.U);
    result.w = lambda.u;

    result.id = lambda.id;
    result.Q.resize(lambda.W.size());
    transform(lambda.W.begin(), lambda.W.end(), result.Q.begin(), [](const Vector4d& Wi) { return SOT3::SOT3Exp(Wi); });

    return result;
}

[[nodiscard]] VIOAlgebra operator*(const double& c, const VIOAlgebra& lambda) { return lambda * c; }