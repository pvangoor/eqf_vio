#include "testing_utilities.h"

using namespace Eigen;
using namespace std;

VIOState randomStateElement(const vector<int>& ids) {
    VIOState xi;
    xi.pose.R().fromQuaternion(Quaterniond::UnitRandom());
    xi.pose.x().setRandom();
    xi.velocity.setRandom();

    // xi.cameraOffset.setIdentity();
    xi.cameraOffset.R().fromQuaternion(Quaterniond::UnitRandom());
    // xi.cameraOffset.x().setRandom();

    xi.bodyLandmarks.resize(ids.size());
    for (int i = 0; i < ids.size(); ++i) {
        xi.bodyLandmarks[i].p.setRandom();
        xi.bodyLandmarks[i].id = ids[i];
    }

    return xi;
}

VectorXd stateVecDiff(const VIOState& xi1, const VIOState& xi2) {
    // Compute the tangent vector from xi1 to xi2
    VectorXd vecDiff = VectorXd(6 + 3 + 3 * xi1.bodyLandmarks.size());
    vecDiff.setConstant(NAN);
    vecDiff.block<6, 1>(0, 0) = SE3::SE3Log(xi1.pose.inverse() * xi2.pose);
    vecDiff.block<3, 1>(6, 0) = (xi2.velocity - xi1.velocity);
    assert(xi1.bodyLandmarks.size() == xi2.bodyLandmarks.size());
    for (int i = 0; i < xi1.bodyLandmarks.size(); ++i) {
        assert(xi1.bodyLandmarks[i].id == xi2.bodyLandmarks[i].id);
        vecDiff.block<3, 1>(9 + 3 * i, 0) = (xi2.bodyLandmarks[i].p - xi1.bodyLandmarks[i].p);
    }
    assert(!vecDiff.hasNaN());
    return vecDiff;
}

IMUVelocity randomVelocityElement() {
    IMUVelocity vel;
    vel.omega.setRandom();
    vel.accel.setRandom();
    vel.stamp = 0;
    return vel;
}

VIOGroup randomGroupElement(const vector<int>& ids) {
    VIOGroup X;
    X.A.R().fromQuaternion(Quaterniond::UnitRandom());
    X.A.x().setRandom();
    X.w.setRandom();
    X.id = ids;
    X.Q.resize(ids.size());
    for (int i = 0; i < ids.size(); ++i) {
        X.Q[i].R().fromQuaternion(Quaterniond::UnitRandom());
        X.Q[i].a() = 5.0 * (double)rand() / RAND_MAX + 1.0;
    }

    return X;
}

double logNorm(const VIOGroup& X) {
    double result = 0;
    result += SE3::SE3Log(X.A).norm();
    result += X.w.norm();
    for (const SOT3& Qi : X.Q) {
        result += SOT3::SOT3Log(Qi).norm();
    }
    return result;
}

double stateDistance(const VIOState& xi1, const VIOState& xi2) {
    double dist = 0;
    dist += SE3::SE3Log(xi1.pose.inverse() * xi2.pose).norm();
    dist += (xi1.velocity - xi2.velocity).norm();
    assert(xi1.bodyLandmarks.size() == xi2.bodyLandmarks.size());
    for (int i = 0; i < xi1.bodyLandmarks.size(); ++i) {
        assert(xi1.bodyLandmarks[i].id == xi2.bodyLandmarks[i].id);
        dist += (xi1.bodyLandmarks[i].p - xi2.bodyLandmarks[i].p).norm();
    }
    return dist;
}

double stateDistance(const VIOManifoldState& xi1, const VIOManifoldState& xi2) {
    double dist = 0;
    dist += (xi1.gravityDir - xi2.gravityDir).norm();
    dist += (xi1.velocity - xi2.velocity).norm();
    assert(xi1.bodyLandmarks.size() == xi2.bodyLandmarks.size());
    for (int i = 0; i < xi1.bodyLandmarks.size(); ++i) {
        assert(xi1.bodyLandmarks[i].id == xi2.bodyLandmarks[i].id);
        dist += (xi1.bodyLandmarks[i].p - xi2.bodyLandmarks[i].p).norm();
    }
    return dist;
}

VisionMeasurement randomVisionMeasurement(const vector<int>& ids) {
    VisionMeasurement result;
    result.stamp = 0.0;
    result.numberOfBearings = ids.size();
    result.bearings.resize(ids.size());
    for (int i = 0; i < ids.size(); ++i) {
        result.bearings[i].id = ids[i];
        do {
            result.bearings[i].p.setRandom();
        } while (result.bearings[i].p.norm() < 1e-4);
        result.bearings[i].p.normalize();
    }
    return result;
}

double measurementDistance(const VisionMeasurement& y1, const VisionMeasurement& y2) {
    double dist = 0;
    assert(y1.bearings.size() == y2.bearings.size());
    for (int i = 0; i < y1.bearings.size(); ++i) {
        assert(y1.bearings[i].id == y2.bearings[i].id);
        dist += (y1.bearings[i].p - y2.bearings[i].p).norm();
    }
    return dist;
}