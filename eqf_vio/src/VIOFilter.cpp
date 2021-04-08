#include <numeric>

#include "eigen3/unsupported/Eigen/MatrixFunctions"

#include "eqf_vio/EqFMatrices.h"
#include "eqf_vio/VIOFilter.h"
#include "eqf_vio/VIOFilterSettings.h"

using namespace Eigen;
using namespace std;

void removeRows(MatrixXd& mat, int startRow, int numRows) {
    int rows = mat.rows();
    int cols = mat.cols();

    assert(startRow + numRows <= rows);
    mat.block(startRow, 0, rows - numRows - startRow, cols) =
        mat.block(startRow + numRows, 0, rows - numRows - startRow, cols);
    mat.conservativeResize(rows - numRows, NoChange);
}

void removeCols(MatrixXd& mat, int startCol, int numCols) {
    int rows = mat.rows();
    int cols = mat.cols();

    assert(startCol + numCols <= cols);
    mat.block(0, startCol, rows, cols - numCols - startCol) =
        mat.block(0, startCol + numCols, rows, cols - numCols - startCol);
    mat.conservativeResize(NoChange, cols - numCols);
}

VIOFilter::VIOFilter(const AuxiliaryFilterData& auxiliaryData) { setAuxiliaryData(auxiliaryData); }

VIOFilter::VIOFilter(const AuxiliaryFilterData& auxiliaryData, const VIOFilter::Settings& settings) {
    setAuxiliaryData(auxiliaryData);
    this->settings = make_unique<VIOFilter::Settings>(settings);
    Sigma.block<3, 3>(0, 0) = Matrix3d::Identity() * settings.initialBiasOmegaVariance;
    Sigma.block<3, 3>(3, 3) = Matrix3d::Identity() * settings.initialBiasAccelVariance;
    Sigma.block<2, 2>(6, 6) = Matrix2d::Identity() * settings.initialGravityVariance;
    Sigma.block<3, 3>(8, 8) = Matrix3d::Identity() * settings.initialVelocityVariance;
}

VIOFilter::VIOFilter(const VIOFilter::Settings& settings) {
    this->settings = make_unique<VIOFilter::Settings>(settings);
    Sigma.block<3, 3>(0, 0) = Matrix3d::Identity() * settings.initialBiasOmegaVariance;
    Sigma.block<3, 3>(3, 3) = Matrix3d::Identity() * settings.initialBiasAccelVariance;
    Sigma.block<2, 2>(6, 6) = Matrix2d::Identity() * settings.initialGravityVariance;
    Sigma.block<3, 3>(8, 8) = Matrix3d::Identity() * settings.initialVelocityVariance;
    xi0.pose.setIdentity();
    xi0.velocity.setZero();
    xi0.cameraOffset = settings.cameraOffset;

    inputBias.block<3, 1>(0, 0) = settings.initialOmegaBias;
    inputBias.block<3, 1>(3, 0) = settings.initialAccelBias;
}

void VIOFilter::setAuxiliaryData(const AuxiliaryFilterData& auxiliaryData) {
    auxData = auxiliaryData;
    xi0.pose.R() = SO3(auxiliaryData.initialAttitude);
    xi0.pose.x() = auxiliaryData.initialPosition;
    xi0.velocity.setZero();
    initialisedFlag = true;

    xi0.cameraOffset = auxiliaryData.cameraOffset;
}

void VIOFilter::reset() {
    xi0 = VIOState();
    X = VIOGroup::Identity();
    Sigma = Eigen::MatrixXd::Identity(11, 11);

    currentTime = -1;
    currentVelocity = IMUVelocity::Zero();
}

void VIOFilter::setInertialPoints(const std::vector<Point3d>& inertialPoints) {
    const int N = inertialPoints.size();

    // Set up the state and transforms
    xi0.bodyLandmarks.resize(N);
    X.Q.resize(N);
    X.id.resize(N);
    const SE3 inertialToCameraTF = (xi0.pose * xi0.cameraOffset).inverse();
    for (int i = 0; i < N; ++i) {
        // Set identity transform
        X.id[i] = inertialPoints[i].id;
        X.Q[i].setIdentity();

        // Compute the point in the camera frame
        const Vector3d& p_i = inertialPoints[i].p;
        const Vector3d q_i = inertialToCameraTF * p_i;
        xi0.bodyLandmarks[i].p = q_i;
        xi0.bodyLandmarks[i].id = inertialPoints[i].id;
    }

    // Update Sigma
    MatrixXd SigmaNew =
        Eigen::MatrixXd::Identity(SIGMA_BASE_SIZE + 3 * N, SIGMA_BASE_SIZE + 3 * N) * settings->initialPointVariance;
    SigmaNew.block<SIGMA_BASE_SIZE, SIGMA_BASE_SIZE>(0, 0) = Sigma;
    Sigma = SigmaNew;
}

void VIOFilter::processIMUData(const IMUVelocity& imuVelocity) {
    IMUVelocity unbiasedVelocity = imuVelocity - inputBias;
    if (!initialisedFlag) {
        initialiseFromIMUData(unbiasedVelocity);
    }

    integrateUpToTime(imuVelocity.stamp, !settings->fastRiccati);

    // Update the velocity and time
    currentVelocity = unbiasedVelocity;
    currentTime = imuVelocity.stamp;
}

void VIOFilter::initialiseFromIMUData(const IMUVelocity& imuVelocity) {
    xi0.pose.setIdentity();
    xi0.velocity.setZero();
    initialisedFlag = true;

    // Compute the attitude from the gravity vector
    // accel \approx g R^\top e_3,
    // e_3 \approx R accel.normalized()

    const Vector3d& approxGravity = imuVelocity.accel.normalized();
    xi0.pose.R() = SO3::SO3FromVectors(approxGravity, Vector3d::Unit(2));
}

bool VIOFilter::integrateUpToTime(const double& newTime, const bool doRiccati) {
    if (currentTime < 0)
        return false;

    const double dt = newTime - currentTime;
    if (dt <= 0)
        return false;

    accumulatedTime += dt;
    accumulatedVelocity = accumulatedVelocity + currentVelocity * dt;

    const int N = xi0.bodyLandmarks.size();
    const VIOState currentState = stateEstimate();

    if (doRiccati) {
        // Lift the velocity and compute the Riccati process matrices
        MatrixXd PMat = MatrixXd::Identity(Sigma.rows(), Sigma.cols());
        PMat.block<3, 3>(0, 0) *= settings->biasOmegaProcessVariance;
        PMat.block<3, 3>(3, 3) *= settings->biasAccelProcessVariance;
        PMat.block<2, 2>(6, 6) *= settings->gravityProcessVariance;
        PMat.block<3, 3>(8, 8) *= settings->velocityProcessVariance;
        PMat.block(11, 11, 3 * N, 3 * N) *= settings->pointProcessVariance;

        const MatrixXd A0t = EqFStateMatrixA_euclid(X, xi0, accumulatedVelocity * (1.0 / accumulatedTime));

        // Compute the Riccati velocity matrix
        const MatrixXd Bt = EqFInputMatrixB_euclid(X, xi0);
        Matrix<double, 6, 6> R = Matrix<double, 6, 6>::Identity();
        R.block<3, 3>(0, 0) *= settings->velOmegaVariance;
        R.block<3, 3>(3, 3) *= settings->velAccelVariance;

        // Create Bias filter matrices
        MatrixXd A0tBiased = MatrixXd::Zero(A0t.rows() + 6, A0t.cols() + 6);
        A0tBiased.block(6, 6, A0t.rows(), A0t.cols()) = A0t;
        A0tBiased.block(6, 0, Bt.rows(), Bt.cols()) = -Bt;
        // const MatrixXd A0tBiasedExp = (A0tBiased * dt).exp();
        const MatrixXd A0tBiasedExp =
            MatrixXd::Identity(A0tBiased.rows(), A0tBiased.cols()) + A0tBiased * accumulatedTime;
        MatrixXd BtBiased = MatrixXd::Zero(Bt.rows() + 6, Bt.cols());
        BtBiased.block(6, 0, Bt.rows(), Bt.cols()) = Bt;

        // Sigma += dt * (PMat + Bt * R * Bt.transpose() + A0tBiased * Sigma + Sigma * A0tBiased.transpose());
        Sigma = accumulatedTime * (PMat + BtBiased * R * BtBiased.transpose()) +
                A0tBiasedExp * Sigma * A0tBiasedExp.transpose();
        assert(!Sigma.hasNaN());

        accumulatedVelocity = IMUVelocity::Zero();
        accumulatedTime = 0.0;
    }

    // Integrate the equations
    if (settings->useDiscreteVelocityLift) {
        VIOGroup liftedVelocity = liftVelocityDiscrete(currentState, currentVelocity, dt);
        X = X * liftedVelocity;
    } else {
        VIOAlgebra liftedVelocity = liftVelocity(currentState, currentVelocity);
        X = X * VIOExp(dt * liftedVelocity);
    }

    assert(!X.A.x().hasNaN());

    currentTime = newTime;
    return true;
}

VisionMeasurement VIOFilter::matchMeasurementsToState(const VisionMeasurement& measurement) const {
    // Rearrange the measurements into a new vector so that the ids match the state ids,
    // up to the point where there are new measurements. All new measurements should be
    // at the end of the measurement vector

    const vector<int>& stateIds = X.id;

    VisionMeasurement matched;
    matched.stamp = measurement.stamp;
    matched.numberOfBearings = measurement.numberOfBearings;
    matched.bearings.resize(measurement.bearings.size());
    int newLandmarkPos = stateIds.size() - 1;
    for (const Point3d yi : measurement.bearings) {
        const auto it = find(stateIds.begin(), stateIds.end(), yi.id);
        const int idx = (it != stateIds.end()) ? distance(stateIds.begin(), it) : ++newLandmarkPos;
        matched.bearings[idx] = yi;
    }

    return matched;
}

void VIOFilter::processVisionData(const VisionMeasurement& measurement) {
    // Use the stored velocity input to bring the filter up to the current timestamp
    bool integrationFlag = integrateUpToTime(measurement.stamp);
    if (!integrationFlag || !initialisedFlag)
        return;

    // Ensure the measurements are sorted with ascending ids
    assert(is_sorted(measurement.bearings.begin(), measurement.bearings.end(),
        [](const Point3d& meas1, const Point3d& meas2) { return meas1.id <= meas2.id; }));

    removeOldLandmarks(measurement);
    VisionMeasurement matchedMeasurement = matchMeasurementsToState(measurement);

    assert(matchedMeasurement.bearings.size() >= X.id.size());
    for (int i = X.id.size() - 1; i >= 0; --i) {
        assert(matchedMeasurement.bearings[i].id == X.id[i]);
    }

    removeOutliers(matchedMeasurement);
    addNewLandmarks(matchedMeasurement);

    assert(matchedMeasurement.bearings.size() == X.id.size());
    for (int i = X.id.size() - 1; i >= 0; --i) {
        assert(matchedMeasurement.bearings[i].id == X.id[i]);
    }

    if (matchedMeasurement.bearings.empty())
        return;

    // --------------------------
    // Compute the EqF innovation
    // --------------------------
    const VisionMeasurement originMeasurement = measureSystemState(xi0);
    const VisionMeasurement errorMeasurement = outputGroupAction(X.inverse(), matchedMeasurement);
    const VectorXd delta = outputCoordinateChart(errorMeasurement, originMeasurement);
    const MatrixXd C0 = EqFOutputMatrixC_euclid(xi0);
    const int N = xi0.bodyLandmarks.size();
    const MatrixXd QMat = settings->measurementVariance * MatrixXd::Identity(2 * N, 2 * N);

    // Create the bias matrix
    MatrixXd C0Biased = MatrixXd::Zero(C0.rows(), C0.cols() + 6);
    C0Biased.block(0, 6, C0.rows(), C0.cols()) = C0;

    // Use the discrete update form
    const MatrixXd S = C0Biased * Sigma * C0Biased.transpose() + QMat;
    const MatrixXd K = Sigma * C0Biased.transpose() * S.inverse();

    const VectorXd baseInnovationBiased = K * delta;
    const VectorXd& baseInnovationEqF = baseInnovationBiased.block(6, 0, baseInnovationBiased.rows() - 6, 1);
    const VectorXd& baseInnovationBias = baseInnovationBiased.block<6, 1>(0, 0);
    VIOGroup Delta;
    VIOAlgebra DeltaAlg;
    if (settings->useInnovationLift) {
        VectorXd Gamma = bundleLift(baseInnovationEqF, xi0, X, Sigma.block(6, 6, 5 + 3 * N, 5 + 3 * N));
        if (settings->useDiscreteInnovationLift) {
            Delta = liftTotalSpaceInnovationDiscrete(Gamma, xi0);
        } else {
            Delta = VIOExp(liftTotalSpaceInnovation(Gamma, xi0));
        }
    } else {
        Delta = VIOExp(liftInnovation(baseInnovationEqF, xi0));
    }

    inputBias = inputBias + baseInnovationBias;
    X = Delta * X;
    Sigma = Sigma - K * C0Biased * Sigma;

    assert(!Sigma.hasNaN());
    assert(!X.A.x().hasNaN());
    // assert(Sigma.eigenvalues().real().minCoeff() > 0);
}

VIOState VIOFilter::stateEstimate() const { return stateGroupAction(this->X, this->xi0); }

Eigen::MatrixXd VIOFilter::stateCovariance() const {
    // TODO: Propagate Sigma to the local tangent space
    return Sigma;
}

std::ostream& operator<<(std::ostream& os, const VIOFilter& filter) {
    const Vector3d position = filter.xi0.pose.x();
    const Quaterniond attitude = filter.xi0.pose.R().asQuaternion();
    os << position.x() << ", " << position.y() << ", " << position.z() << ", ";
    os << attitude.w() << ", " << attitude.x() << ", " << attitude.y() << ", " << attitude.z() << ", ";
    os << filter.xi0.velocity.x() << ", " << filter.xi0.velocity.y() << ", " << filter.xi0.velocity.z() << ", ";

    const Vector3d translation = filter.X.A.x();
    const Quaterniond rotation = filter.X.A.R().asQuaternion();
    os << translation.x() << ", " << translation.y() << ", " << translation.z() << ", ";
    os << rotation.w() << ", " << rotation.x() << ", " << rotation.y() << ", " << rotation.z() << ", ";
    os << filter.X.w.x() << ", " << filter.X.w.y() << ", " << filter.X.w.z() << ", ";

    const int N = filter.xi0.bodyLandmarks.size();
    assert(N == filter.X.Q.size());
    os << N;
    for (int i = 0; i < N; ++i) {
        const Point3d& blm = filter.xi0.bodyLandmarks[i];
        assert(blm.id == filter.X.id[i]);
        os << ", " << blm.id << ", " << blm.p.x() << ", " << blm.p.y() << ", " << blm.p.z();
        const Quaterniond QR = filter.X.Q[i].R().asQuaternion();
        const double Qa = filter.X.Q[i].a();
        os << ", " << QR.w() << ", " << QR.x() << ", " << QR.y() << ", " << QR.z() << ", " << Qa;
    }

    assert(filter.Sigma.cols() == SIGMA_BASE_SIZE + 3 * N);
    assert(filter.Sigma.rows() == SIGMA_BASE_SIZE + 3 * N);
    os << ", " << filter.Sigma.format(IOFormat(-1, 0, ", ", ", "));

    return os;
}

double VIOFilter::getTime() const { return currentTime; }

void VIOFilter::addNewLandmarks(const VisionMeasurement& measurement) {
    // Grab all the new landmarks
    vector<Point3d> newLandmarks(measurement.bearings.size());
    const auto newLandmarkEnd = copy_if(
        measurement.bearings.begin(), measurement.bearings.end(), newLandmarks.begin(), [&](const Point3d& meas) {
            return none_of(X.id.begin(), X.id.end(), [&meas](const int& i) { return i == meas.id; });
        });
    newLandmarks.resize(distance(newLandmarks.begin(), newLandmarkEnd));
    if (newLandmarks.empty())
        return;

    // Initialise all landmarks to the median scene depth
    const vector<Point3d> landmarks = this->stateEstimate().bodyLandmarks;
    vector<double> depthsSquared(landmarks.size());
    transform(landmarks.begin(), landmarks.end(), depthsSquared.begin(),
        [](const Point3d& blm) { return blm.p.squaredNorm(); });
    const auto midway = depthsSquared.begin() + depthsSquared.size() / 2;
    nth_element(depthsSquared.begin(), midway, depthsSquared.end());
    double medianDepth = settings->initialSceneDepth;
    if (!(midway == depthsSquared.end())) {
        medianDepth = pow(*midway, 0.5);
    }
    for_each(newLandmarks.begin(), newLandmarks.end(), [&medianDepth](Point3d& blm) { blm.p *= medianDepth; });

    // ----------------------------------
    // Add all the landmarks to the state
    // ----------------------------------
    xi0.bodyLandmarks.insert(xi0.bodyLandmarks.end(), newLandmarks.begin(), newLandmarks.end());

    vector<int> newIds(newLandmarks.size());
    transform(newLandmarks.begin(), newLandmarks.end(), newIds.begin(), [](const Point3d& blm) { return blm.id; });
    X.id.insert(X.id.end(), newIds.begin(), newIds.end());

    vector<SOT3> newTransforms(newLandmarks.size());
    for (SOT3& newTf : newTransforms) {
        newTf.setIdentity();
    }
    X.Q.insert(X.Q.end(), newTransforms.begin(), newTransforms.end());

    const int newN = newLandmarks.size();
    const int ogSize = Sigma.rows();
    Sigma.conservativeResize(ogSize + 3 * newN, ogSize + 3 * newN);
    Sigma.block(ogSize, 0, 3 * newN, ogSize).setZero();
    Sigma.block(0, ogSize, ogSize, 3 * newN).setZero();
    Sigma.block(ogSize, ogSize, 3 * newN, 3 * newN) =
        MatrixXd::Identity(3 * newN, 3 * newN) * settings->initialPointVariance;
}

void VIOFilter::removeOldLandmarks(const VisionMeasurement& measurement) {
    // Determine which indices have been lost
    vector<int> lostIndices(X.id.size());
    iota(lostIndices.begin(), lostIndices.end(), 0);
    if (lostIndices.empty())
        return;

    vector<int> measurementIds(measurement.bearings.size());
    transform(measurement.bearings.begin(), measurement.bearings.end(), measurementIds.begin(),
        [](const Point3d& meas) { return meas.id; });

    const auto lostIndicesEnd = remove_if(lostIndices.begin(), lostIndices.end(), [&](const int& lidx) {
        const int& oldId = X.id[lidx];
        return any_of(
            measurementIds.begin(), measurementIds.end(), [&oldId](const int& measId) { return measId == oldId; });
    });
    lostIndices.erase(lostIndicesEnd, lostIndices.end());

    if (lostIndices.empty())
        return;

    // Remove the origin state and transforms and Sigma bits corresponding to these indices.
    reverse(lostIndices.begin(), lostIndices.end()); // Should be in descending order now
    for (const int li : lostIndices) {
        removeLandmarkAtIndex(li);
    }
}

void VIOFilter::removeLandmarkAtIndex(const int& idx) {
    xi0.bodyLandmarks.erase(xi0.bodyLandmarks.begin() + idx);
    X.id.erase(X.id.begin() + idx);
    X.Q.erase(X.Q.begin() + idx);
    removeRows(Sigma, SIGMA_BASE_SIZE + 3 * idx, 3);
    removeCols(Sigma, SIGMA_BASE_SIZE + 3 * idx, 3);
}

void VIOFilter::removeOutliers(VisionMeasurement& measurement) {
    const VIOState xiHat = stateEstimate();
    const VisionMeasurement yHat = measureSystemState(xiHat);
    // Remove if the difference between the true and expected measurement exceeds a threshold
    assert(measurement.bearings.size() >= yHat.bearings.size());
    for (int i = yHat.bearings.size() - 1; i >= 0; --i) {
        assert(measurement.bearings[i].id == yHat.bearings[i].id);
        const double bearingError = (measurement.bearings[i].p - yHat.bearings[i].p).norm();
        if (bearingError > settings->outlierThreshold) {
            removeLandmarkAtIndex(i);
            measurement.bearings.erase(measurement.bearings.begin() + i);
            --measurement.numberOfBearings;
        }
    }
}