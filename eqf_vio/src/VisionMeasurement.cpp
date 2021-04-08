#include "eqf_vio/VisionMeasurement.h"
#include "eqf_vio/VIOState.h"

using namespace std;
using namespace Eigen;

VectorXd outputCoordinateChart(const VisionMeasurement& y, const VisionMeasurement& y0) {
    const int N = y.numberOfBearings;
    assert(y.bearings.size() == y0.bearings.size());
    VectorXd delta(2 * N);
    for (int i = 0; i < N; ++i) {
        assert(y.bearings[i].id == y0.bearings[i].id);
        const Vector2d yiCoords = stereoSphereChart(y.bearings[i].p, y0.bearings[i].p);
        delta.block<2, 1>(2 * i, 0) = yiCoords;
    }
    return delta;
}

VisionMeasurement outputCoordinateChartInv(const Eigen::VectorXd& delta, const VisionMeasurement& y0) {
    const int N = y0.numberOfBearings;
    assert(delta.size() == 2 * N);
    VisionMeasurement y;
    y.stamp = y0.stamp;
    y.numberOfBearings = y0.numberOfBearings;
    y.bearings.resize(N);

    for (int i = 0; i < N; ++i) {
        y.bearings[i].p = stereoSphereChartInv(delta.block<2, 1>(2 * i, 0), y0.bearings[i].p);
        y.bearings[i].id = y0.bearings[i].id;
    }

    return y;
}
