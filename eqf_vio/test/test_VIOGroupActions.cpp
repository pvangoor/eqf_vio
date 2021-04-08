#include "eigen3/Eigen/Dense"
#include "eqf_vio/VIOGroup.h"
#include "eqf_vio/VIOState.h"
#include "eqf_vio/VisionMeasurement.h"
#include "testing_utilities.h"
#include "gtest/gtest.h"

using namespace Eigen;
using namespace std;

TEST(VIOActionTest, StateAction) {
    vector<int> ids = {0, 1, 2, 3, 4};
    const VIOGroup groupId = VIOGroup::Identity(ids);
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOGroup X1 = randomGroupElement(ids);
        const VIOGroup X2 = randomGroupElement(ids);

        const VIOState xi0 = randomStateElement(ids);

        // Check the distance function works
        const double dist00 = stateDistance(xi0, xi0);
        EXPECT_LE(dist00, NEAR_ZERO);

        // Check action identity
        const VIOState xi0_id = stateGroupAction(groupId, xi0);
        const double dist0id = stateDistance(xi0_id, xi0);
        EXPECT_LE(dist0id, NEAR_ZERO);

        // Check action compatibility
        const VIOState xi1 = stateGroupAction(X2, stateGroupAction(X1, xi0));
        const VIOState xi2 = stateGroupAction(X1 * X2, xi0);
        const double dist12 = stateDistance(xi1, xi2);
        EXPECT_LE(dist12, NEAR_ZERO);
    }
}

TEST(VIOActionTest, OutputAction) {
    vector<int> ids = {0, 1, 2, 3, 4};
    const VIOGroup groupId = VIOGroup::Identity(ids);
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOGroup X1 = randomGroupElement(ids);
        const VIOGroup X2 = randomGroupElement(ids);

        VisionMeasurement y0 = randomVisionMeasurement(ids);

        // Check the distance function works
        const double dist00 = measurementDistance(y0, y0);
        EXPECT_LE(dist00, NEAR_ZERO);

        // Check action identity
        const VisionMeasurement y0_id = outputGroupAction(groupId, y0);
        const double dist0id = measurementDistance(y0_id, y0);
        EXPECT_LE(dist0id, NEAR_ZERO);

        // Check action compatibility
        const VisionMeasurement y1 = outputGroupAction(X2, outputGroupAction(X1, y0));
        const VisionMeasurement y2 = outputGroupAction(X1 * X2, y0);
        const double dist12 = measurementDistance(y1, y2);
        EXPECT_LE(dist12, NEAR_ZERO);
    }
}

TEST(VIOActionTest, OutputEquivariance) {
    vector<int> ids = {0, 1, 2, 3, 4};
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOGroup X = randomGroupElement(ids);
        const VIOState xi0 = randomStateElement(ids);

        // Check the state/output equivariance
        const VisionMeasurement y1 = measureSystemState(stateGroupAction(X, xi0));
        const VisionMeasurement y2 = outputGroupAction(X, measureSystemState(xi0));
        const double dist12 = measurementDistance(y1, y2);
        EXPECT_LE(dist12, NEAR_ZERO);
    }
}
