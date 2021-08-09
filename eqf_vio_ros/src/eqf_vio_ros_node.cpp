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

#include "geometry_msgs/PoseStamped.h"
#include "ros/ros.h"
#include "sensor_msgs/Imu.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/image_encodings.h"
#include "cv_bridge/cv_bridge.h"

#include "yaml-cpp/yaml.h"

#include "eqf_vio/VIOFilter.h"
#include "eqf_vio/VIOFilterSettings.h"

#include "GIFT/PointFeatureTracker.h"

struct CallbackStruct {
    GIFT::PointFeatureTracker featureTracker;
    VIOFilter filter;
    ros::Publisher pose_publisher;

    void callbackImu(const sensor_msgs::Imu& msg);
    void callbackImage(const sensor_msgs::ImageConstPtr& msg);
};

VisionMeasurement convertGIFTFeatures(const std::vector<GIFT::Feature>& GIFTFeatures, const double& stamp);

int main(int argc, char* argv[]) {
    ros::init(argc, argv, "eqf_vio");
    ros::NodeHandle nh;

    // Load configuration information
    std::string eqf_vioConfig_fname, giftConfig_fname, cameraIntrinsics_fname;
    bool configuration_flag = nh.getParam("/eqf_vio/eqf_vio_config", eqf_vioConfig_fname) &&
                              nh.getParam("/eqf_vio/gift_config", giftConfig_fname) &&
                              nh.getParam("/eqf_vio/camera_intrinsics", cameraIntrinsics_fname);
    ROS_ASSERT(configuration_flag);
    const YAML::Node eqf_vioConfig = YAML::LoadFile(eqf_vioConfig_fname);
    const YAML::Node giftConfig = YAML::LoadFile(giftConfig_fname);

    // Initialise the filter, camera, feature tracker
    CallbackStruct cbSys;
    VIOFilter::Settings filterSettings(eqf_vioConfig["eqf"]);
    cbSys.filter = VIOFilter(filterSettings);
    ROS_INFO_STREAM("EqF configured from\n" << eqf_vioConfig_fname);

    GIFT::PinholeCamera camera = GIFT::PinholeCamera(cv::String(cameraIntrinsics_fname));
    ROS_INFO_STREAM("Camera configured from\n" << cameraIntrinsics_fname);

    cbSys.featureTracker = GIFT::PointFeatureTracker(camera);
    cbSys.featureTracker.settings.configure(giftConfig["GIFT"]);
    ROS_INFO_STREAM("GIFT configured from\n" << giftConfig_fname);
    

    // Set up publishers and subscribers
    cbSys.pose_publisher = nh.advertise<geometry_msgs::PoseStamped>("eqf_vio/pose", 50);

    ros::Subscriber subImu = nh.subscribe("/eqf_vio/imu", 100, &CallbackStruct::callbackImu, &cbSys);
    ros::Subscriber subImage = nh.subscribe("/eqf_vio/image", 5, &CallbackStruct::callbackImage, &cbSys);

    ros::spin();
    return 0;
}

void CallbackStruct::callbackImu(const sensor_msgs::Imu& msg){
    // Convert to IMU measurement

    ROS_DEBUG("IMU Message Received.");

    IMUVelocity imuVel;
    imuVel.stamp = msg.header.stamp.toSec();
    imuVel.omega << msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z;
    imuVel.accel << msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z;

    this->filter.processIMUData(imuVel);
}

void CallbackStruct::callbackImage(const sensor_msgs::ImageConstPtr& msg) {

    ROS_DEBUG("Image Message Received.");

    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_8UC1);

    featureTracker.processImage(cv_ptr->image);
    const std::vector<GIFT::Feature> features = featureTracker.outputFeatures();
    const VisionMeasurement visionData = convertGIFTFeatures(features, msg->header.stamp.toSec());

    filter.processVisionData(visionData);

    // Write pose message

    VIOState estimatedState = filter.stateEstimate();
    // estimatedState.pose.R(); estimatedState.pose.x();
    geometry_msgs::PoseStamped poseMsg;
    poseMsg.header.frame_id = "map";
    poseMsg.header.stamp = msg->header.stamp;

    const Eigen::Quaterniond& attitude = estimatedState.pose.R().asQuaternion();
    const Eigen::Vector3d& position = estimatedState.pose.x();
    poseMsg.pose.orientation.w = attitude.w();
    poseMsg.pose.orientation.x = attitude.x();
    poseMsg.pose.orientation.y = attitude.y();
    poseMsg.pose.orientation.z = attitude.z();
    poseMsg.pose.position.x = position.x();
    poseMsg.pose.position.y = position.y();
    poseMsg.pose.position.z = position.z();

    pose_publisher.publish(poseMsg);
}

VisionMeasurement convertGIFTFeatures(const std::vector<GIFT::Feature>& GIFTFeatures, const double& stamp) {
    VisionMeasurement measurement;
    measurement.stamp = stamp;
    measurement.numberOfBearings = GIFTFeatures.size();
    measurement.bearings.resize(GIFTFeatures.size());
    std::transform(GIFTFeatures.begin(), GIFTFeatures.end(), measurement.bearings.begin(), [](const GIFT::Feature& f) {
        Point3d pt;
        pt.p = f.sphereCoordinates();
        pt.id = f.idNumber;
        return pt;
    });
    return measurement;
}