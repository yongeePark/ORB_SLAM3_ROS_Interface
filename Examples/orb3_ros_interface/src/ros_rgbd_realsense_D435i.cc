/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include <signal.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>

#include <condition_variable>

#include <opencv2/core/core.hpp>

#include <librealsense2/rs.hpp>
#include "librealsense2/rsutil.h"

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <librealsense2/rs_advanced_mode.hpp> // this is for d435i Auto Exposure
// #include <librealsense2/rs_option.h>


#include <pcl/point_cloud.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>

#include <System.h>

using namespace std;

void DrawFeature(cv::Mat& im_feature, const cv::Mat im,std::vector<cv::KeyPoint> keypoints, float imageScale, vector<bool> mvbVO,vector<bool> mvbMap);
void PublishPointCloud(vector<Eigen::Matrix<float,3,1>>& global_points, vector<Eigen::Matrix<float,3,1>>& local_points,
ros::Publisher& global_pc_pub, ros::Publisher& local_pc_pub);
bool b_continue_session;

void exit_loop_handler(int s){
    cout << "Finishing session" << endl;
    b_continue_session = false;

}

rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams);
bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev);

void interpolateData(const std::vector<double> &vBase_times,
                     std::vector<rs2_vector> &vInterp_data, std::vector<double> &vInterp_times,
                     const rs2_vector &prev_data, const double &prev_time);

rs2_vector interpolateMeasure(const double target_time,
                              const rs2_vector current_data, const double current_time,
                              const rs2_vector prev_data, const double prev_time);

static rs2_option get_sensor_option(const rs2::sensor& sensor)
{
    // Sensors usually have several options to control their properties
    //  such as Exposure, Brightness etc.

    std::cout << "Sensor supports the following options:\n" << std::endl;

    // The following loop shows how to iterate over all available options
    // Starting from 0 until RS2_OPTION_COUNT (exclusive)
    for (int i = 0; i < static_cast<int>(RS2_OPTION_COUNT); i++)
    {
        rs2_option option_type = static_cast<rs2_option>(i);
        //SDK enum types can be streamed to get a string that represents them
        std::cout << "  " << i << ": " << option_type;

        // To control an option, use the following api:

        // First, verify that the sensor actually supports this option
        if (sensor.supports(option_type))
        {
            std::cout << std::endl;

            // Get a human readable description of the option
            const char* description = sensor.get_option_description(option_type);
            std::cout << "       Description   : " << description << std::endl;

            // Get the current value of the option
            float current_value = sensor.get_option(option_type);
            std::cout << "       Current Value : " << current_value << std::endl;

            //To change the value of an option, please follow the change_sensor_option() function
        }
        else
        {
            std::cout << " is not supported" << std::endl;
        }
    }

    uint32_t selected_sensor_option = 0;
    return static_cast<rs2_option>(selected_sensor_option);
}

int main(int argc, char **argv) {
    // debug
    /*
    cout << endl
             << "Number of arguments : " << argc << endl
             << argv[0] << endl
             << argv[1] << endl 
             << argv[2] << endl
             << argv[3] << endl
             << argv[4] << endl
             << "End of arguments" <<endl
             << "Usage: ./mono_inertial_realsense_D435i path_to_vocabulary path_to_settings (trajectory_file_name)"
             << endl;
    */

    // if (argc < 3 || argc > 4) {
    if (argc < 3 ) { 
        cerr << endl
             << "Number of arguments : " << argc << endl
             << argv[0] << endl
             << argv[1] << endl 
             << argv[2] << endl
             << argv[3] << endl
             << argv[4] << endl
             << "End of arguments" <<endl
             << "Usage: ./mono_inertial_realsense_D435i path_to_vocabulary path_to_settings (trajectory_file_name)"
             << endl;
        return 1;
    }
    // ros

    ros::init(argc, argv,"ros_rgbd_realsense");
    ros::NodeHandle nh;
    ros::NodeHandle nh_param("~");
    ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("orb_pose",1);
    ros::Publisher odom_pub = nh.advertise<nav_msgs::Odometry>("orb_odom",1);
    ros::Publisher global_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/ORB3/globalmap",1);
    ros::Publisher  local_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/ORB3/localmap",1);

    bool enable_pangolin;
    if (!nh_param.getParam("/rgbd_realsense/enable_pangolin",enable_pangolin))
    {
        std::cout<<"It has not been decided whether to use Pangolin."<<std::endl
        <<"shut down the program"<<std::endl;
        return 1;
    }
    bool enable_auto_exposure;
    if (!nh_param.getParam("/rgbd_realsense/enable_auto_exposure",enable_auto_exposure))
    {
        std::cout<<"It has not been decided whether to use auto_exposure."<<std::endl
        <<"shut down the program"<<std::endl;
        return 1;
    }
    int ae_meanIntensitySetPoint;
    if(enable_auto_exposure)
    {
    if (!nh_param.getParam("/rgbd_realsense/ae_meanIntensitySetPoint",ae_meanIntensitySetPoint))
    {
        std::cout<<"It has not been decided the number of the mean Intensity Set Point."<<std::endl
        <<"shut down the program"<<std::endl;
        return 1;
    }
    }

    // fps
    int camera_fps;
    if (!nh_param.getParam("/rgbd_realsense/camera_fps",camera_fps))
    {
        std::cout<<"Please set camera_fps"<<std::endl
        <<"shut down the program"<<std::endl;
        return 1;
    }
    
    int camera_exposure;
    nh_param.getParam("/rgbd_realsense/exposure",camera_exposure);

    // for image handling
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub_image         = it.advertise("/camera/color/image_raw", 1);
    image_transport::Publisher pub_image_feature = it.advertise("/orb3_feature_image", 1);
    image_transport::Publisher pub_depth         = it.advertise("/camera/depth/image_raw", 1);
    sensor_msgs::ImagePtr image_msg;
    sensor_msgs::ImagePtr image_feature_msg;
    sensor_msgs::ImagePtr depth_msg;

    // MSJ
    ros::Publisher color_info_pub = nh.advertise<sensor_msgs::CameraInfo>("/camera/color/camera_info", 1);
    ros::Publisher depth_info_pub = nh.advertise<sensor_msgs::CameraInfo>("/camera/depth/camera_info", 1);


    string file_name;
    bool bFileName = false;

    if (argc == 4) {
        file_name = string(argv[argc - 1]);
        bFileName = true;
    }

    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = exit_loop_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, NULL);
    b_continue_session = true;

    double offset = 0; // ms

    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    rs2::device selected_device;
    if (devices.size() == 0)
    {
        std::cerr << "No device connected, please connect a RealSense device" << std::endl;
        return 0;
    }
    else
        selected_device = devices[0];

    std::vector<rs2::sensor> sensors = selected_device.query_sensors();
    int index = 0;
    // We can now iterate the sensors and print their names
    for (rs2::sensor sensor : sensors)
        if (sensor.supports(RS2_CAMERA_INFO_NAME)) {
            ++index;
            if (index == 1) {
                sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
                //sensor.set_option(RS2_OPTION_AUTO_EXPOSURE_LIMIT,50000);
                sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1); // emitter on for depth information
            }
            // std::cout << "  " << index << " : " << sensor.get_info(RS2_CAMERA_INFO_NAME) << std::endl;
            get_sensor_option(sensor);
            if (index == 2){
                // RGB camera
                sensor.set_option(RS2_OPTION_EXPOSURE,camera_exposure);

            }

            if (index == 3){
                sensor.set_option(RS2_OPTION_ENABLE_MOTION_CORRECTION,0);
            }

        }

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;

    // Create a configuration for configuring the pipeline with a non default profile
    rs2::config cfg;

    // RGB stream
    cfg.enable_stream(RS2_STREAM_COLOR,640, 480, RS2_FORMAT_RGB8, camera_fps);

    // Depth stream
    // cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH,640, 480, RS2_FORMAT_Z16, camera_fps);

    // IMU stream
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F); //, 250); // 63
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F); //, 400);

    // IMU callback
    std::mutex imu_mutex;
    std::condition_variable cond_image_rec;

    vector<double> v_accel_timestamp;
    vector<rs2_vector> v_accel_data;
    vector<double> v_gyro_timestamp;
    vector<rs2_vector> v_gyro_data;

    double prev_accel_timestamp = 0;
    rs2_vector prev_accel_data;
    double current_accel_timestamp = 0;
    rs2_vector current_accel_data;
    vector<double> v_accel_timestamp_sync;
    vector<rs2_vector> v_accel_data_sync;

    cv::Mat imCV, depthCV;
    int width_img, height_img;
    double timestamp_image = -1.0;
    bool image_ready = false;
    int count_im_buffer = 0; // count dropped frames

    // start and stop just to get necessary profile
    rs2::pipeline_profile pipe_profile = pipe.start(cfg);

    
    
    if(enable_auto_exposure == false)
    {
        pipe_profile.get_device().first<rs2::depth_sensor>().set_option(rs2_option::RS2_OPTION_ENABLE_AUTO_EXPOSURE,false);
        pipe_profile.get_device().query_sensors()[1].set_option(rs2_option::RS2_OPTION_ENABLE_AUTO_EXPOSURE,false);
        //depth_sensor = profile.get_device().first_depth_sensor() //python
    }
    else
    {
        pipe_profile.get_device().first<rs2::depth_sensor>().set_option(rs2_option::RS2_OPTION_ENABLE_AUTO_EXPOSURE, true);
        pipe_profile.get_device().query_sensors()[1].set_option(rs2_option::RS2_OPTION_ENABLE_AUTO_EXPOSURE,true);

    // sensors.at(0).set_option(rs2_option::RS2_OPTION_ENABLE_AUTO_EXPOSURE,true);
    // rs400::advanced_mode advnc_mode(selected_device);
    // STAEControl ae_ctrl = advnc_mode.get_ae_control();
    // ae_ctrl.meanIntensitySetPoint = ae_meanIntensitySetPoint;
    // advnc_mode.set_ae_control(ae_ctrl);
    }


    // set AE

    pipe.stop();

    // Align depth and RGB frames
    //Pipeline could choose a device that does not have a color stream
    //If there is no color stream, choose to align depth to another stream
    rs2_stream align_to = find_stream_to_align(pipe_profile.get_streams());

    // Create a rs2::align object.
    // rs2::align allows us to perform alignment of depth frames to others frames
    //The "align_to" is the stream type to which we plan to align depth frames.
    rs2::align align(align_to);
    rs2::frameset fsSLAM;

    auto imu_callback = [&](const rs2::frame& frame)
    {
        std::unique_lock<std::mutex> lock(imu_mutex);

        if(rs2::frameset fs = frame.as<rs2::frameset>())
        {
            count_im_buffer++;

            double new_timestamp_image = fs.get_timestamp()*1e-3;
            if(abs(timestamp_image-new_timestamp_image)<0.001){
                count_im_buffer--;
                return;
            }

            if (profile_changed(pipe.get_active_profile().get_streams(), pipe_profile.get_streams()))
            {
                //If the profile was changed, update the align object, and also get the new device's depth scale
                pipe_profile = pipe.get_active_profile();
                align_to = find_stream_to_align(pipe_profile.get_streams());
                align = rs2::align(align_to);
            }

            //Align depth and rgb takes long time, move it out of the interruption to avoid losing IMU measurements
            fsSLAM = fs;

            /*
            //Get processed aligned frame
            auto processed = align.process(fuse_pangolin continue iteration
            if (!depth_frame || !color_frame) {
                cout << "Not synchronized depth and image\n";
                return;
            }


            imCV = cv::Mat(cv::Size(width_img, height_img), CV_8UC3, (void*)(color_frame.get_data()), cv::Mat::AUTO_STEP);
            depthCV = cv::Mat(cv::Size(width_img, height_img), CV_16U, (void*)(depth_frame.get_data()), cv::Mat::AUTO_STEP);

            cv::Mat depthCV_8U;
            depthCV.convertTo(depthCV_8U,CV_8U,0.01);
            cv::imshow("depth image", depthCV_8U);*/

            timestamp_image = fs.get_timestamp()*1e-3;
            image_ready = true;

            while(v_gyro_timestamp.size() > v_accel_timestamp_sync.size())
            {
                int index = v_accel_timestamp_sync.size();
                double target_time = v_gyro_timestamp[index];

                v_accel_data_sync.push_back(current_accel_data);
                v_accel_timestamp_sync.push_back(target_time);
            }

            lock.unlock();
            cond_image_rec.notify_all();
        }
    };



    pipe_profile = pipe.start(cfg, imu_callback);



    rs2::stream_profile cam_stream = pipe_profile.get_stream(RS2_STREAM_COLOR);




    rs2_intrinsics intrinsics_cam = cam_stream.as<rs2::video_stream_profile>().get_intrinsics();
    width_img = intrinsics_cam.width;
    height_img = intrinsics_cam.height;
    std::cout << " fx = " << intrinsics_cam.fx << std::endl;
    std::cout << " fy = " << intrinsics_cam.fy << std::endl;
    std::cout << " cx = " << intrinsics_cam.ppx << std::endl;
    std::cout << " cy = " << intrinsics_cam.ppy << std::endl;
    std::cout << " height = " << intrinsics_cam.height << std::endl;
    std::cout << " width = " << intrinsics_cam.width << std::endl;
    std::cout << " Coeff = " << intrinsics_cam.coeffs[0] << ", " << intrinsics_cam.coeffs[1] << ", " <<
    intrinsics_cam.coeffs[2] << ", " << intrinsics_cam.coeffs[3] << ", " << intrinsics_cam.coeffs[4] << ", " << std::endl;
    std::cout << " Model = " << intrinsics_cam.model << std::endl;

    // MSJ
    sensor_msgs::CameraInfo color_info;
    color_info.width = intrinsics_cam.width;
    color_info.height = intrinsics_cam.height;
    color_info.distortion_model = "plumb_bob";
    color_info.K[0] = intrinsics_cam.fx;
    color_info.K[4] = intrinsics_cam.fy;
    color_info.K[2] = intrinsics_cam.ppx;
    color_info.K[5] = intrinsics_cam.ppy;
    color_info.K[8] = 1.0;

    color_info.P[0] = intrinsics_cam.fx;
    color_info.P[5] = intrinsics_cam.fy;
    color_info.P[2] = intrinsics_cam.ppx;
    color_info.P[6] = intrinsics_cam.ppy;
    color_info.P[10] = 1.0;

    color_info.R[0] = 1.0;
    color_info.R[4] = 1.0;
    color_info.R[8] = 1.0;

    color_info.D.push_back(0.0);
    color_info.D.push_back(0.0);
    color_info.D.push_back(0.0);
    color_info.D.push_back(0.0);
    color_info.D.push_back(0.0);

    rs2::stream_profile depth_stream = pipe_profile.get_stream(RS2_STREAM_DEPTH);
    

    rs2_intrinsics intrinsics_depth = depth_stream.as<rs2::video_stream_profile>().get_intrinsics();

    sensor_msgs::CameraInfo depth_info;
    depth_info.width = intrinsics_depth.width;
    depth_info.height = intrinsics_depth.height;
    depth_info.distortion_model = "plumb_bob";
    depth_info.K[0] = intrinsics_depth.fx;
    depth_info.K[4] = intrinsics_depth.fy;
    depth_info.K[2] = intrinsics_depth.ppx;
    depth_info.K[5] = intrinsics_depth.ppy;
    depth_info.K[8] = 1.0;

    depth_info.P[0] = intrinsics_depth.fx;
    depth_info.P[5] = intrinsics_depth.fy;
    depth_info.P[2] = intrinsics_depth.ppx;
    depth_info.P[6] = intrinsics_depth.ppy;
    depth_info.P[10] = 1.0;

    depth_info.R[0] = 1.0;
    depth_info.R[4] = 1.0;
    depth_info.R[8] = 1.0;

    depth_info.D.push_back(0.0);
    depth_info.D.push_back(0.0);
    depth_info.D.push_back(0.0);
    depth_info.D.push_back(0.0);
    depth_info.D.push_back(0.0);


    // Create SLAM system. It initializes all system threads and gets ready to process frames.

    // I will not open pangolin viewer!
    // ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::RGBD, true, 0, file_name);
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::RGBD, enable_pangolin, 0, file_name);
    float imageScale = SLAM.GetImageScale();

    double timestamp;
    cv::Mat im, depth, im_feature;

    double t_resize = 0.f;
    double t_track = 0.f;
    rs2::frameset fs;

    Sophus::SE3f output;
    
    // Eigen::Matrix4f current_camera_pose, current_base_pose;
    Eigen::Matrix4f changer;
    /*
    changer << 0.0f, -1.0f,  0.0f,  0.0f,
               0.0f,  0.0f, -1.0f,  0.0f,
               1.0f,  0.0f,  0.0f,  0.0f,
               0.0f,  0.0f,  0.0f,  1.0f; // x 90, and z 90
    */
    changer <<  0.0f,  0.0f,  1.0f,  0.0f,
               -1.0f,  0.0f,  0.0f,  0.0f,
                0.0f, -1.0f,  0.0f,  0.0f,
                0.0f,  0.0f,  0.0f,  1.0f; // x 90, and z 90
    
    // main loop
    int print_index=0;
    geometry_msgs::PoseStamped current_pose;
    nav_msgs::Odometry current_odom;

    vector<Eigen::Matrix<float,3,1>> global_points;
    vector<Eigen::Matrix<float,3,1>> local_points;

    
    // measure time!
    //clock_t start;
    //clock_t align_time;
    //clock_t track_time;
    //double duration;

    std::chrono::steady_clock::time_point start;

    while (!SLAM.isShutDown() && ros::ok())
    {
        {
            std::unique_lock<std::mutex> lk(imu_mutex);
            if(!image_ready)
            {
                cond_image_rec.wait(lk);
            }

// #ifdef COMPILEDWITHC11
//             std::chrono::steady_clock::time_point time_Start_Process = std::chrono::steady_clock::now();
// #else
//             std::chrono::monotonic_clock::time_point time_Start_Process = std::chrono::monotonic_clock::now();
// #endif

	    std::cout<<"================================="<<std::endl; 
	    std::cout<<"Got image set!"<<std::endl;

	    //start = clock();
	    start  = std::chrono::steady_clock::now();

            fs = fsSLAM;

            if(count_im_buffer>1)
                cout << count_im_buffer -1 << " dropped frs\n";
            count_im_buffer = 0;

            timestamp = timestamp_image;
            im = imCV.clone();
            depth = depthCV.clone();

            image_ready = false;
        }

        // Perform alignment here
        auto processed = align.process(fs);

	std::chrono::steady_clock::time_point align_time = std::chrono::steady_clock::now();
	std::chrono::duration<double> dt = align_time - start;
	long long dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count(); 
	std::cout<<"align is done!"<<std::endl;
	std::cout<<"time consumed : "<<dt_ms<<" ms"<<std::endl;

        // Trying to get both other and aligned depth frames
        rs2::video_frame color_frame = processed.first(align_to);
        rs2::depth_frame depth_frame = processed.get_depth_frame();

        im = cv::Mat(cv::Size(width_img, height_img), CV_8UC3, (void*)(color_frame.get_data()), cv::Mat::AUTO_STEP);
        //im_feature = cv::Mat(cv::Size(width_img, height_img), CV_8UC3, (void*)(color_frame.get_data()), cv::Mat::AUTO_STEP);
        depth = cv::Mat(cv::Size(width_img, height_img), CV_16U, (void*)(depth_frame.get_data()), cv::Mat::AUTO_STEP);

        /*cv::Mat depthCV_8U;
        depthCV.convertTo(depthCV_8U,CV_8U,0.01);
        cv::imshow("depth image", depthCV_8U);*/
        //cv::imshow("depth image",depth);
        if(imageScale != 1.f)
        {
// #ifdef REGISTER_TIMES
//     #ifdef COMPILEDWITHC11
//             std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
//     #else
//             std::chrono::monotonic_clock::time_point t_Start_Resize = std::chrono::monotonic_clock::now();
//     #endif
// #endif
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
            cv::resize(depth, depth, cv::Size(width, height));

// #ifdef REGISTER_TIMES
//     #ifdef COMPILEDWITHC11
//             std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
//     #else
//             std::chrono::monotonic_clock::time_point t_End_Resize = std::chrono::monotonic_clock::now();
//     #endif
//             t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
//             SLAM.InsertResizeTime(t_resize);
// #endif
        }

// #ifdef REGISTER_TIMES
//     #ifdef COMPILEDWITHC11
//         std::chrono::steady_clock::time_point t_Start_Track = std::chrono::steady_clock::now();
//     #else
//         std::chrono::monotonic_clock::time_point t_Start_Track = std::chrono::monotonic_clock::now();
//     #endif
// #endif
        // Pass the image to the SLAM system
        output = SLAM.TrackRGBD(im, depth, timestamp); //, vImuMeas); depthCV

	//track_time = clock();
	//duration = (double)(track_time-align_time); 
	//std::cout<<"tracking is done!"<<std::endl;
	//std::cout<<"time consumed : "<<duration<<std::endl;

	std::chrono::steady_clock::time_point track_time = std::chrono::steady_clock::now();
	std::chrono::duration<double> dt2 = track_time - align_time;
	long long dt2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(dt2).count(); 
	std::cout<<"track is done!"<<std::endl;
	std::cout<<"time consumed : "<<dt2_ms<<" ms"<<std::endl;

        // ROS
        //publish
        // current pose based on camera frame, whose z axis is going foward!
        // current_camera_pose = output.inverse().matrix() * changer;
        Eigen::Matrix4f current_camera_pose = output.inverse().matrix();
        // Sophus::SE3f current_base_pose(current_camera_pose * changer);
        Sophus::SE3f current_base_pose(changer * current_camera_pose);
        // Sophus::SE3f current_base_pose(current_camera_pose);
        // Sophus::SO3f aligned_heading(changer * current_camera_pose);
        // Sophus::SE3f current_camera_pose, current_base_pose;
        // Eigen::Quaternionf q = current_base_pose.so3().unit_quaternion();
        // Eigen::Quaternionf q = output.inverse().so3().unit_quaternion();
        // Eigen::Quaternionf q(1,0,0,0);  // w,x,y,z
        Eigen::Quaternionf q(0.5, 0.5, -0.5, 0.5);
        q = current_base_pose.so3().unit_quaternion() * q;

        //float rpy[3];

        current_pose.header.stamp = ros::Time::now();
        current_pose.header.frame_id = "orb_odom"; // MSJ before map
        current_pose.pose.position.x = current_base_pose.translation()(0,0);
        current_pose.pose.position.y = current_base_pose.translation()(1,0);
        current_pose.pose.position.z = current_base_pose.translation()(2,0);
        
        current_pose.pose.orientation.x = q.x();
        current_pose.pose.orientation.y = q.y();
        current_pose.pose.orientation.z = q.z();
        current_pose.pose.orientation.w = q.w();

        current_odom.header.stamp = ros::Time::now();
        current_odom.header.frame_id = "orb_odom"; // MSJ before map
        current_odom.pose.pose.position.x = current_base_pose.translation()(0,0);
        current_odom.pose.pose.position.y = current_base_pose.translation()(1,0);
        current_odom.pose.pose.position.z = current_base_pose.translation()(2,0);
        
        current_odom.pose.pose.orientation.x = q.x();
        current_odom.pose.pose.orientation.y = q.y();
        current_odom.pose.pose.orientation.z = q.z();
        current_odom.pose.pose.orientation.w = q.w();

        // current_pose.pose.orientation.x = 
        pose_pub.publish(current_pose);
        odom_pub.publish(current_odom);
        // show output
        if(ros::ok() && print_index >  5 )
        {
            // Eigen::Matrix3f rotation_matrix = output.so3().matrix();
            
            // std::cout<<"translation vector : "<< output.translation()

            /*
            inline void getEulerAnglesFromQuaternion(const Eigen::Quaternion<double>& q,
                                         Eigen::Vector3d* euler_angles) {
            {
                assert(euler_angles != NULL);

                *euler_angles << std::atan2(2.0 * (q.w() * q.x() + q.y() * q.z()),
                                    1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y())),

                    std::asin(2.0 * (q.w() * q.y() - q.z() * q.x())),

                    std::atan2(2.0 * (q.w() * q.z() + q.x() * q.y()),
                        1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));
                }
            }
            */
            // rpy[0] = std::atan2(2.0 * (q.w() * q.x() + q.y() * q.z()),
            //                         1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y()));

            // rpy[1] = std::asin(2.0 * (q.w() * q.y() - q.z() * q.x()));

            // rpy[2] = std::atan2(2.0 * (q.w() * q.z() + q.x() * q.y()),
            //             1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));



            std::cout<<"current pose"<<std::endl
            
            <<"x : "<<current_base_pose.translation()(0,0)<<std::endl
            <<"y : "<<current_base_pose.translation()(1,0)<<std::endl
            <<"z : "<<current_base_pose.translation()(2,0)<<std::endl

            <<"========================"<<std::endl
            //<<"translation vector : "<< current_camera_pose.translation() <<std::endl
            // << " rotation : "<<rpy[0] <<", "<<rpy[1]<<", "<<rpy[2] << std::endl;
            << " quaternion(x,y,z,w) : "<<q.x() <<", "<<q.y()<<", "<<q.z() <<", "<<q.w()<< std::endl;
            print_index=0;
        }
                // Publish image
        
        // reference! DO NOT UNCOMMENT BELOW 2 LINES!!!
        //im = cv::Mat(cv::Size(width_img, height_img), CV_8UC3, (void*)(color_frame.get_data()), cv::Mat::AUTO_STEP);
        //depth = cv::Mat(cv::Size(width_img, height_img), CV_16U, (void*)(depth_frame.get_data()), cv::Mat::AUTO_STEP);
        std::vector<cv::KeyPoint> keypoints = SLAM.GetTrackedKeyPointsUn();
        vector<bool> mvbMap, mvbVO;
        int N = keypoints.size();
        mvbVO = vector<bool>(N,false);
        mvbMap = vector<bool>(N,false);

        SLAM.GetVOandMap(mvbVO,mvbMap);
        DrawFeature(im_feature,im,keypoints,imageScale,mvbVO,mvbMap);

        image_msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", im).toImageMsg();
        image_feature_msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", im_feature).toImageMsg();
        depth_msg = cv_bridge::CvImage(std_msgs::Header(), "mono16", depth).toImageMsg();

        // MSJ
        image_msg->header.stamp = ros::Time::now();
        image_msg->header.frame_id = "camera_color_optical_frame";

        // draw features in the image
        pub_image.publish(image_msg);
        pub_image_feature.publish(image_feature_msg);
        pub_depth.publish(depth_msg);

        // MSJ
        color_info.header.stamp = image_msg->header.stamp;
	    color_info.header.frame_id = "camera_depth_optical_frame";
	    color_info_pub.publish(color_info);

        depth_info.header.stamp = image_msg->header.stamp;
	    depth_info.header.frame_id = "camera_depth_optical_frame";
	    depth_info_pub.publish(depth_info);

        // ***********************************************************************************

        // pub pointcloud
        vector<Eigen::Matrix<float,3,1>> global_points, local_points;
        
        // SLAM.GetPointCloud(global_points,local_points);
        // PublishPointCloud(global_points,local_points,global_pc_pub,local_pc_pub);


        
        

        print_index++;
        if (!ros::ok())
        {
            std::cout<<"ros shutdown!"<<std::endl;
            break;
        }

// #ifdef REGISTER_TIMES
//     #ifdef COMPILEDWITHC11
//         std::chrono::steady_clock::time_point t_End_Track = std::chrono::steady_clock::now();
//     #else
//         std::chrono::monotonic_clock::time_point t_End_Track = std::chrono::monotonic_clock::now();
//     #endif
//         t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Track - t_Start_Track).count();
//         SLAM.InsertTrackTime(t_track);
// #endif
        // end of the loop
    }
    cout << "System shutdown!\n";
}

rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
{
    //Given a vector of streams, we try to find a depth stream and another stream to align depth with.
    //We prioritize color streams to make the view look better.
    //If color is not available, we take another stream that (other than depth)
    rs2_stream align_to = RS2_STREAM_ANY;
    bool depth_stream_found = false;
    bool color_stream_found = false;
    for (rs2::stream_profile sp : streams)
    {
        rs2_stream profile_stream = sp.stream_type();
        if (profile_stream != RS2_STREAM_DEPTH)
        {
            if (!color_stream_found)         //Prefer color
                align_to = profile_stream;

            if (profile_stream == RS2_STREAM_COLOR)
            {
                color_stream_found = true;
            }
        }
        else
        {
            depth_stream_found = true;
        }
    }

    if(!depth_stream_found)
        throw std::runtime_error("No Depth stream available");

    if (align_to == RS2_STREAM_ANY)
        throw std::runtime_error("No stream found to align with Depth");

    return align_to;
}


bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev)
{
    for (auto&& sp : prev)
    {
        //If previous profile is in current (maybe just added another)
        auto itr = std::find_if(std::begin(current), std::end(current), [&sp](const rs2::stream_profile& current_sp) { return sp.unique_id() == current_sp.unique_id(); });
        if (itr == std::end(current)) //If it previous stream wasn't found in current
        {
            return true;
        }
    }
    return false;
}

void DrawFeature(cv::Mat& im_feature, const cv::Mat im,std::vector<cv::KeyPoint> keypoints, float imageScale, vector<bool> mvbVO,vector<bool> mvbMap)
{
    // copy IMAGE
    im.copyTo(im_feature);

    cv::Point2f point(100,100);
    // cv::circle(im_feature,point,2,cv::Scalar(0,255,0),-1);   


    
    std::vector<cv::KeyPoint> keypoints_ = keypoints;
    std::vector<bool>         vbVO = mvbVO;
    std::vector<bool>         vbMap = mvbMap;
    const float r = 5;
    int n = keypoints_.size();
    
    for(int i=0;i<n;i++)
    {
        if(vbVO[i] || vbMap[i])
        {
            cv::Point2f pt1,pt2;
            cv::Point2f point;
            
            point = keypoints_[i].pt / imageScale;
            float px = keypoints_[i].pt.x / imageScale;
            float py = keypoints_[i].pt.y / imageScale;
            pt1.x=px-r;
            pt1.y=py-r;
            pt2.x=px+r;
            pt2.y=py+r;
            
            cv::rectangle(im_feature,pt1,pt2,cv::Scalar(0,255,0));
            cv::circle(im_feature,point,2,cv::Scalar(0,255,0),-1);
        }
    }
    
}

void PublishPointCloud(vector<Eigen::Matrix<float,3,1>>& global_points, vector<Eigen::Matrix<float,3,1>>& local_points,
ros::Publisher& global_pc_pub, ros::Publisher& local_pc_pub)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_pointcloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_pointcloud(new pcl::PointCloud<pcl::PointXYZ>());

    //global
    std::cout<<"global_points size : "<<global_points.size()<<std::endl;
    for(int i=0; i<global_points.size();i++)
    {
        pcl::PointXYZ pt;
        pt.x =  global_points[i](2,0);
        pt.y = -global_points[i](0,0);
        pt.z = -global_points[i](1,0);
        global_pointcloud->points.push_back(pt);
    }

    for(int i=0;i<local_points.size();i++)
    {
        pcl::PointXYZ pt;
        pt.x =  local_points[i](2,0);
        pt.y = -local_points[i](0,0);
        pt.z = -local_points[i](1,0);
        global_pointcloud->points.push_back(pt);
         local_pointcloud->points.push_back(pt); 
    }
    sensor_msgs::PointCloud2 global_map_msg;
    sensor_msgs::PointCloud2 local_map_msg;
    pcl::toROSMsg(*global_pointcloud,global_map_msg);
    pcl::toROSMsg(*local_pointcloud,local_map_msg);
    
    global_map_msg.header.frame_id = "map";
    global_map_msg.header.stamp = ros::Time::now();
    global_pc_pub.publish(global_map_msg);
    
    local_map_msg.header.frame_id = "map";
    local_map_msg.header.stamp = ros::Time::now();
    local_pc_pub.publish(local_map_msg);
}
