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

#include <fstream>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

#include <thread>

using namespace nvinfer1;
using namespace std;
// using namespace cudawrapper;

void* g_deviceInput;
void* g_deviceOutput;

void DrawFeature(cv::Mat& im_feature, const cv::Mat im,std::vector<cv::KeyPoint> keypoints, float imageScale, vector<bool> mvbVO,vector<bool> mvbMap);
void PublishPointCloud(vector<Eigen::Matrix<float,3,1>>& global_points, vector<Eigen::Matrix<float,3,1>>& local_points,
ros::Publisher& global_pc_pub, ros::Publisher& local_pc_pub);
bool b_continue_session;



//tensorRT
using namespace nvinfer1;
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // You can customize the logging behavior here
        if (severity != Severity::kINFO)
        {
            // std::cout << "TensorRT Logger: " << msg << std::endl;

        }
    }
};
static Logger gLogger;
// Function to read an ONNX model file and create a TensorRT engine
ICudaEngine* createEngine(const std::string& onnxModelPath, int maxBatchSize)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
                                    // nvonnxparser::createParser(*network, gLogger)

    if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(ILogger::Severity::kWARNING)))
    {
        // gLogger.getTRTLogger() << "Error parsing ONNX file" << std::endl;
        return nullptr;
    }

    builder->setMaxBatchSize(maxBatchSize);
    // builder->setMaxWorkspaceSize(1 << 30);
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions("input", OptProfileSelector::kMIN, Dims4(1, 1, 480, 640));

    profile->setDimensions("input", OptProfileSelector::kOPT, Dims4(1, 1, 480, 640));
    profile->setDimensions("input", OptProfileSelector::kMAX, Dims4(1, 1, 480, 640));



    // ICudaEngine* engine = builder->buildCudaEngine(*network);
    IBuilderConfig* buildConfig = builder->createBuilderConfig();
    buildConfig->setMaxWorkspaceSize(1 << 30);
    buildConfig->addOptimizationProfile(profile);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *buildConfig);

    if (!engine)
    {
        // gLogger.getTRTLogger() << "Error building TensorRT engine" << std::endl;
        std::cout<<"Error occuered while building engine"<<std::endl;
        return nullptr;
    }

    parser->destroy();
    network->destroy();
    // builder->destroy();

    return engine;
}

bool doInference(IExecutionContext* context, cv::Mat& input, cv::Mat& output,size_t inputSize,size_t outputSize)
{

    cudaMemcpy(g_deviceInput, input.data, inputSize, cudaMemcpyHostToDevice);

    // Execute the TensorRT engine
    // context->execute(1, &deviceInput, &deviceOutput);
    void* buffers[] = {g_deviceInput, g_deviceOutput};
    context->executeV2(buffers);

    // Copy output data from GPU to host
    cudaMemcpy(output.data, g_deviceOutput, outputSize, cudaMemcpyDeviceToHost);

    // ITERATE

    // Cleanup
    // cudaFree(g_deviceInput);
    // cudaFree(g_deviceOutput);

    // context->destroy();

    return true;
}


// Functions for mutlithreading : notused
void processFrames(rs2::align align, rs2::frameset fs, rs2::frameset& processedFrames) {
    // Process frames
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    processedFrames = align.process(fs);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = end - start;
    long long dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();
    std::cout<<"align dt : "<<dt_ms<<std::endl;

}

void inferenceFrames(IExecutionContext* context,rs2::frameset fs, cv::Mat& im, cv::Mat& im_origin, size_t inputSize, size_t outputSize) {
    // Extract color and depth frames from frameset
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    rs2::video_frame color_frame = fs.get_color_frame();
    //rs2::depth_frame depth_frame = fs.get_depth_frame();

    // Convert frames to OpenCV Mat
    im = cv::Mat(cv::Size(640, 480), CV_8UC3, (void*)(color_frame.get_data()), cv::Mat::AUTO_STEP);
    im_origin = im;
    //cv::Mat depth = cv::Mat(cv::Size(width_img, height_img), CV_16U, (void*)(depth_frame.get_data()), cv::Mat::AUTO_STEP);

    // Convert color image to YUV
    cv::Mat img1f;
    im.convertTo(img1f, CV_32FC3, 1/255.0);
    cv::Mat imgyuv;
    cv::cvtColor(img1f, imgyuv, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> channels;
    cv::split(imgyuv, channels);

    // Perform inference on the Y channel
    cv::Mat infer_input = channels[0];
    cv::Mat infer_output(480, 640, CV_32FC1);
    infer_input = infer_input * 2.0 - 1.0;

    // Perform inference
    std::chrono::steady_clock::time_point before_infer = std::chrono::steady_clock::now();
    if(doInference(context, infer_input, infer_output, inputSize, outputSize))
    {
        std::cout<<"Success"<<std::endl;
    }
    else
    {
        std::cout<<"Fail"<<std::endl;
    }
    

    std::chrono::steady_clock::time_point after_infer = std::chrono::steady_clock::now();
    channels[0] = infer_output*0.5 + 0.5;
    cv::Mat imyuv_output, im_32f;
    cv::merge(channels, imyuv_output);

    cv::cvtColor(imyuv_output,im_32f,cv::COLOR_YCrCb2BGR);
    im_32f.convertTo(im, CV_8UC3, 255);
    
    std::chrono::steady_clock::time_point process_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = process_time - start;
    long long dt_ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();

    std::chrono::duration<double> dt2 = before_infer - start;
    long long dt_ms2 = std::chrono::duration_cast<std::chrono::milliseconds>(dt2).count();

    std::chrono::duration<double> dt3 = after_infer - before_infer;
    long long dt_ms3 = std::chrono::duration_cast<std::chrono::milliseconds>(dt3).count();

    std::chrono::duration<double> dt4 = process_time - after_infer;
    long long dt_ms4 = std::chrono::duration_cast<std::chrono::milliseconds>(dt4).count();
    std::cout<<"process time1 : "<<dt_ms1<<std::endl;
    std::cout<<"process time2 : "<<dt_ms2<<std::endl;
    std::cout<<"process time3 : "<<dt_ms3<<std::endl;
    std::cout<<"process time4 : "<<dt_ms4<<std::endl;

}
/*
void preprocessImage(cv::Mat& frame, float* gpu_input, const nvinfer1::Dims& dims)
{
    
    cv::cuda::GpuMat gpu_frame;
    // upload image to GPU
    gpu_frame.upload(frame);

    auto input_width = dims.d[2];
    auto input_height = dims.d[1];
    auto channels = dims.d[0];
    auto input_size = cv::Size(input_width, input_height);
    // resize
    cv::cuda::GpuMat resized;
    //cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);
    cv::cuda::GpuMat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);

    std::vector< cv::cuda::GpuMat > chw;
    for (size_t i = 0; i < channels; ++i)
    {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));
    }
    cv::cuda::split(flt_image, chw);
}

void postprocessResults(float *gpu_output, const nvinfer1::Dims &dims, int batch_size)
{
    // get class names
    // auto classes = getClassNames("imagenet_classes.txt");
 
    // copy results from GPU to CPU
    std::vector< float > cpu_output(getSizeByDim(dims) * batch_size);
    //cudaMemcpy(source, destination, number of byte, cudaMemDeviceToHost)

    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // calculate softmax
    std::transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val) {return std::exp(val);});
    auto sum = std::accumulate(cpu_output.begin(), cpu_output.end(), 0.0);
    // find top classes predicted by the model
    std::vector< int > indices(getSizeByDim(dims) * batch_size);
    // generate sequence 0, 1, 2, 3, ..., 999
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&cpu_output](int i1, int i2) {return cpu_output[i1] > cpu_output[i2];});
    // print results
    int i = 0;
    while (cpu_output[indices[i]] / sum > 0.005)
    {
        if (classes.size() > indices[i])
        {
            std::cout << "class: " << classes[indices[i]] << " | ";
        }
        std::cout << "confidence: " << 100 * cpu_output[indices[i]] / sum << "% | index: " << indices[i] << "n";
        ++i;
    }
}

void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
                    TRTUniquePtr< nvinfer1::IExecutionContext >& context)
{
    TRTUniquePtr< nvinfer1::IBuilder > builder{nvinfer1::createInferBuilder(gLogger)};
    TRTUniquePtr< nvinfer1::INetworkDefinition > network{builder->createNetwork()};
    TRTUniquePtr< nvonnxparser::IParser > parser{nvonnxparser::createParser(*network, gLogger)};
    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast< int >(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model.\n";
        return;
    }

    TRTUniquePtr< nvinfer1::IBuilderConfig > config{builder->createBuilderConfig()};
    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(1ULL << 30);
    // use FP16 mode if possible
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // we have only one image in batch
    builder->setMaxBatchSize(1);

    engine.reset(builder->buildEngineWithConfig(*network, *config));
    context.reset(engine->createExecutionContext());
}
// class Logger : public nvinfer1::ILogger
// {
// public:
//     void log(Severity severity, const char* msg) noexcept override {
//         // remove this 'if' if you need more logged info
//         if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
//             std::cout << msg << "n";
//         }
//     }
// } gLogger;
*/

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

    // ros
    ros::init(argc, argv,"ros_rgbd_realsense");
    ros::NodeHandle nh;
    ros::NodeHandle nh_param("~");
    ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("orb_pose",1);
    ros::Publisher odom_pub = nh.advertise<nav_msgs::Odometry>("orb_odom",1);
    ros::Publisher global_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/ORB3/globalmap",1);
    ros::Publisher  local_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/ORB3/localmap",1);

    string nodename = ros::this_node::getName();

    bool enable_pangolin;
    if (!nh_param.getParam(nodename+"/enable_pangolin",enable_pangolin))
    {
        std::cout<<"It has not been decided whether to use Pangolin."<<std::endl
        <<"shut down the program"<<std::endl;
        return 1;
    }
    bool enable_auto_exposure;
    if (!nh_param.getParam(nodename+"/enable_auto_exposure",enable_auto_exposure))
    {
        std::cout<<"It has not been decided whether to use auto_exposure."<<std::endl
        <<"shut down the program"<<std::endl;
        return 1;
    }
    int ae_meanIntensitySetPoint;
    if(enable_auto_exposure)
    {
        if (!nh_param.getParam(nodename+"/ae_meanIntensitySetPoint",ae_meanIntensitySetPoint))
        {
            std::cout<<"It has not been decided the number of the mean Intensity Set Point."<<std::endl
            <<"shut down the program"<<std::endl;
            return 1;
        }
    }
    
    std::string onnx_path;
    if (!nh_param.getParam(nodename+"/path_to_onnx",onnx_path))
    {
        std::cout<<"Please set onnx path."<<std::endl
        <<"shut down the program"<<std::endl;
        return 1;
    }




    // tensorRT
    // nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    // std::cout<<"onnx path : "<<onnx_path<<std::endl;
    // std::ifstream engine_file(onnx_path, std::ios::binary);
    // if (!engine_file) {
    //     std::cerr << "Failed to open engine file." << std::endl;
    //     return 1;
    // }
    // engine_file.seekg(0, engine_file.end);
    // const size_t engine_size = engine_file.tellg();
    // engine_file.seekg(0, engine_file.beg);

    // std::vector<char> engine_data(engine_size);
    // engine_file.read(engine_data.data(), engine_size);
    // engine_file.close();


    // nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_size, nullptr);

    // 1. TensorRT 런타임 생성
    // nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);

    // // 2. 저장된 엔진 파일 불러오기
    // std::ifstream engineFile(onnx_path, std::ios::binary);
    // if (!engineFile) {
    //     std::cerr << "Failed to open engine file." << std::endl;
    //     return 1;
    // }

    // engineFile.seekg(0, engineFile.end);
    // size_t engineSize = engineFile.tellg();
    // engineFile.seekg(0, engineFile.beg);

    // std::vector<char> engineData(engineSize);
    // engineFile.read(engineData.data(), engineSize);
    // engineFile.close();

    // // 3. 런타임에서 엔진 디-시리얼라이즈
    // nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineSize, nullptr);

    // not debuged from here

    // nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    // context->execute(batch_size, bindings);


    // Options options;
    // nvinfer1::ICudaEngine* engine = createEngine(onnxModelPath, maxBatchSize);
    // std::unique_ptr<nvinfer1::ICudaEngine, cudawrapper::Destroy<nvinfer1::ICudaEngine>> engine{nullptr};
    //******************************************
    std::string onnxModelPath = onnx_path;
    // 1. from onnx
    // std::cout<<"Uploading onnx model"<<std::endl;
    // std::cout<<"It takes pretty long time"<<std::endl;
    
    // nvinfer1::ICudaEngine* engine = createEngine(onnxModelPath, 1);

    // IExecutionContext* context = engine->createExecutionContext();

    // 2. from trt
    IRuntime* runtime = createInferRuntime(gLogger);

    std::ifstream engineFile(onnxModelPath, std::ios::binary);
    if (!engineFile) {
        std::cerr << "Failed to open engine file." << std::endl;
        return 1;
    }
    engineFile.seekg(0, engineFile.end);
    size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(engineSize);
    engineFile.read(engineData.data(), engineSize);
    engineFile.close();

    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineSize, nullptr);
    IExecutionContext* context = engine->createExecutionContext();






    cv::Mat temp_mat(480, 640, CV_32FC1);


    size_t inputSize = temp_mat.total() * temp_mat.elemSize();
    size_t outputSize = temp_mat.total() * temp_mat.elemSize();

    cudaMalloc(&g_deviceInput, inputSize);
    cudaMalloc(&g_deviceOutput, outputSize);


    std::cout<<"ONNX model is uploaded"<<std::endl;



    std::cout<<"\ntensorrt uploading is done"<<std::endl;

// for image handling
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub_image         = it.advertise("/camera/color/image_raw", 1);
    image_transport::Publisher pub_image_feature = it.advertise("/orb3_feature_image", 1);
    image_transport::Publisher pub_depth         = it.advertise("/camera/depth/image_rect_raw", 1);
    sensor_msgs::ImagePtr image_msg;
    sensor_msgs::ImagePtr image_feature_msg;
    sensor_msgs::ImagePtr depth_msg;


    string file_name;

    if (argc == 4) {
        file_name = string(argv[argc - 1]);
    }

    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = exit_loop_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, NULL);
    b_continue_session = true;

    // double offset = 0; // ms

    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    rs2::device selected_device;

    


    if (devices.size() == 0)
    {
        std::cerr << "No device connected, please connect a RealSense device" << std::endl;
        return 0;
    }
    else
    {
        selected_device = devices[0];
    }
    std::vector<rs2::sensor> sensors = selected_device.query_sensors();
    int index = 0;
    // We can now iterate the sensors and print their names
    for (rs2::sensor sensor : sensors)
        if (sensor.supports(RS2_CAMERA_INFO_NAME)) {
            ++index;
            if (index == 1) {
                sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
                sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1); // emitter on for depth information
            }
            // std::cout << "  " << index << " : " << sensor.get_info(RS2_CAMERA_INFO_NAME) << std::endl;
            get_sensor_option(sensor);
            if (index == 2){
                // RGB camera
                sensor.set_option(RS2_OPTION_EXPOSURE,80.f);

            }

            if (index == 3){
                sensor.set_option(RS2_OPTION_ENABLE_MOTION_CORRECTION,0);
            }

        }

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    rs2::config cfg;

    // RGB stream
    cfg.enable_stream(RS2_STREAM_COLOR,640, 480, RS2_FORMAT_RGB8, 15);
    cfg.enable_stream(RS2_STREAM_DEPTH,640, 480, RS2_FORMAT_Z16, 15);


    // IMU callback
    std::mutex imu_mutex;
    std::condition_variable cond_image_rec;

    vector<double> v_accel_timestamp;
    vector<rs2_vector> v_accel_data;
    vector<double> v_gyro_timestamp;
    vector<rs2_vector> v_gyro_data;

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
    }


    // set AE

    pipe.stop();


    rs2_stream align_to = find_stream_to_align(pipe_profile.get_streams());

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

            fsSLAM = fs;

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



    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::RGBD, enable_pangolin, 0, file_name);
    float imageScale = SLAM.GetImageScale();

    double timestamp;
    cv::Mat im, depth, im_feature, im_origin;

    rs2::frameset fs;

    Sophus::SE3f output;
    
    Eigen::Matrix4f changer;

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

    rs2::frameset processed;
    while (!SLAM.isShutDown() && ros::ok())
    {
        {
            std::unique_lock<std::mutex> lk(imu_mutex);
            if(!image_ready)
                cond_image_rec.wait(lk);
    
            fs = fsSLAM;

            if(count_im_buffer>1)
                cout << count_im_buffer -1 << " dropped frs\n";
            count_im_buffer = 0;

            timestamp = timestamp_image;
            im = imCV.clone();
            depth = depthCV.clone();

            image_ready = false;
        }

        

        // Multithread
        // it is better not to use multithread, since align doesn't take much time
        // std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        // std::thread t1(inferenceFrames,std::ref(context), fs, std::ref(im), std::ref(im_origin),inputSize, outputSize);
        // std::thread t2(processFrames, align, fs, std::ref(processed));
        // t2.join();
        // t1.join();
        
        // depth = cv::Mat(cv::Size(width_img, height_img), CV_16U, (void*)(processed.get_depth_frame().get_data()), cv::Mat::AUTO_STEP);

        // std::chrono::steady_clock::time_point totalprocess_time = std::chrono::steady_clock::now();
        // std::chrono::duration<double> dt = totalprocess_time - start;
        // long long dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();
        // std::cout<<"Total process time : "<<dt_ms<<std::endl;

        // Perform alignment here
        auto processed = align.process(fs);

        // Trying to get both other and aligned depth frames
        rs2::video_frame color_frame = processed.first(align_to);
        rs2::depth_frame depth_frame = processed.get_depth_frame();

        // rs2::video_frame color_frame = fs.get_color_frame();
        // rs2::depth_frame depth_frame = fs.get_depth_frame();

        im = cv::Mat(cv::Size(width_img, height_img), CV_8UC3, (void*)(color_frame.get_data()), cv::Mat::AUTO_STEP);
        //im_feature = cv::Mat(cv::Size(width_img, height_img), CV_8UC3, (void*)(color_frame.get_data()), cv::Mat::AUTO_STEP);
        depth = cv::Mat(cv::Size(width_img, height_img), CV_16U, (void*)(depth_frame.get_data()), cv::Mat::AUTO_STEP);

        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

        cv::Mat img1f;
        im.convertTo(img1f, CV_32FC3, 1/255.0);
        cv::Mat imgyuv;
        cv::cvtColor(img1f,imgyuv,cv::COLOR_BGR2YCrCb);
        // cv::imshow("img1",imgyuv);
        // cv::waitKey(10);
        std::vector<cv::Mat> channels;
        cv::split(imgyuv, channels);


        // ********************************************************************
        cv::Mat infer_input = channels[0];
        cv::Mat infer_output(480, 640, CV_32FC1);
        infer_input = infer_input * 2.0 - 1.0;
        // Perform inference
        std::chrono::steady_clock::time_point before_infer = std::chrono::steady_clock::now();

        if (!doInference(context, infer_input, infer_output, inputSize, outputSize))
        {
            std::cout<<"Fail to inference"<<std::endl;
        }
        std::chrono::steady_clock::time_point after_infer = std::chrono::steady_clock::now();
        channels[0] = infer_output*0.5 + 0.5;
        cv::Mat imyuv_output, im_32f;
        cv::merge(channels, imyuv_output);

        cv::cvtColor(imyuv_output,im_32f,cv::COLOR_YCrCb2BGR);
        im_32f.convertTo(im, CV_8UC3, 255);
        std::chrono::steady_clock::time_point process_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> dt = process_time - start;
        long long dt_ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();

        std::chrono::duration<double> dt2 = before_infer - start;
        long long dt_ms2 = std::chrono::duration_cast<std::chrono::milliseconds>(dt2).count();

        std::chrono::duration<double> dt3 = after_infer - before_infer;
        long long dt_ms3 = std::chrono::duration_cast<std::chrono::milliseconds>(dt3).count();

        std::chrono::duration<double> dt4 = process_time - after_infer;
        long long dt_ms4 = std::chrono::duration_cast<std::chrono::milliseconds>(dt4).count();
        std::cout<<"Total processing : "<<dt_ms1<<std::endl;
        std::cout<<"Before inference : "<<dt_ms2<<std::endl;
        std::cout<<"during inference : "<<dt_ms3<<std::endl;
        std::cout<<"after inference  : "<<dt_ms4<<std::endl;


        


        // ********************************************************************

        if(imageScale != 1.f)
        {

            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
            cv::resize(depth, depth, cv::Size(width, height));

        }

        // Pass the image to the SLAM system
        output = SLAM.TrackRGBD(im, depth, timestamp); //, vImuMeas); depthCV

        // ROS publish
        
        Eigen::Matrix4f current_camera_pose = output.inverse().matrix();
        Sophus::SE3f current_base_pose(changer * current_camera_pose);

        Eigen::Quaternionf q(0.5, 0.5, -0.5, 0.5);
        q = current_base_pose.so3().unit_quaternion() * q;

        current_pose.header.stamp = ros::Time::now();
        current_pose.header.frame_id = "map";
        current_pose.pose.position.x = current_base_pose.translation()(0,0);
        current_pose.pose.position.y = current_base_pose.translation()(1,0);
        current_pose.pose.position.z = current_base_pose.translation()(2,0);
        
        current_pose.pose.orientation.x = q.x();
        current_pose.pose.orientation.y = q.y();
        current_pose.pose.orientation.z = q.z();
        current_pose.pose.orientation.w = q.w();

        current_odom.header.stamp = ros::Time::now();
        current_odom.header.frame_id = "map";
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

            std::cout<<"current pose"<<std::endl
            
            <<"x : "<<current_base_pose.translation()(0,0)<<std::endl
            <<"y : "<<current_base_pose.translation()(1,0)<<std::endl
            <<"z : "<<current_base_pose.translation()(2,0)<<std::endl

            <<"========================"<<std::endl

            << " quaternion(x,y,z,w) : "<<q.x() <<", "<<q.y()<<", "<<q.z() <<", "<<q.w()<< std::endl;
            print_index=0;
        }
                // Publish image
        
        // reference! DO NOT UNCOMMENT BELOW 2 LINES!!!
        //im = cv::Mat(cv::Size(width_img, height_img), CV_8UC3, (void*)(color_frame.get_data()), cv::Mat::AUTO_STEP);
        //depth = cv::Mat(cv::Size(width_img, height_img), CV_16U, (void*)(depth_frame.get_data()), cv::Mat::AUTO_STEP);

        // draw features in the image

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

        pub_image.publish(image_msg);
        pub_image_feature.publish(image_feature_msg);
        pub_depth.publish(depth_msg);

        // pub pointcloud
        vector<Eigen::Matrix<float,3,1>> global_points, local_points;
        // global_points.clear();
        // local_points.clear();
        SLAM.GetPointCloud(global_points,local_points);

        PublishPointCloud(global_points,local_points,global_pc_pub,local_pc_pub);


        
        

        print_index++;
        if (!ros::ok())
        {
            std::cout<<"ros shutdown!"<<std::endl;
            break;
        }


        // end of the loop
    }

    cudaFree(g_deviceInput);
    cudaFree(g_deviceOutput);

    context->destroy();
    cout << "System shutdown!\n";
}

rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
{
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
    // std::cout<<"global_points size : "<<global_points.size()<<std::endl;
    for(int i=0; i<(int)global_points.size();i++)
    {
        pcl::PointXYZ pt;
        pt.x =  global_points[i](2,0);
        pt.y = -global_points[i](0,0);
        pt.z = -global_points[i](1,0);
        global_pointcloud->points.push_back(pt);
    }

    for(int i=0;i<(int)local_points.size();i++)
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
