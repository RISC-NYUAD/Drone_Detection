#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <chrono>
#include <numeric>
#include <fstream>


//#include <darknet.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

constexpr auto default_batch_size = 1;


float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

float get_color(int c, int x, int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    return r;
}

double get_time_point() {
    std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
    //uint64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(current_time.time_since_epoch()).count();
    return std::chrono::duration_cast<std::chrono::microseconds>(current_time.time_since_epoch()).count();
}


struct config_type {
    std::string name;
    int backend;
    int target;
};

// select backend target combinations that you want to test
std::vector<config_type> backends = {
    {"OCV CPU", cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU},
    {"OCV OpenCL", cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_OPENCL},
    {"OCV OpenCL FP16", cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_OPENCL_FP16},

    //{"IE CPU", cv::dnn::DNN_BACKEND_INFERENCE_ENGINE, cv::dnn::DNN_TARGET_CPU},

    //{"CUDA FP32", cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA},
    //{"CUDA FP16", cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA_FP16}
};


int main(int argc, char *argv[])
{
    string cfgfile = "../cfg/yolov4-tiny-3l-drone.cfg";
    if(argc > 1)
        cfgfile = argv[1]; 
    string weightfile = "../weights/yolov4-tiny-3l-drone.weights";
    if(argc > 2)
        weightfile = argv[2]; 
    string videopath = "../demo/cut_drone.mp4";
    if(argc > 3)
        videopath = argv[3]; 
    string namefilepath = "../cfg/drone.names";
    if(argc > 4)
        namefilepath = argv[4]; 
    string savepath = "output.avi";
    if(argc > 5)
        savepath = argv[5]; 

    vector<string> classNamesVec;
    ifstream classNamesFile(namefilepath);

    if (classNamesFile.is_open()){
        string className = "";
        while (getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }

    float thresh=0.5;//参数设置
    float nms=0.35;
    int classes=1;

    
    auto net = cv::dnn::readNetFromDarknet(cfgfile, weightfile);
    //net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    //net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    //net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    //net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL_FP16);
    //net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
    //net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    auto output_names = net.getUnconnectedOutLayersNames();

    VideoCapture capture(videopath);
    int frame_w = capture.get(3);
    int frame_h = capture.get(4);
    VideoWriter video(savepath, cv::VideoWriter::fourcc('M','J','P','G'),10, Size(frame_w,frame_h));
    //capture.set(CV_CAP_PROP_FRAME_WIDTH,1920);
    //capture.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
    Mat frame, blob;
    vector<cv::Mat> detections;

    bool stop=false;
    while(!stop)
    {
        //cout<<frame.size<<endl;
        if (!capture.read(frame))
        {
            printf("fail to read.\n");
            return 0;
        }

        cv::dnn::blobFromImage(frame, blob, 1./255., cv::Size(416, 416), cv::Scalar(), true, false, CV_32F);
        net.setInput(blob);

        double before = get_time_point();
        net.forward(detections, output_names);
        double after  = get_time_point();
        float fps = 1000000. / (after - before);


        std::vector<int> indices[classes];
        std::vector<cv::Rect> boxes[classes];
        std::vector<float> scores[classes];

        for (auto& output : detections)
        {
            const auto num_boxes = output.rows;
            for (int i = 0; i < num_boxes; i++)
            {
                auto x = output.at<float>(i, 0) * frame.cols;
                auto y = output.at<float>(i, 1) * frame.rows;
                auto width = output.at<float>(i, 2) * frame.cols;
                auto height = output.at<float>(i, 3) * frame.rows;
                cv::Rect rect(x - width/2, y - height/2, width, height);
                //cout << x << y << width << height << num_boxes << output<< endl;
                for (int c = 0; c < classes; c++)
                {
                    auto confidence = *output.ptr<float>(i, 5 + c);
                    if (confidence >= thresh)
                    {
                        boxes[c].push_back(rect);
                        scores[c].push_back(confidence);
                    }
                }
            }
        }

        for (int c = 0; c < classes; c++)
            cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, nms, indices[c]);   

        for (int c= 0; c < classes; c++)
        {
            for (size_t i = 0; i < indices[c].size(); ++i)
            {
                int offset = 123457 % 80;
                float red = 255*get_color(2,offset,80);
                float green = 255*get_color(1,offset,80);
                float blue = 255*get_color(0,offset,80);
                const auto color = Scalar(blue, green, red);

                auto idx = indices[c][i];
                const auto& rect = boxes[c][idx];
                cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

                std::ostringstream label_ss;
                label_ss << classNamesVec[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
                auto label = label_ss.str();
                
                int baseline;
                auto label_bg_sz = getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
                rectangle(frame, Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
                putText(frame, label.c_str(), Point(rect.x, rect.y - baseline - 5), FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
            }
        }
        string fpsString = to_string(fps);
        putText(frame, fpsString, Point(50,50), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0),2);
        imshow("video",frame);
        int c=waitKey(1);
        if((char)c==27)
            break;
        else if(c>=0)
            waitKey(0);
        video.write(frame);

    }
    video.release();
    return 0;
}

//export PKG_CONFIG_PATH=/home/daitao/libs/opencv4.5.0/lib/pkgconfig:$PKG_CONFIG_PATH
//g++ -o main  main.cpp  `pkg-config opencv4 --cflags --libs` 
//