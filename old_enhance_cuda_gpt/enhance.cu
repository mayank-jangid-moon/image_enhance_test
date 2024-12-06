#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>
#include <iostream>
#include <thread>

using namespace std;

// CUDA Red Compensation Function
cv::cuda::GpuMat redCompensate(const cv::cuda::GpuMat& img, int window) {
    vector<cv::cuda::GpuMat> channels;
    cv::cuda::split(img, channels);
    cv::cuda::GpuMat r = channels[2];
    cv::cuda::GpuMat g = channels[1];

    cv::cuda::GpuMat compensatedR;
    // Example logic: r = r + alpha * (mean(g) - mean(r));
    cv::Scalar rMean = cv::cuda::mean(r);
    cv::Scalar gMean = cv::cuda::mean(g);
    float alpha = 1.0f;
    cv::cuda::addWeighted(r, 1.0, g, alpha * (gMean[0] - rMean[0]), 0, compensatedR);

    channels[2] = compensatedR;
    cv::cuda::GpuMat result;
    cv::cuda::merge(channels, result);
    return result;
}

// CUDA Gamma Correction Function
cv::cuda::GpuMat gammaCorrection(const cv::cuda::GpuMat& img, float gamma) {
    cv::Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i) {
        lut.at<uchar>(i) = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }

    cv::cuda::GpuMat d_lut;
    d_lut.upload(lut);

    cv::cuda::GpuMat output;
    cv::cuda::LUT(img, d_lut, output);
    return output;
}

// CUDA Gray Balance Function
cv::cuda::GpuMat gray_balance(const cv::cuda::GpuMat& img) {
    vector<cv::cuda::GpuMat> channels;
    cv::cuda::split(img, channels);

    cv::Scalar means = cv::cuda::mean(img);
    float Ravg = means[2], Gavg = means[1], Bavg = means[0];
    float maxChannel = max({Ravg, Gavg, Bavg});

    vector<float> ratios = { maxChannel / Bavg, maxChannel / Gavg, maxChannel / Ravg };
    for (int i = 0; i < 3; i++) {
        cv::cuda::multiply(channels[i], ratios[i], channels[i]);
    }

    cv::cuda::GpuMat balanced;
    cv::cuda::merge(channels, balanced);
    return balanced;
}

// Display Original Video
void displayOriginalVideo(const string& windowName, const string& videoPath) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open video file " << videoPath << endl;
        return;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        auto frame_start = chrono::high_resolution_clock::now();

        auto frame_end = chrono::high_resolution_clock::now();
        chrono::duration<double> frame_elapsed = frame_end - frame_start;
        double current_fps = 1.0 / frame_elapsed.count();

        string fps_text = "FPS: " + to_string(current_fps);
        cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

        cv::imshow(windowName, frame);
        if (cv::waitKey(33) == 27) break;
    }

    cap.release();
    cv::destroyWindow(windowName);
}

// Display Enhanced Video
void displayEnhancedVideo(const string& windowName, const string& videoPath) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open video file " << videoPath << endl;
        return;
    }

    cv::Mat frame;
    cv::cuda::GpuMat d_frame, d_result;

    while (cap.read(frame)) {
        auto frame_start = chrono::high_resolution_clock::now();

        d_frame.upload(frame);

        // Apply Red Compensation
        d_result = redCompensate(d_frame, 1);

        // Apply Gray Balance
        d_result = gray_balance(d_result);

        // Apply Gamma Correction
        d_result = gammaCorrection(d_result, 1.2f);

        d_result.download(frame);

        auto frame_end = chrono::high_resolution_clock::now();
        chrono::duration<double> frame_elapsed = frame_end - frame_start;
        double current_fps = 1.0 / frame_elapsed.count();

        string fps_text = "FPS: " + to_string(current_fps);
        cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

        cv::imshow(windowName, frame);
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyWindow(windowName);
}

// Main Function
int main() {
    string videoPath = "video.mkv";

    thread originalVideoThread(displayOriginalVideo, "Original Video", videoPath);
    thread enhancedVideoThread(displayEnhancedVideo, "Enhanced Video", videoPath);

    originalVideoThread.join();
    enhancedVideoThread.join();

    return 0;
}
