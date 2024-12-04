#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <iostream>
#include <thread>

using namespace std;

cv::Mat redCompensate(const cv::Mat& img, int window) {
    float alpha = 1.0f;
    cv::Mat r, g, b;
    vector<cv::Mat> channels;
    cv::split(img, channels);
    r = channels[2];
    g = channels[1];
    b = channels[0];

    r.convertTo(r, CV_32F, 1.0 / 255.0);
    g.convertTo(g, CV_32F, 1.0 / 255.0);
    b.convertTo(b, CV_32F, 1.0 / 255.0);

    int height = img.rows, width = img.cols;

    cv::Mat integral_r, integral_g, integral_b;
    cv::integral(r, integral_r, CV_32F);
    cv::integral(g, integral_g, CV_32F);
    cv::integral(b, integral_b, CV_32F);

    cv::Mat ret = img.clone();

#pragma omp parallel for collapse(2)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int x1 = max(j - window / 2, 0);
            int y1 = max(i - window / 2, 0);
            int x2 = min(j + window / 2, width - 1);
            int y2 = min(i + window / 2, height - 1);

            int area = (x2 - x1 + 1) * (y2 - y1 + 1);

            float r_mean = (integral_r.at<float>(y2 + 1, x2 + 1) - integral_r.at<float>(y1, x2 + 1) -
                integral_r.at<float>(y2 + 1, x1) + integral_r.at<float>(y1, x1)) / area;
            float g_mean = (integral_g.at<float>(y2 + 1, x2 + 1) - integral_g.at<float>(y1, x2 + 1) -
                integral_g.at<float>(y2 + 1, x1) + integral_g.at<float>(y1, x1)) / area;

            float lrc = r.at<float>(i, j) + alpha * (g_mean - r_mean) * (1 - r.at<float>(i, j)) * g.at<float>(i, j);
            ret.at<cv::Vec3b>(i, j)[2] = static_cast<uchar>(min(max(lrc * 255.0f, 0.0f), 255.0f));
        }
    }

    return ret;
}

cv::Mat gray_balance(const cv::Mat& image) {
    int L = 255;
    vector<cv::Mat> channels;
    cv::split(image, channels);
    cv::Scalar means = cv::mean(image);
    float Ravg = means[2], Gavg = means[1], Bavg = means[0];

    float Max = max({ Ravg, Gavg, Bavg });
    vector<float> ratio = { Max / Ravg, Max / Gavg, Max / Bavg };
    vector<float> satLevel = { 0.005f * ratio[0], 0.005f * ratio[1], 0.005f * ratio[2] };

    cv::Mat output = image.clone();

#pragma omp parallel for
    for (int ch = 0; ch < 3; ch++) {
        int histSize = 256;
        float range[] = { 0, 256 };
        const float* histRange = { range };
        cv::Mat hist;
        cv::calcHist(&channels[ch], 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

        vector<float> cumulative_hist(histSize, 0.0f);
        cumulative_hist[0] = hist.at<float>(0);
        for (int i = 1; i < histSize; i++) {
            cumulative_hist[i] = cumulative_hist[i - 1] + hist.at<float>(i);
        }
        for (int i = 0; i < histSize; ++i) {
            cumulative_hist[i] /= cumulative_hist[histSize - 1];
        }

        int pmin_idx = 0, pmax_idx = histSize - 1;
        while (cumulative_hist[pmin_idx] < satLevel[ch] && pmin_idx < histSize - 1) pmin_idx++;
        while (cumulative_hist[pmax_idx] > (1 - satLevel[ch]) && pmax_idx > 0) pmax_idx--;

        float pmin = static_cast<float>(pmin_idx);
        float pmax = static_cast<float>(pmax_idx);

#pragma omp parallel for
        for (int i = 0; i < channels[ch].rows; ++i) {
            uchar* pixel = channels[ch].ptr<uchar>(i);
            for (int j = 0; j < channels[ch].cols; ++j) {
                float val = static_cast<float>(pixel[j]);
                val = min(max(val, pmin), pmax);
                output.at<cv::Vec3b>(i, j)[ch] = static_cast<uchar>((val - pmin) * L / (pmax - pmin));
            }
        }
    }
    return output;
}

cv::Mat gammaCorrection(const cv::Mat& img, float alpha, float gamma) {
    static cv::Mat lut(1, 256, CV_8U);
    static bool initialized = false;

    if (!initialized) {
#pragma omp parallel for
        for (int i = 0; i < 256; i++) {
            lut.at<uchar>(i) = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
        }
        initialized = true;
    }

    cv::Mat output;
    cv::LUT(img, lut, output);
    output.convertTo(output, CV_8U, alpha);
    return output;
}

void displayOriginalVideo(const string& windowName, const string& videoPath) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open video file " << videoPath << endl;
        return;
    }

    cv::Mat frame;
    while (true) {
        auto frame_start = chrono::high_resolution_clock::now();

        if (!cap.read(frame)) {
            break;
        }

        auto frame_end = chrono::high_resolution_clock::now();
        chrono::duration<double> frame_elapsed = frame_end - frame_start;
        double current_fps = 1.0 / frame_elapsed.count();

        string fps_text = "FPS: " + to_string(current_fps);
        cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

        cv::imshow(windowName, frame);

        if (cv::waitKey(33) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyWindow(windowName);
}

void displayEnhancedVideo(const string& windowName, const string& videoPath) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open video file " << videoPath << endl;
        return;
    }

    cv::Mat frame;
    while (1) {
        auto frame_start = chrono::high_resolution_clock::now();

        if (!cap.read(frame)) {
            break;
        }

        cv::Mat red_comp_img = redCompensate(frame, 1);
        cv::Mat wb_img = gray_balance(red_comp_img);
        float gamma = 1.2f;
        cv::Mat gamma_crct_img = gammaCorrection(wb_img, 1.0f, gamma);

        auto frame_end = chrono::high_resolution_clock::now();
        chrono::duration<double> frame_elapsed = frame_end - frame_start;
        double current_fps = 1.0 / frame_elapsed.count();

        string fps_text = "FPS: " + to_string(current_fps);
        cv::putText(gamma_crct_img, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

        cv::imshow(windowName, gamma_crct_img);

        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyWindow(windowName);
}

int main() {
    string videoPath = "cray.mkv";

    thread originalVideoThread(displayOriginalVideo, "Original Video", videoPath);
    thread enhancedVideoThread(displayEnhancedVideo, "Enhanced Video", videoPath);

    originalVideoThread.join();
    enhancedVideoThread.join();

    return 0;
}