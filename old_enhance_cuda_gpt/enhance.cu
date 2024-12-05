#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void redCompensateKernel(const uchar3* input, uchar3* output, int width, int height, float alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Normalize RGB channels
    float r = input[idx].z / 255.0f;
    float g = input[idx].y / 255.0f;
    float b = input[idx].x / 255.0f;

    // Local red compensation formula
    float lrc = r + alpha * (g - r) * (1.0f - r) * g;
    output[idx].z = static_cast<uchar>(min(max(lrc * 255.0f, 0.0f), 255.0f));  // Red channel
    output[idx].y = input[idx].y;                                             // Green channel
    output[idx].x = input[idx].x;                                             // Blue channel
}

__global__ void grayBalanceKernel(const uchar3* input, uchar3* output, int width, int height, float r_ratio, float g_ratio, float b_ratio) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    output[idx].z = min(max(static_cast<int>(input[idx].z * r_ratio), 0), 255);  // Adjusted red
    output[idx].y = min(max(static_cast<int>(input[idx].y * g_ratio), 0), 255);  // Adjusted green
    output[idx].x = min(max(static_cast<int>(input[idx].x * b_ratio), 0), 255);  // Adjusted blue
}


__global__ void gammaCorrectionKernel(const uchar3* input, uchar3* output, int width, int height, const uchar* lut) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    output[idx].z = lut[input[idx].z];  // Red channel
    output[idx].y = lut[input[idx].y];  // Green channel
    output[idx].x = lut[input[idx].x];  // Blue channel
}


void redCompensate(const cv::Mat& input, cv::Mat& output, float alpha) {
    int width = input.cols;
    int height = input.rows;

    uchar3* d_input;
    uchar3* d_output;
    size_t size = width * height * sizeof(uchar3);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input.ptr<uchar3>(), size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    redCompensateKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, alpha);

    cudaMemcpy(output.ptr<uchar3>(), d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

void grayBalance(const cv::Mat& input, cv::Mat& output) {
    int width = input.cols;
    int height = input.rows;

    uchar3* d_input;
    uchar3* d_output;
    size_t size = width * height * sizeof(uchar3);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input.ptr<uchar3>(), size, cudaMemcpyHostToDevice);

    // Compute mean values on CPU
    cv::Scalar means = cv::mean(input);
    float r_ratio = max({means[2], means[1], means[0]}) / means[2];
    float g_ratio = max({means[2], means[1], means[0]}) / means[1];
    float b_ratio = max({means[2], means[1], means[0]}) / means[0];

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    grayBalanceKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, r_ratio, g_ratio, b_ratio);

    cudaMemcpy(output.ptr<uchar3>(), d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

void gammaCorrection(const cv::Mat& input, cv::Mat& output, float gamma) {
    int width = input.cols;
    int height = input.rows;

    uchar3* d_input;
    uchar3* d_output;
    uchar* d_lut;

    size_t size = width * height * sizeof(uchar3);
    size_t lut_size = 256 * sizeof(uchar);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_lut, lut_size);

    cudaMemcpy(d_input, input.ptr<uchar3>(), size, cudaMemcpyHostToDevice);

    // Create gamma LUT on the CPU
    uchar lut[256];
    for (int i = 0; i < 256; i++) {
        lut[i] = static_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }
    cudaMemcpy(d_lut, lut, lut_size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    gammaCorrectionKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, d_lut);

    cudaMemcpy(output.ptr<uchar3>(), d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_lut);
}


int main() {
    string videoPath = "cray.mkv";
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        cerr << "Error: Cannot open video file " << videoPath << endl;
        return -1;
    }

    cv::Mat frame, redComp, balanced, gammaCorrected;
    while (true) {
        if (!cap.read(frame)) break;

        redComp = frame.clone();
        balanced = frame.clone();
        gammaCorrected = frame.clone();

        redCompensate(frame, redComp, 1.0f);
        grayBalance(redComp, balanced);
        gammaCorrection(balanced, gammaCorrected, 1.2f);

        cv::imshow("Original Video", frame);
        cv::imshow("Enhanced Video", gammaCorrected);

        if (cv::waitKey(30) == 27) break; // Exit on 'Esc'
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

