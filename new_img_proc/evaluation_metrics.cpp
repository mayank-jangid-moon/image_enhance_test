/*********************************************************************/
/* Project: uw-img-proc                                              */
/* Module:  evaluationmetrics                                        */
/* File:    evaluationmetrics.h                                      */
/* Created: 09/12/2018                                               */
/* Description:
    C++ Module of image quality metrics for image processing evaluation
 /********************************************************************/

 /********************************************************************/
 /* Created by:                                                      */
 /* Geraldine Barreto (@geraldinebc)                                 */
 /********************************************************************/

///Basic C and C++ libraries
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

/// OpenCV libraries. May need review for the final release
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

// C++ namespaces
using namespace cv;
using namespace cuda;
using namespace std;
using namespace xfeatures2d;

/// CUDA specific libraries
#if USE_GPU
#include <opencv2/cudafilters.hpp>
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#endif

/*
    @brief      Computes the entropy of a grayscale image according to the Shannon Index
    @function   float entropy(cv::Mat img)
*/
float entropy(cv::Mat img);

/*
    @brief      Computes the average entropy of a RGB image according to the Shannon Index
    @function   float averageEntropy(std::vector<Mat> channel)
*/
float averageEntropy(std::vector<Mat> channel);

/*
    @brief      Computes the average contrast using image gradients
    @function   float averageContrast(std::vector<Mat> chanRGB)
*/
float averageContrast(std::vector<Mat> chanRGB);

/*
    @brief      Computes the average luminance using the luminance channel in YCrCb
    @function   float averageLuminance(cv::Mat L)
*/
float averageLuminance(cv::Mat L);

/*
    @brief      Computes the normalized neighborhood function using the average luminance
    @function   float getNNF(float AL)
*/
float getNNF(float AL);

/*
    @brief      Computes the comprehensive assessment function
    @function   float getCAF(float AE, float AC, float NNF)
*/
float getCAF(float AE, float AC, float NNF);

/*
    @brief      Computes the mean square error
    @function   float getMSE(cv::Mat src, cv::Mat dst)
*/
float getMSE(cv::Mat src, cv::Mat dst);

/*
    @brief      Computes the peak signal to noise ratio
    @function   float getPSNR(double mse)
*/
float getPSNR(float mse);

/*
    @brief      Computes the Frequency Domain Image Sharpness
    @function   float sharpness(cv::Mat src)
*/
float sharpness(cv::Mat src);

/*
    @brief      Computes the intensity distribution histogram
    @function   void getHistogram(cv::Mat *channel, cv::Mat *hist)
*/
void getHistogram(cv::Mat *channel, cv::Mat *hist);

/*
    @brief      Creates an image that represents the Histogram of one image channel
    @function   printHist(Mat histogram, Scalar color);
*/
cv::Mat printHist(cv::Mat histogram, Scalar color);

#if USE_GPU
float entropy_GPU(cv::cuda::GpuMat img);

float averageEntropy_GPU(std::vector<cv::cuda::GpuMat> chanRGB);

float averageContrast_GPU(std::vector<cv::cuda::GpuMat> chanRGB);

float averageLuminance_GPU(cv::cuda::GpuMat L);

float getMSE_GPU(cv::cuda::GpuMat src, cv::cuda::GpuMat dst);

float sharpness_GPU(cv::cuda::GpuMat srcGPU);
#endif

/*********************************************************************/
/* Project: uw-img-proc                                              */
/* Module:  evaluationmetrics                                        */
/* File:    evaluationmetrics.cpp                                    */
/* Created: 09/12/2018                                               */
/* Description:
    C++ Module of image quality metrics for image processing evaluation
 /********************************************************************/

 /********************************************************************/
 /* Created by:                                                      */
 /* Geraldine Barreto (@geraldinebc)                                 */
 /********************************************************************/


float entropy(cv::Mat img) {
    cv::Mat hist, normhist, prob, logP;
    getHistogram(&img, &hist);
    normalize(hist, normhist, 0, 1, NORM_MINMAX);               // Normalized histogram
    prob = normhist / sum(normhist).val[0];                     // Probability
    prob += 0.00000001;                                         // Added 0.00000001 to avoid errors calculating the logarithm
    log(prob, logP);                                            // Natural logarithm of the probability
    float entropy = -1.0*sum(prob.mul(logP/log(2))).val[0];     // Computes the entropy according to the Shannon Index  
    return entropy;
}

float averageEntropy(std::vector<Mat> chanRGB) {            // Entropy considering the RGB components
    float AE = sqrt((pow(entropy(chanRGB[0]), 2) + pow(entropy(chanRGB[1]), 2) + pow(entropy(chanRGB[2]), 2))/3);
    return AE;
}

float averageContrast(std::vector<Mat> chanRGB) {
    cv::Mat Grad[3], Grad2[3];
    int kernel_size = 3, scale = 1, delta = 0, ddepth = CV_32F;
    Laplacian(chanRGB[0], Grad[0], ddepth, kernel_size, scale, delta, BORDER_DEFAULT);      // RGB Gradients
    Laplacian(chanRGB[1], Grad[1], ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
    Laplacian(chanRGB[2], Grad[2], ddepth, kernel_size, scale, delta, BORDER_DEFAULT);  
    
    pow(Grad[0], 2, Grad2[0]), pow(Grad[1], 2, Grad2[1]), pow(Grad[2], 2, Grad2[2]);        // Grad^2
    cv::Mat Grads, sqrGrads;
    Grads = (Grad2[0] + Grad2[1] + Grad2[2]) / 3;
    sqrt(Grads, sqrGrads);
    cv::Mat C = sqrGrads(Rect(0, 0, sqrGrads.cols-2, sqrGrads.rows-2));
    float AC = sum(C)[0] / ((sqrGrads.rows - 1)*(sqrGrads.cols - 1));                       // Normalized sum of gradients
    return AC;
}

float averageLuminance(cv::Mat L) {
    float AL = sum(L)[0] / L.total();                                       // Normalized sum of the luminance component
    return AL;
}

float getNNF(float AL) {
    float OL = 127.5;
    float NNF = (OL - abs(AL-OL)) / OL;                                     // Normalized Neighborhood Function
    return NNF;
}

float getCAF(float AE, float AC, float NNF) {
    float CAF = AE + pow(AC,1/4) + pow(NNF,3);                              // Comprehensive Assessment Function
    return CAF;
}

float getMSE(cv::Mat src, cv::Mat dst) {
    cv::Mat diff(src.rows, src.cols, CV_32F);
    absdiff(src, dst, diff);
    pow(diff, 2, diff);
    float mse = sum(diff)[0] / src.total();                             // Mean Square Error
    return mse;
}

float getPSNR(float mse) {
    float psnr = 20 * log10(255 / sqrt(mse));                               // Peak Signal to Noise Ratio
    return psnr;
}

float sharpness(cv::Mat src){
    cv::Mat padded, src_dft, dft_mag;
    int m = getOptimalDFTSize(src.rows);                    // Optimal Size to calculate the Discrete Fourier Transform 
    int n = getOptimalDFTSize(src.cols);
    copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0)); // Resize to optimal FFT size
    cv::Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    merge(planes, 2, src_dft);                              // Plane with zeros is added so the the real and complex results will fit in the source matrix
    dft(src_dft, src_dft);                                  // Discrete Fourier Transform
    split(src_dft / src.total(), planes);                   // The normalized result is splitted -> planes[0] = Re(DFT(src)), planes[1] = Im(DFT(src))
    magnitude(planes[0], planes[1], dft_mag);               // dft_mag = sqrt(Re(DFT(src))^2+Im(DFT(src))^2)
    double max;
    minMaxIdx(dft_mag, NULL, &max);                         // Maximum value of the Fourier transform magnitude
    double thresh = max/1000;                               // Threshold to calculate the IQM
    cv::Mat dft_thresh;
    threshold(dft_mag, dft_thresh, thresh, 1, 0);           // If dft_mag > thresh set to 1, else set to 0
    float TH = countNonZero(dft_thresh);                    // Number of pixels in dft_mag whose pixel value > thres
    float IQM = TH / src.total();                           // Computes the Sharpness Measure
    return IQM;
}

void getHistogram(cv::Mat *channel, cv::Mat *hist) {        // Computes the intensity distribution histogram
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    calcHist(channel, 1, 0, Mat(), *hist, 1, &histSize, &histRange, true, false);
}

cv::Mat printHist(cv::Mat histogram, Scalar color) {
    // Finding the maximum value of the histogram. It will be used to scale the histogram to fit the image
    int max = 0;
    for (int i = 0; i < 256; i++) {
        if (histogram.at<float>(i, 0) > max) max = histogram.at<float>(i, 0);
    }
    // Histogram Image
    cv::Mat imgHist(1480, 1580, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Point pt1, pt2;
    pt1.y = 1380;
    for (int i = 0; i < 256; i++) {
        pt1.x = 150 + 5 * i + 1;
        pt2.x = 150 + 5 * i + 3;
        pt2.y = 1380 - 1280 * histogram.at<float>(i, 0) / max;
        cv::rectangle(imgHist, pt1, pt2, color, cv::FILLED);
    }
    // y-axis labels
    cv::rectangle(imgHist, cv::Point(130, 1400), cv::Point(1450, 80), cv::Scalar(0, 0, 0), 1);
    cv::putText(imgHist, std::to_string(max), cv::Point(10, 100), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 0), 2.0);
    cv::putText(imgHist, std::to_string(max * 3 / 4), cv::Point(10, 420), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 0), 2.0);
    cv::putText(imgHist, std::to_string(max / 2), cv::Point(10, 740), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 0), 2.0);
    cv::putText(imgHist, std::to_string(max / 4), cv::Point(10, 1060), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 0), 2.0);
    cv::putText(imgHist, std::to_string(0), cv::Point(10, 1380), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 0), 2.0);
    // x-axis labels
    cv::putText(imgHist, std::to_string(0), cv::Point(152 - 7 * 1, 1430), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 0), 2.0);
    cv::putText(imgHist, std::to_string(63), cv::Point(467 - 7 * 2, 1430), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 0), 2.0);
    cv::putText(imgHist, std::to_string(127), cv::Point(787 - 7 * 3, 1430), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 0), 2.0);
    cv::putText(imgHist, std::to_string(191), cv::Point(1107 - 7 * 3, 1430), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 0), 2.0);
    cv::putText(imgHist, std::to_string(255), cv::Point(1427 - 7 * 3, 1430), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 0), 2.0);

    return imgHist;
}

#if USE_GPU
float entropy_GPU(cv::cuda::GpuMat img) {
    cv::cuda::GpuMat hist, normhist, prob, logP, mult;
    cv::cuda::calcHist(img, hist);
    cv::cuda::normalize(hist, normhist, 0, 1, NORM_MINMAX);             // Normalized histogram
    cv::cuda::divide(normhist, cv::cuda::sum(normhist).val[0], prob);   // Probability
    cv::cuda::add(prob, 0.000000001, prob);                             // Add 0.000000001 to avoid errors calculating the logarithm
    cv::cuda::log(prob, logP);                                          // Natural logarithm of the probability
    cv::cuda::multiply(prob, logP, mult);
    cv::cuda::divide(mult, log(2), mult);
    float entropy = -1.0*cv::cuda::sum(mult).val[0];                    // Computes the entropy according to the Shannon Index  
    return entropy;
}

float averageEntropy_GPU(std::vector<cv::cuda::GpuMat> chanRGB) {           // Entropy considering the RGB components
    float AE = sqrt((pow(entropy_GPU(chanRGB[0]), 2) + pow(entropy_GPU(chanRGB[1]), 2) + pow(entropy_GPU(chanRGB[2]), 2))/3);
    return AE;
}

float averageContrast_GPU(std::vector<cv::cuda::GpuMat> chanRGB) {
    cv::cuda::GpuMat Grad[3], Grad2[3];
    int kernel_size = 3, scale = 1, delta = 0, ddepth = CV_32F;
    cv::cuda::Laplacian(chanRGB[0], Grad[0], ddepth, kernel_size, scale, delta, BORDER_DEFAULT);                    // RGB Gradients
    cv::cuda::Laplacian(chanRGB[1], Grad[1], ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
    cv::cuda::Laplacian(chanRGB[2], Grad[2], ddepth, kernel_size, scale, delta, BORDER_DEFAULT);

    cv::cuda::pow(Grad[0], 2, Grad2[0]), cv::cuda::pow(Grad[1], 2, Grad2[1]), cv::cuda::pow(Grad[2], 2, Grad2[2]);  // Grad^2
    cv::cuda::GpuMat Grads, sqrGrads;
    cv::cuda::add(Grad2[0], Grad2[1], Grads);
    cv::cuda::add(Grads, Grad2[2], Grads);
    cv::cuda::divide(Grads, 3, Grads);
    cv::cuda::sqrt(Grads, sqrGrads);
    cv::cuda::GpuMat C = sqrGrads(Rect(0, 0, sqrGrads.cols - 2, sqrGrads.rows - 2));
    float AC = cv::cuda::sum(C)[0] / ((sqrGrads.rows - 1)*(sqrGrads.cols - 1));                                     // Normalized sum of gradients
    return AC;
}

float averageLuminance_GPU(cv::cuda::GpuMat L) {
    float AL = cv::cuda::sum(L)[0] / L.total();                                 // Normalized sum of the luminance component
    return AL;
}

float getMSE_GPU(cv::cuda::GpuMat src, cv::cuda::GpuMat dst) {
    cv::cuda::GpuMat diff, diff2;
    cv::cuda::absdiff(src, dst, diff);
    cv::cuda::pow(diff, 2, diff2);
    float mse = cv::cuda::sum(diff2)[0] / src.total();                          // Mean Square Error
    return mse;
}

float sharpness_GPU(cv::cuda::GpuMat srcGPU) {
    cv::cuda::GpuMat padded, gpu_dft, dft_mag;
    int m = getOptimalDFTSize(srcGPU.rows);                 // Optimal Size to calculate the Discrete Fourier Transform 
    int n = getOptimalDFTSize(srcGPU.cols);
    cv::cuda::copyMakeBorder(srcGPU, padded, 0, m - srcGPU.rows, 0, n - srcGPU.cols, BORDER_CONSTANT, Scalar::all(0));  // Resize to optimal FFT size
    cv::cuda::GpuMat gpu_dft = cv::cuda::GpuMat(m, n, CV_32FC2);
    cv::cuda::dft(padded, gpu_dft, padded.size());          // Discrete Fourier Transform
    cv::cuda::magnitude(gpu_dft, dft_mag);                  // dft_mag = sqrt(Re(DFT(src))^2+Im(DFT(src))^2)
    double max;
    cv::cuda::minMax(dft_mag, NULL, &max);                  // Maximum value of the Fourier transform magnitude
    double thresh = max / 1000;                             // Threshold to calculate the IQM
    cv::cuda::GpuMat dft_thresh;
    cv::cuda::threshold(dft_mag, dft_thresh, thresh, 1, 0); // If dft_mag > thresh set to 1, else set to 0
    float TH = cv::cuda::countNonZero(dft_thresh);          // Number of pixels in dft_mag whose pixel value > thres
    float IQM = TH / srcGPU.total();                        // Computes the Image Sharpness
    return IQM;
}
#endif

/*********************************************************************/
/* Project: uw-img-proc									             */
/* Module: 	evaluationmetrics							             */
/* File: 	main.cpp										         */
/* Created:	09/12/2018				                                 */
/* Description:
	C++ Module of image quality metrics for image processing evaluation
 /********************************************************************/

 /********************************************************************/
 /* Created by:                                                      */
 /* Geraldine Barreto (@geraldinebc)                                 */
 /********************************************************************/

#define ABOUT_STRING "Image Quality Assessment Module"


// Time measurements
#define _VERBOSE_ON_
double t;	// Timing monitor

/*!
	@fn		int main(int argc, char* argv[])
	@brief	Main function
*/
int main(int argc, char *argv[]) {

	//*********************************************************************************
	/*	PARSER section */
	/*  Uses built-in OpenCV parsing method cv::CommandLineParser. It requires a string containing the arguments to be parsed from
		the command line. Further details can be obtained from opencv webpage
	*/
	String keys =
		"{@processed |<none> | Processed image file}"							// Processed image is the first argument (positional)
		"{@original  |<none> | Original image file}"							// Original image is the second argument (positional)
		"{m       |r      | Image Quality Metric to calculate}"					// Metric to calculate
		"{cuda    |       | Use CUDA or not (CUDA ON: 1, CUDA OFF: 0)}"         // Use CUDA (if available) (optional)
		"{save    |       | Save measurements or not (ON: 1, OFF: 0)}"			// Save measurements (optional)
		"{show    |       | Show result (ON: 1, OFF: 0)}"						// Show the measurements (optional)
		"{help h usage ?  |       | Print help message}";						// Show help (optional)

	CommandLineParser cvParser(argc, argv, keys);
	cvParser.about(ABOUT_STRING);	// Adds "about" information to the parser method

	//**************************************************************************
	std::cout << ABOUT_STRING << endl;
	std::cout << "Built with OpenCV " << CV_VERSION << endl;

	// If the number of arguments is lower than 4, or contains "help" keyword, then we show the help
	if (argc < 5 || cvParser.has("help")) {
		std::cout << endl << "C++ Module for calculating Image Quality Metrics" << endl;
		std::cout << endl << "Arguments are:" << endl;
		std::cout << "\t*Processed: Processed image name with path and extension" << endl;
		std::cout << "\t*Original: Original image name with path and extension" << endl;
		std::cout << "\t*-cuda=0 or -cuda=1 (ON: 1, OFF: 0)" << endl;
		std::cout << "\t*-save=0 or -save=1 (ON: 1, OFF: 0)" << endl;
		std::cout << "\t*-show=0 or -show=1 (ON: 1, OFF: 0)" << endl;
		std::cout << "\t*Argument 'm=<metrics>' is a string containing a list of the desired metrics to be calculated" << endl;
		std::cout << endl << "Complete options of evaluation metrics are:" << endl;
		std::cout << "\t-m=E for Entropy" << endl;
		std::cout << "\t-m=A for Average Entropy" << endl;
		std::cout << "\t-m=L for Average Luminance" << endl;
		std::cout << "\t-m=C for Average Contrast" << endl;
		std::cout << "\t-m=N for Normalized Neighborhood Function" << endl;
		std::cout << "\t-m=F for Comprehensive Assessment Function" << endl;
		std::cout << "\t-m=M for Mean Square Error" << endl;
		std::cout << "\t-m=P for Peak Signal to Noise Ratio" << endl;
		std::cout << "\t-m=S for Frequency Domain Image Sharpness Measure" << endl;
		std::cout << "\t-m=U for Feature detection using SURF" << endl;
		std::cout << "\t-m=H for Histogram" << endl;
		std::cout << "\t-m=X for All Metrics" << endl;
		std::cout << endl << "Example:" << endl;
		std::cout << "\tproc.jpg orig.jpg -cuda=0 -save=1 -show=0 -m=EM" << endl;
		std::cout << "\tThis will open 'proc.jpg' and 'orig.jpg' and calculate the Entropy and the MSE and save the results in a csv file" << endl << endl;
		return 0;
	}

	int CUDA = 0;											// Default option (running with CPU)
	int Save = 0;											// Default option (not saving results)
	int Show = 0;											// Default option (not showing results)

	std::string ProcessedFile = cvParser.get<cv::String>(0);// String containing the input file path+name+extension from cvParser function
	std::string OriginalFile = cvParser.get<cv::String>(1); // String containing the input file path+name+extension from cvParser function
	std::string metric = cvParser.get<cv::String>("m");		// Gets argument -m=x, where 'x' is the quality metric
	std::string implementation;								// CPU or GPU implementation
	Show = cvParser.get<int>("show");						// Gets argument -show=x, where 'x' defines if the results will be shown or not
	Save = cvParser.get<int>("save");						// Gets argument -save=x, where 'x' defines if the results will be saves

	// Check if any error occurred during parsing process
	if (!cvParser.check()) {
		cvParser.printErrors();
		return -1;
	}

	//************************************************************************************************
	int nCuda = -1;    // Defines number of detected CUDA devices. By default, -1 acting as error value
#if USE_GPU
	CUDA = cvParser.get<int>("cuda");	        // gets argument -cuda=x, where 'x' define to use CUDA or not
	nCuda = cuda::getCudaEnabledDeviceCount();	// Try to detect any existing CUDA device
	// Deactivate CUDA from parse
	if (CUDA == 0) {
		implementation = "CPU";
		cout << "CUDA deactivated" << endl;
		cout << "Exiting... use non-GPU version instead" << endl;
	}
	// Find CUDA devices
	else if (nCuda > 0) {
		implementation = "GPU";
		cuda::DeviceInfo deviceInfo;
		cout << "CUDA enabled devices detected: " << deviceInfo.name() << endl;
		cuda::setDevice(0);
	}
	else {
		CUDA = 0;
		implementation = "CPU";
		cout << "No CUDA device detected" << endl;
		cout << "Exiting... use non-GPU version instead" << endl;
	}
#endif
	//************************************************************************************************

	std::cout << endl << "********************************************************************************" << endl;
	std::cout << "Original Image: " << OriginalFile << endl;
	std::cout << "Processed Image: " << ProcessedFile << endl;

	Mat src, dst;
	src = imread(ProcessedFile, cv::IMREAD_COLOR);
	dst = imread(OriginalFile, cv::IMREAD_COLOR);

	if (src.empty() | dst.empty()) {
		std::cout << "Error occured when loading an image" << endl << endl;
		return -1;
	}

	if (Show == 0 & Save == 0) {
		std::cout << "Please choose one of the options: Show or Save" << endl << endl;
		return -1;
	}

	int num_metrics = metric.length();

	// Name for the output csv file where the results will be saved (where the processed images are)
	std::size_t pos;
	if (ProcessedFile.find(92) != std::string::npos) pos = ProcessedFile.find_last_of(92);	// If string contains '\' search for the last one
	else pos = ProcessedFile.find_last_of('/');												// If does not contain '\' search for the last '/'
	std::string Output = ProcessedFile.substr(0, pos + 1);
	std::string ext = "evaluation_metrics.csv";
	Output.insert(pos + 1, ext);
	ofstream file;
	std::string Name = ProcessedFile.substr(pos + 1);										// Name for the processed image

	std::cout << endl << "Calculating Image Quality Metrics" << endl << endl;

	// Start time measurement
	t = (double)getTickCount();

	// GPU Implementation
#if USE_GPU
	if (CUDA) {
		cv::cuda::GpuMat srcGPU, dstGPU, descriptorsGPU[2];
		srcGPU.upload(src);
		dstGPU.upload(dst);
		
		cv::cuda::GpuMat srcGPU_gray, dstGPU_gray, srcGPU_LAB, chanGPU_LAB[3], histRGB_GPU[3], histLAB_GPU[3], keypointsGPU, descriptorsGPU;
		std::vector<cv::cuda::GpuMat> chanRGB_GPU;
		cv::Mat hist_RGB[3], hist_LAB[3], RGB_hist[3], LAB_hist[3], histRGB, histLAB, hist;
		float entropy_src, AE_src = 0, AC_src = 0, AL_src = 0, NNF_src = 0, CAF_src, IQM_src, MSE = 0, psnr = 0;
		cv::cuda::SURF_CUDA surf;
		std::vector<KeyPoint> keypoint;
		std::size_t pos = ProcessedFile.find(".");
		std::string Histograms = ProcessedFile.substr(0, pos) + "_hist.jpg";
		Scalar colorA, colorB;

		// Now, according to parameters provided at CLI calling time
		file.open(Output, std::ios::app);	// Open the csv file in append mode
		file << endl << Name << ";";		// Adds endl to start adding information in the next line
		file.close();

		for (int nm = 0; nm < num_metrics; nm++) {
			char M = metric[nm];

			switch (M) {

			case 'E':	// Entropy
				if (srcGPU.empty()) cv::cuda::cvtColor(srcGPU, srcGPU_gray, COLOR_BGR2GRAY);
				entropy_src = entropy_GPU(srcGPU_gray);
				if (Show) std::cout << "Grayscale Entropy: " << fixed << setprecision(3) << entropy_src << endl;
				if (Save) {
					file.open(Output, std::ios::app);
					file << fixed << setprecision(3) << entropy_src << ";";
				}
			break;

			case 'A':	// Average Entropy
				if (chanRGB_GPU.empty()) cv::cuda::split(srcGPU, chanRGB_GPU);
				AE_src = averageEntropy_GPU(chanRGB_GPU);
				if (Show) std::cout << "Average Color Entropy: " << fixed << setprecision(3) << AE_src << endl;
				if (Save) {
					file.open(Output, std::ios::app);
					file << fixed << setprecision(3) << AE_src << ";";
				}
			break;

			case 'C':	//  Average Contrast
				if (chanRGB_GPU.empty()) cv::cuda::split(srcGPU, chanRGB_GPU);
				AC_src = averageContrast_GPU(chanRGB_GPU);
				if (Show) std::cout << "Average Contrast: " << fixed << setprecision(3) << AC_src << endl;
				if (Save) {
					file.open(Output, std::ios::app);
					file << fixed << setprecision(3) << AC_src << ";";
				}
			break;

			case 'L':	// Average Luminance
				if (srcGPU_LAB.empty()) {
					cv::cuda::cvtColor(srcGPU, srcGPU_LAB, COLOR_BGR2Lab);
					cv::cuda::split(srcGPU_LAB, chanGPU_LAB);
				}
				AL_src = averageLuminance_GPU(chanGPU_LAB[0]);
				if (Show) std::cout << "Average Luminance: " << fixed << setprecision(3) << AL_src << endl;
				if (Save) {
					file.open(Output, std::ios::app);
					file << fixed << setprecision(3) << AL_src << ";";
				}
			break;

			case 'N':	// Normalized Neighborhood Function
				if (!AL_src) {
					if (srcGPU_LAB.empty()) {
						cv::cuda::cvtColor(srcGPU, srcGPU_LAB, COLOR_BGR2Lab);
						cv::cuda::split(srcGPU_LAB, chanGPU_LAB);
					}
					AL_src = averageLuminance_GPU(chanGPU_LAB[0]);
				}
				NNF_src = getNNF(AL_src);
				if (Show) std::cout << "Normalized Neighborhood Function: " << fixed << setprecision(5) << NNF_src << endl;
				if (Save) {
					file.open(Output, std::ios::app);
					file << fixed << setprecision(5) << NNF_src << ";";
				}
			break;

			case 'F':	// Comprehensive Assesment Function
				if (!AE_src) {
					cv::cuda::split(srcGPU, chanRGB_GPU);
					AE_src = averageEntropy_GPU(chanRGB_GPU);
				}
				if (!AC_src) {
					cv::cuda::split(srcGPU, chanRGB_GPU);
					AC_src = averageContrast_GPU(chanRGB_GPU);
				}
				if (!NNF_src) {
					if (!AL_src) {
						if (srcGPU_LAB.empty()) {
							cv::cuda::cvtColor(srcGPU, srcGPU_LAB, COLOR_BGR2Lab);
							cv::cuda::split(srcGPU_LAB, chanGPU_LAB);
						}
						AL_src = averageLuminance_GPU(chanGPU_LAB[0]);
					}
					NNF_src = getNNF(AL_src);
				}
				CAF_src = getCAF(AE_src, AC_src, NNF_src);
				if (Show) std::cout << "Comprehensive Assessment Function: " << fixed << setprecision(3) << CAF_src << endl;
				if (Save) {
					file.open(Output, std::ios::app);
					file << fixed << setprecision(3) << CAF_src << ";";
				}
			break;

			case 'S':	// Image Sharpness in Frequency Domain
				if (srcGPU_gray.empty()) cv::cuda::cvtColor(srcGPU, srcGPU_gray, COLOR_BGR2GRAY);
				IQM_src = sharpness_GPU(srcGPU_gray);
				if (Show) std::cout << "Image Sharpness: " << fixed << setprecision(5) << IQM_src << endl;
				if (Save) {
					file.open(Output, std::ios::app);
					file << fixed << setprecision(5) << IQM_src << ";";
				}
			break;

			case 'U':	// Feature detection using SURF
				if (srcGPU_gray.empty()) cv::cuda::cvtColor(srcGPU, srcGPU_gray, COLOR_BGR2GRAY);
				surf(srcGPU, cv::cuda::GpuMat(), keypointsGPU, descriptorsGPU);
				surf.downloadKeypoints(keypointsGPU, keypoint);
				std::cout << endl << "Using SURF as detector" << endl;
				if (Show) std::cout << "Features Detected: " << keypoint.size() << endl;
				if (Save) {
					file.open(Output, std::ios::app);
					file << keypoint.size() << ";";
				}
			break;

			case 'M':	// Mean Square Error
				if (srcGPU_gray.empty()) cv::cuda::cvtColor(srcGPU, srcGPU_gray, COLOR_BGR2GRAY);
				if (dstGPU_gray.empty()) cv::cuda::cvtColor(dstGPU, dstGPU_gray, COLOR_BGR2GRAY);
				MSE = getMSE_GPU(srcGPU_gray, dstGPU_gray);
				if (Show) std::cout << fixed << setprecision(3) << "MSE: " << MSE << endl;
				if (Save) {
					file.open(Output, std::ios::app);
					file << int(MSE) << ";";
				}
			break;

			case 'P':	// Peak Signal to Noise Ratio
				if (!MSE) {
					if (srcGPU_gray.empty()) cv::cuda::cvtColor(srcGPU, srcGPU_gray, COLOR_BGR2GRAY);
					if (dstGPU_gray.empty()) cv::cuda::cvtColor(dstGPU, dstGPU_gray, COLOR_BGR2GRAY);
					MSE = getMSE_GPU(srcGPU_gray, dstGPU_gray);
				}
				psnr = getPSNR(MSE);
				if (Show) std::cout << "PSNR: " << fixed << setprecision(3) << psnr << endl;
				if (Save) {
					file.open(Output, std::ios::app);
					file << fixed << setprecision(3) << psnr << ";";
				}
			break;

			case 'H':	// Histogram
				if (chanRGB.empty()) cv::cuda::split(srcGPU, chanRGB_GPU);
				if (chanLAB->empty()) {
					cv::cuda::cvtColor(srcGPU, srcGPU_LAB, COLOR_BGR2Lab);
					cv::cuda::split(srcGPU_LAB, chanGPU_LAB);
				}

				// RGB histogram
				cv::cuda::calcHist(chanRGB_GPU[0], histRGB_GPU[0]);
				histRGB_GPU[0].download(hist_RGB[0]);
				RGB_hist[0] = printHist(hist_RGB[0], { 255,0,0 });
				cv::cuda::calcHist(chanRGB_GPU[1], histRGB_GPU[1]);
				histRGB_GPU[1].download(hist_RGB[1]);
				RGB_hist[1] = printHist(hist_RGB[1], { 0,255,0 });
				cv::cuda::calcHist(chanRGB_GPU[2], histRGB_GPU[2]);
				histRGB_GPU[2].download(hist_RGB[2]);
				RGB_hist[2] = printHist(hist_RGB[2], { 0,0,255 });
				cv::vconcat(RGB_hist[0], RGB_hist[1], histRGB);
				cv::vconcat(histRGB, RGB_hist[2], histRGB);

				// LAB histogram
				cv::cuda::calcHist(chanGPU_LAB[0], histLAB_GPU[0]);
				histLAB_GPU[0].download(hist_LAB[0]);
				LAB_hist[0] = printHist(hist_LAB[0], { 0,0,0 });
				if (mean(chanLAB[1])[0] > 127.5) colorA = { 150,15,235 };
				else colorA = { 75,155,10 };
				cv::cuda::calcHist(chanGPU_LAB[1], histLAB_GPU[1]);
				histLAB_GPU[1].download(hist_LAB[1]);
				LAB_hist[1] = printHist(hist_LAB[1], colorA);
				if (mean(chanLAB[2])[0] > 127.5) colorB = { 7,217,254 };
				else colorB = { 240,210,40 };
				cv::cuda::calcHist(chanGPU_LAB[2], histLAB_GPU[2]);
				histLAB_GPU[2].download(hist_LAB[2]);
				LAB_hist[2] = printHist(hist_LAB[2], colorB);
				cv::vconcat(LAB_hist[0], LAB_hist[1], histLAB);
				cv::vconcat(histLAB, LAB_hist[2], histLAB);

				// Histogram comparison
				cv::hconcat(histRGB, histLAB, hist);

				if (Show & !Save) {
					std::cout << "Showing histograms" << endl;
					namedWindow("Histograms", WINDOW_KEEPRATIO);
					imshow("Histograms", hist);
				}
				if (Save) {
					cv::imwrite(Histograms, hist);
					std::cout << "Histograms saved" << endl << endl;
				}
			break;

			case 'X':	// All Metrics
				if (OriginalFile.compare(ProcessedFile) == 0) metric = "EACLNFSH";
				else metric = "EACLNFSUMPH";
				num_metrics = metric.length();
				nm--;
			break;

			default:	// Unrecognized option
				std::cout << "Option " << M << " not recognized, skipping..." << endl << endl;
			break;
			}

			file.close();
		}
	}
#endif

	std::cout << "********************************************************************************" << endl;

	// CPU Implementation
	if (!CUDA) {

		cv::Mat graysrc, graydst, chanLAB[3], src_LAB, hist_RGB[3], hist_LAB[3], RGB_hist[3], LAB_hist[3], histRGB, histLAB, hist, descriptor;
		std::vector<Mat> chanRGB;
		float entropy_src, AE_src = 0, AC_src = 0, AL_src = 0, NNF_src = 0, CAF_src, IQM_src, MSE = 0, psnr = 0;
		int minHessian = 400;
		cv::Ptr<Feature2D> detector;
		std::vector<KeyPoint> keypoint;
		std::size_t pos = ProcessedFile.find(".");
		std::string Histograms = ProcessedFile.substr(0, pos) + "_hist.jpg";
		Scalar colorA, colorB;

		// Now, according to parameters provided at CLI calling time
		file.open(Output, std::ios::app);	// Open the csv file in append mode
		file << endl << Name << ";";		// Adds endl to start adding information in the next line
		file.close();

		for (int nm = 0; nm <= num_metrics; nm++) {
			char M = metric[nm];

			switch (M) {

				case 'E':	// Entropy
					if (graysrc.empty()) cvtColor(src, graysrc, COLOR_BGR2GRAY);
					entropy_src = entropy(graysrc);
					if (Show) std::cout << "Grayscale Entropy: " << fixed << setprecision(3) << entropy_src << endl;
					if (Save) {
						file.open(Output, std::ios::app);
						file << fixed << setprecision(3) << entropy_src << ";";
					}
				break;

				case 'A':	// Average Entropy
					if (chanRGB.empty()) split(src, chanRGB);
					AE_src = averageEntropy(chanRGB);
					if (Show) std::cout << "Average Color Entropy: " << fixed << setprecision(3) << AE_src << endl;
					if (Save) {
						file.open(Output, std::ios::app);
						file << fixed << setprecision(3) << AE_src << ";";
					}
				break;

				case 'C':	//  Average Contrast
					if (chanRGB.empty()) split(src, chanRGB);
					AC_src = averageContrast(chanRGB);
					if (Show) std::cout << "Average Contrast: " << fixed << setprecision(3) << AC_src << endl;
					if (Save) {
						file.open(Output, std::ios::app);
						file << fixed << setprecision(3) << AC_src << ";";
					}
				break;

				case 'L':	// Average Luminance
					if (src_LAB.empty()) {
						cvtColor(src, src_LAB, COLOR_BGR2Lab);
						split(src_LAB, chanLAB);
					}
					AL_src = averageLuminance(chanLAB[0]);
					if (Show) std::cout << "Average Luminance: " << fixed << setprecision(3) << AL_src << endl;
					if (Save) {
						file.open(Output, std::ios::app);
						file << fixed << setprecision(3) << AL_src << ";";
					}
				break;

				case 'N':	// Normalized Neighborhood Function
					if (!AL_src) {
						if (src_LAB.empty()) {
							cvtColor(src, src_LAB, COLOR_BGR2Lab);
							split(src_LAB, chanLAB);
						}
						AL_src = averageLuminance(chanLAB[0]);
					}
					NNF_src = getNNF(AL_src);
					if (Show) std::cout << "Normalized Neighborhood Function: " << fixed << setprecision(5) << NNF_src << endl;
					if (Save) {
						file.open(Output, std::ios::app);
						file << fixed << setprecision(5) << NNF_src << ";";
					}
				break;

				case 'F':	// Comprehensive Assesment Function
					if (!AE_src) {
						split(src, chanRGB);
						AE_src = averageEntropy(chanRGB);
					}
					if (!AC_src) {
						split(src, chanRGB);
						AC_src = averageContrast(chanRGB);
					}
					if (!NNF_src) {
						if (!AL_src) {
							if (src_LAB.empty()) {
								cvtColor(src, src_LAB, COLOR_BGR2Lab);
								split(src_LAB, chanLAB);
							}
							AL_src = averageLuminance(chanLAB[0]);
						}
						NNF_src = getNNF(AL_src);
					}
					CAF_src = getCAF(AE_src, AC_src, NNF_src);
					if (Show) std::cout << "Comprehensive Assessment Function: " << fixed << setprecision(3) << CAF_src << endl;
					if (Save) {
						file.open(Output, std::ios::app);
						file << fixed << setprecision(3) << CAF_src << ";";
					}
				break;

				case 'S':	// Image Sharpness in Frequency Domain
					if (graysrc.empty()) cvtColor(src, graysrc, COLOR_BGR2GRAY);
					IQM_src = sharpness(graysrc);
					if (Show) std::cout << "Image Sharpness: " << fixed << setprecision(5) << IQM_src << endl;
					if (Save) {
						file.open(Output, std::ios::app);
						file << fixed << setprecision(5) << IQM_src << ";";
					}
				break;

				case 'U':	// Feature detection using SURF
					if (graysrc.empty()) cvtColor(src, graysrc, COLOR_BGR2GRAY);
					detector = SURF::create(minHessian);
					detector->detectAndCompute(graysrc, Mat(), keypoint, descriptor);
					if (Show) std::cout << "Features Detected: " << keypoint.size() << endl;
					if (Save) {
						file.open(Output, std::ios::app);
						file << keypoint.size() << ";";
					}
				break;

				case 'M':	// Mean Square Error
					if (graysrc.empty()) cvtColor(src, graysrc, COLOR_BGR2GRAY);
					if (graydst.empty()) cvtColor(dst, graydst, COLOR_BGR2GRAY);
					MSE = getMSE(graysrc, graydst);
					if (Show) std::cout << fixed << setprecision(3) << "MSE: " << MSE << endl;
					if (Save) {
						file.open(Output, std::ios::app);
						file << int(MSE) << ";";
					}
					break;

				case 'P':	// Peak Signal to Noise Ratio
					if (!MSE) {
						if (graysrc.empty()) cvtColor(src, graysrc, COLOR_BGR2GRAY);
						if (graydst.empty()) cvtColor(dst, graydst, COLOR_BGR2GRAY);
						MSE = getMSE(graysrc, graydst);
					}
					psnr = getPSNR(MSE);
					if (Show) std::cout << "PSNR: " << fixed << setprecision(3) << psnr << endl;
					if (Save) {
						file.open(Output, std::ios::app);
						file << fixed << setprecision(3) << psnr << ";";
					}
					break;

				case 'H':	// Histogram
					if (chanRGB.empty()) split(src, chanRGB);
					if (chanLAB->empty()) {
						cvtColor(src, src_LAB, COLOR_BGR2Lab);
						split(src_LAB, chanLAB);
					}

					// RGB histogram
					getHistogram(&chanRGB[0], &hist_RGB[0]);
					RGB_hist[0] = printHist(hist_RGB[0], { 255,0,0 });
					getHistogram(&chanRGB[1], &hist_RGB[1]);
					RGB_hist[1] = printHist(hist_RGB[1], { 0,255,0 });
					getHistogram(&chanRGB[2], &hist_RGB[2]);
					RGB_hist[2] = printHist(hist_RGB[2], { 0,0,255 });
					cv::vconcat(RGB_hist[0], RGB_hist[1], histRGB);
					cv::vconcat(histRGB, RGB_hist[2], histRGB);

					// LAB histogram
					getHistogram(&chanLAB[0], &hist_LAB[0]);
					LAB_hist[0] = printHist(hist_LAB[0], { 0,0,0 });
					getHistogram(&chanLAB[1], &hist_LAB[1]);
					if (mean(chanLAB[1])[0] > 127.5) colorA = { 150,15,235 };
					else colorA = { 75,155,10 };
					LAB_hist[1] = printHist(hist_LAB[1], colorA);
					getHistogram(&chanLAB[2], &hist_LAB[2]);
					if (mean(chanLAB[2])[0] > 127.5) colorB = { 7,217,254 };
					else colorB = { 240,210,40 };
					LAB_hist[2] = printHist(hist_LAB[2], colorB);
					cv::vconcat(LAB_hist[0], LAB_hist[1], histLAB);
					cv::vconcat(histLAB, LAB_hist[2], histLAB);

					// Histogram comparison
					cv::hconcat(histRGB, histLAB, hist);

					if (Show & !Save) {
						std::cout << "Showing histograms" << endl;
						namedWindow("Histograms", WINDOW_KEEPRATIO);
						imshow("Histograms", hist);
					}
					if (Save) {
						cv::imwrite(Histograms, hist);
						std::cout << "Histograms saved" << endl << endl;
					}
				break;

				case 'X':	// All Metrics
					if (OriginalFile.compare(ProcessedFile) == 0) metric = "EACLNFSUH";
					else metric = "EACLNFSUMPH";
					num_metrics = metric.length();
					nm--;
				break;

				default:	// Unrecognized option
					std::cout << "Option " << M << " not recognized, skipping..." << endl << endl;
				break;
			}

			file.close();
		}
	}

	//  End time measurement
	t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
	std::cout << "Execution Time" << implementation << ": " << t << " ms " << endl;

	if (Save) std::cout << endl << "Evaluation metrics saved in " << Output << endl;

	waitKey(0);
	return 0;
}