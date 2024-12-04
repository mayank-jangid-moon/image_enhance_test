/********************************************************************/
/* Project: uw_img_proc									            */
/* Module:  dehazing								                */
/* File: 	dehazing.h									            */
/* Created:	05/02/2019				                                */
/* Description:
	C++ Module for image dehazing using Gao's Bright Channel Prior	*/
 /*******************************************************************/

 /*******************************************************************/
 /* Created by:                                                     */
 /* Geraldine Barreto (@geraldinebc)                                */
 /*******************************************************************/

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
#include <opencv2/ximgproc.hpp>

// C++ namespaces
using namespace cv;
using namespace cuda;
using namespace std;
using namespace ximgproc;

#if USE_GPU
#include <opencv2/cudafilters.hpp>
#include "opencv2/cudaimgproc.hpp"
#endif

/*
	@brief		Generates the Bright Channel Image of an underwater image
	@function	cv::Mat brightChannel(vector<Mat_<uchar>> channels, int size)
*/
cv::Mat brightChannel(vector<Mat_<uchar>> channels, int size);

/*
	@brief		Generates the Maximum Color Difference Image of an underwater image
	@function	cv::Mat mcd = maxColDiff(src_chan)
*/
cv::Mat maxColDiff(vector<Mat_<uchar>> channels);

/*
	@brief		Rectifies the Bright Channel Image
	@function	cv::Mat  rectify(cv::Mat S, cv::Mat bc, cv::Mat mcd)
*/
cv::Mat  rectify(cv::Mat S, cv::Mat bc, cv::Mat mcd);

void getHistogram(cv::Mat *channel, cv::Mat *hist);

/*
	@brief		Estimates the atmospheric light of an underwater image
	@function	vector<uchar> lightEstimation(cv::Mat src_gray, int size, cv::Mat bc, vector<Mat_<uchar>> channels)
*/
vector<uchar> lightEstimation(cv::Mat src_gray, int size, cv::Mat bc, vector<Mat_<uchar>> channels);

/*
	@brief		Computes the transmittance image
	@function	cv::Mat transmittance(cv::Mat correct, vector<uchar> A)
*/
cv::Mat transmittance(cv::Mat correct, vector<uchar> A);

/*
	@brief		Dehazes the underwater image
	@function	cv::Mat dehaze(vector<Mat_<float>> channels, vector<uchar> A, cv::Mat trans)
*/
cv::Mat dehaze(vector<Mat_<float>> channels, vector<uchar> A, cv::Mat trans);

#if USE_GPU
cv::cuda::GpuMat brightChannel_GPU(std::vector<cv::cuda::GpuMat> channels, int size);

cv::cuda::GpuMat maxColDiff_GPU(std::vector<cv::cuda::GpuMat> channels);

cv::cuda::GpuMat rectify_GPU(cv::cuda::GpuMat S, cv::cuda::GpuMat bc, cv::cuda::GpuMat mcd);

std::vector<uchar> lightEstimation_GPU(cv::Mat src_gray, int size, cv::cuda::GpuMat bright_chan, std::vector<cv::Mat> channels);

cv::cuda::GpuMat transmittance_GPU(cv::cuda::GpuMat correct, std::vector<uchar> A);

cv::cuda::GpuMat dehaze_GPU(std::vector<cv::cuda::GpuMat> channels, std::vector<uchar> A, cv::cuda::GpuMat trans);
#endif


/********************************************************************/
/* Project: uw_img_proc									            */
/* Module:  dehazing								                */
/* File: 	dehazing.cpp								            */
/* Created:	05/02/2019				                                */
/* Description:
	C++ Module for image dehazing using Gao's Bright Channel Prior	*/
 /*******************************************************************/

 /*******************************************************************/
 /* Created by:                                                     */
 /* Geraldine Barreto (@geraldinebc)                                */
 /*******************************************************************/

/// Include auxiliary utility libraries

cv::Mat brightChannel(std::vector<cv::Mat_<uchar>> channels, int size) {					// Generates the Bright Channel Image
	cv::Mat maxRGB = max(max(channels[0], channels[1]), channels[2]);						// Maximum Color Image
	cv::Mat element, bright_chan;
	element = getStructuringElement(MORPH_RECT, Size(size, size), Point(-1, -1));			// Maximum filter
	dilate(maxRGB, bright_chan, element);													// Dilates the maxRGB image
	return bright_chan;
}

cv::Mat maxColDiff(std::vector<cv::Mat_<uchar>> channels) {									// Generates the Maximum Color Difference Image
	vector<float> means;
	means.push_back(mean(channels[0])[0]);
	means.push_back(mean(channels[1])[0]);
	means.push_back(mean(channels[2])[0]);
	cv::Mat sorted;
	sortIdx(means, sorted, SORT_EVERY_ROW + SORT_ASCENDING);								// Orders the mean of the channels from low to high

	cv::Mat cmin = channels[sorted.at<int>(0, 0)];
	cv::Mat cmid = channels[sorted.at<int>(0, 1)];
	cv::Mat cmax = channels[sorted.at<int>(0, 2)];
	
	cv::Mat a, b, mcd;
	a = max(cmax - cmin, 0);																// Calculates the maximum values for the MCD image
	b = max(cmid - cmin, 0);
	mcd = 255 - max(a,b);
	return mcd;
}

cv::Mat rectify(cv::Mat S, cv::Mat bc, cv::Mat mcd) {										// Rectifies the Bright Channel Image
	double lambda;
	minMaxLoc(S, NULL, &lambda);															// Maximum value of the Saturation channel
	lambda = lambda / 255.0;																// Normalization for the next step
	cv::Mat correct;
	addWeighted(bc, lambda, mcd, 1.0 - lambda, 0.0, correct);
	return correct;
}

void getHistogram(cv::Mat *channel, cv::Mat *hist) {								// Computes the histogram of a single channel
	int histSize = 256;
	float range[] = { 0, 256 };														// The histograms ranges from 0 to 255
	const float* histRange = { range };
	calcHist(channel, 1, 0, Mat(), *hist, 1, &histSize, &histRange, true, false);
}

std::vector<uchar> lightEstimation(cv::Mat src_gray, int size, cv::Mat bright_chan, std::vector<Mat_<uchar>> channels) {	// Estimates the atmospheric light
	cv::Mat variance, thresholded;
	sqrBoxFilter(src_gray, variance, -1, Size(size, size), Point(-1, -1), true, BORDER_DEFAULT);	// Variance Filter
	cv::Mat histogram;
	getHistogram(&bright_chan, &histogram);
	float percent = 1.0, sum = 0.0, thresh = -1.0;
	int i = 0;
	while (sum <= bright_chan.total()*percent / 100) {
		sum += histogram.at<float>(i, 0);
		thresh++;																					// Finds the threshold value for the 1% darkest pixels in the BC
		i++;
	}
	threshold(bright_chan, thresholded, thresh, 255, 1);											// If the pixels are higher than thresh Mask -> 0 else -> 1
	Point minLoc;
	minMaxLoc(variance, NULL, NULL, &minLoc, NULL, thresholded);									// Finds the variance darkest pixel using the calculated mask
	std::vector<uchar> A;
	for (int i = 0; i < 3; i++) A.push_back(channels[i].at<uchar>(minLoc.y, minLoc.x));
	////PARA VISUALIZAR
	//src_gray.at<char>(minLoc.y, minLoc.x) = 255;
	//namedWindow("A point", WINDOW_KEEPRATIO);
	//imshow("A point", thresholded);
	return A;
}

cv::Mat transmittance(cv::Mat correct, std::vector<uchar> A) {						// Computes the Transmittance Image
	correct.convertTo(correct, CV_32F);
	cv::Mat t[3], acc(correct.size(), CV_32F, Scalar(0));
	for (int i = 0; i < 3; i++) {
		t[i] = 255.0 * ( (correct - A[i]) / (255.0 - A[i]) );
		accumulate(t[i], acc);
	}
	cv::Mat trans = acc/3;
	trans.convertTo(trans, CV_8U);
	return trans;
}

cv::Mat dehaze(vector<Mat_<float>> channels, std::vector<uchar> A, cv::Mat trans) {	// Restores the Underwater Image using the Bright Channel Prior
	trans.convertTo(trans, CV_32F, 1.0/255.0);
	channels[0] = 255.0 - ((channels[0] - (A[0] * (1.0 - trans))) / trans);
	channels[1] = 255.0 - ((channels[1] - (A[1] * (1.0 - trans))) / trans);
	channels[2] = (channels[2] - A[2]) / trans + A[2];
	cv::Mat dehazed, dst;
	merge(channels, dehazed);
	dehazed.convertTo(dst, CV_8U);
	return dst;
}

#if USE_GPU
cv::cuda::GpuMat brightChannel_GPU(std::vector<cv::cuda::GpuMat> channels, int size) {				// Generates the Bright Channel Image
	cv::cuda::GpuMat maxRGB = cv::cuda::max(cv::cuda::max(channels[0], channels[1]), channels[2]);	// Maximum Color Image
	cv::Mat element = getStructuringElement(MORPH_RECT, Size(size, size), Point(-1, -1));			// Maximum filter
	cv::cuda::GpuMat bright_chan;
	cv::cuda::dilate(maxRGB, bright_chan, element);													// Dilates the maxRGB image
	return bright_chan;
}

cv::cuda::GpuMat maxColDiff_GPU(std::vector<cv::cuda::GpuMat> channels) {						// Generates the Maximum Color Difference Image
	vector<float> means;
	means.push_back(mean(channels[0])[0]);
	means.push_back(mean(channels[1])[0]);
	means.push_back(mean(channels[2])[0]);
	cv::Mat sorted;
	sortIdx(means, sorted, SORT_EVERY_ROW + SORT_ASCENDING);									// Orders the mean of the channels from low to high

	cv::cuda::GpuMat cmin = channels[sorted.at<int>(0, 0)];
	cv::cuda::GpuMat cmid = channels[sorted.at<int>(0, 1)];
	cv::cuda::GpuMat cmax = channels[sorted.at<int>(0, 2)];

	cv::cuda::GpuMat diff1, diff2, a, b, maxi, mcd;
	cv::cuda::subtract(cmax, cmin, diff1);
	cv::cuda::subtract(cmid, cmin, diff2);
	cv::cuda::max(diff1, 0, a);
	cv::cuda::max(diff2, 0, b);
	cv::cuda::max(a, b, maxi);
	cv::cuda::subtract(255, maxi, mcd);															// Calculates the maximum values for the MCD image
	return mcd;
}

cv::cuda::GpuMat rectify_GPU(cv::cuda::GpuMat S, cv::cuda::GpuMat bc, cv::cuda::GpuMat mcd) {	// Rectifies the Bright Channel Image
	double lambda;
	cv::cuda::minMax(S, NULL, &lambda);															// Maximum value of the Saturation channel
	lambda = lambda / 255.0;																	// Normalization for the next step
	cv::cuda::GpuMat correct;
	cv::cuda::addWeighted(bc, lambda, mcd, 1.0 - lambda, 0.0, correct);
	return correct;
}

std::vector<uchar> lightEstimation_GPU(cv::Mat src_gray, int size, cv::cuda::GpuMat bright_chan, std::vector<cv::Mat> channels) {	// Estimates the atmospheric light
	cv::Mat variance, histogram;
	sqrBoxFilter(src_gray, variance, -1, Size(size, size), Point(-1, -1), true, BORDER_DEFAULT);		// Variance Filter
	cv::cuda::GpuMat histogramGPU;
	cv::cuda::calHist(bright_chan, histogramGPU);
	histogramGPU.download(histogram);
	float percent = 1.0, sum = 0.0, thresh = -1.0;
	int i = 0;
	while (sum <= bright_chan.total()*percent / 100) {
		sum += histogram.at<float>(i, 0);
		thresh++;
		i++;
	}
	cv::cuda::GpuMat thresholded, varianceGPU;
	cv::cuda::threshold(bright_chan, thresholded, thresh, 255, 1);
	Point minLoc;
	varianceGPU.upload(variance);
	cv::cuda::minMaxLoc(varianceGPU, NULL, NULL, &minLoc, NULL, thresholded);
	std::vector<uchar> A;
	for (int i = 0; i < 3; i++) A.push_back(channels[i].at<uchar>(minLoc.y, minLoc.x));
	return A;
}

cv::cuda::GpuMat transmittance_GPU(cv::cuda::GpuMat correct, std::vector<uchar> A) {			// Computes the Transmittance Image
	correct.convertTo(correct, CV_32F);
	cv::cuda::GpuMat t[3], sub, acc(correct.size(), CV_32F, Scalar(0));
	for (int i = 0; i < 3; i++) {
		cv::cuda::subtract(correct, A[i], sub);
		cv::cuda::multiply(sub, 255.0/(255.0 - A[i]), t[i]);
		cv::cuda::add(acc, t[i], acc);
	}
	cv::cuda::GpuMat trans;
	cv::cuda::divide(acc, 3, trans, CV_8U);
	return trans;
}

cv::cuda::GpuMat dehaze_GPU(std::vector<cv::cuda::GpuMat> channels, std::vector<uchar> A, cv::cuda::GpuMat trans) {	// Restores the Underwater Image using the BCP
	trans.convertTo(trans, CV_32F, 1.0 / 255.0);
	cv::cuda::GpuMat B[3], C[3];
	cv::cuda::addWeighted(channels[0], 1, -A[0], 1.0 - trans, 0.0, B[0]);
	cv::cuda::subtract(255.0, B[0], C[0]);
	cv::cuda::divide(C[0], trans, channels[0]);
	cv::cuda::addWeighted(channels[1], 1, -A[1], 1.0 - trans, 0.0, B[1]);
	cv::cuda::substract(255.0, B[1], C[1]);
	cv::cuda::divide(C[1], trans, channels[1]);
	cv::cuda::subtract(channels[2], A[2], B[2]);
	cv::cuda::divide(B[2], trans, C[2]);
	cv::cuda::add(C[2], A[2], channels[2]);
	cv::cuda::GpuMat dehazed, dst;
	cv::cuda::merge(channels, dehazed);
	dehazed.convertTo(dst, CV_8U);
	return dst;
}
#endif


/********************************************************************/
/* Project: uw_img_proc									            */
/* Module:  dehazing								                */
/* File: 	main.cpp									            */
/* Created:	05/02/2019				                                */
/* Description:
	C++ Module for image dehazing using Gao's Bright Channel Prior	*/
 /*******************************************************************/

 /*******************************************************************/
 /* Created by:                                                     */
 /* Geraldine Barreto (@geraldinebc)                                */
 /*******************************************************************/

#define ABOUT_STRING "Dehazing module using the Bright Channel Prior"


// Time measurements
#define _VERBOSE_ON_
double t;	// Timing monitor


int main(int argc, char *argv[]) {

	//*********************************************************************************
	/*	PARSER section */
	/*  Uses built-in OpenCV parsing method cv::CommandLineParser. It requires a string containing the arguments to be parsed from
		the command line. Further details can be obtained from opencv webpage
	*/
	String keys =
		"{@input  |<none> | Input image file}"									// Input image is the first argument (positional)
		"{@output |<none> | Output image file}"									// Output prefix is the second argument (positional)
		"{show    |       | Show resulting image (ON: 1, OFF: 0)}"				// Show resulting image (optional)
		"{cuda    |       | Use CUDA or not (CUDA ON: 1, CUDA OFF: 0)}"         // Use CUDA (optional)
		"{time    |       | Show time measurements or not (ON: 1, OFF: 0)}"		// Show time measurements (optional)
		"{help h usage ?  |       | Print help message}";						// Show help (optional)

	CommandLineParser cvParser(argc, argv, keys);
	cvParser.about(ABOUT_STRING);												// Adds "about" information to the parser method

	//*********************************************************************************
	std::cout << ABOUT_STRING << endl;
	std::cout << "Built with OpenCV " << CV_VERSION << endl;

	// If the number of arguments is lower than 5, or contains "help" keyword, then we show the help
	if (argc < 5 || cvParser.has("help")) {
		std::cout << endl << "C++ implementation of Gao's dehazing algorithm" << endl;
		std::cout << endl << "Arguments are:" << endl;
		std::cout << "\t*Input: Input image name with path and extension" << endl;
		std::cout << "\t*Output: Output image name with path and extension" << endl;
		std::cout << "\t*-show=0 or -show=1 (ON: 1, OFF: 0)" << endl;
		std::cout << "\t*-cuda=0 or -cuda=1 (ON: 1, OFF: 0)" << endl;
		std::cout << "\t*-time=0 or -time=1 (ON: 1, OFF: 0)" << endl;
		std::cout << endl << "\tExample:" << endl;
		std::cout << "\t input.jpg output.jpg -show=0 -cuda=0 -time=0" << endl;
		std::cout << "\tThis will open 'input.jpg' dehaze the image and save the result in 'output.jpg'" << endl << endl;
		return 0;
	}

	int CUDA = 0;										// Default option (running with CPU)
	int Time = 0;                                       // Default option (not showing time)
	int Show = 0;										// Default option (not showing results)

	std::string InputFile = cvParser.get<cv::String>(0); // String containing the input file path+name+extension from cvParser function
	std::string OutputFile = cvParser.get<cv::String>(1);// String containing the input file path+name+extension from cvParser function
	std::string implementation;							 // CPU or GPU implementation
	Show = cvParser.get<int>("show");					 // Gets argument -show=x, where 'x' defines if the results will show or not
	Time = cvParser.get<int>("time");	                 // Gets argument -time=x, where 'x' defines ifexecution time will show or not

	// Check if any error occurred during parsing process
	if (!cvParser.check()) {
		cvParser.printErrors();
		return -1;
	}

	//************************************************************************************************
	int nCuda = -1;    //Defines number of detected CUDA devices. By default, -1 acting as error value
#if USE_GPU
	CUDA = cvParser.get<int>("cuda");	        // Gets argument -cuda=x, where 'x' define to use CUDA or not
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

	std::cout << endl << "************************************************************************" << endl;
	std::cout << endl << "Input: " << InputFile << endl;
	std::cout << "Output: " << OutputFile << endl;

	cv::Mat src, dst;
	src = imread(InputFile, cv::IMREAD_COLOR);

	if (src.empty()){
		std::cout << endl << "Error occured when loading the image" << endl << endl;
		return -1;
	}

	std::cout << endl << "Aplying dehazing algorithm" << endl;

	// Start time measurement
	if (Time) t = (double)getTickCount();

	// GPU Implementation
#if USE_GPU
	if (CUDA) {
		cv::cuda::GpuMat srcGPU, new_chanGPU, dstGPU;
		srcGPU.upload(src);

		// Split the RGB channels (BGR for OpenCV)
		std::vector<cv::cuda::GpuMat> srcGPU_chan, newGPU_chan;
		cv::cuda::split(srcGPU, srcGPU_chan);

		// Compute the new channels for the dehazing process
		cv::cuda::subtract(255, srcGPU_chan[0], new_chanGPU);
		newGPU_chan.push_back(new_chanGPU);
		cv::cuda::subtract(255, srcGPU_chan[1], new_chanGPU);
		newGPU_chan.push_back(new_chanGPU);
		newGPU_chan.push_back(srcGPU_chan[2]);

		// Compute the bright channel image
		int size = sqrt(src.total()) / 50;
		cv::cuda::GpuMat bright_chan = brightChannel_GPU(newGPU_chan, size);

		// Compute the maximum color difference
		cv::cuda::GpuMat mcd = maxColDiff_GPU(srcGPU_chan);

		// Rectify the bright channel image
		cv::cuda::GpuMat src_HSV, HSV_chan[3];
		cv::cuda::cvtColor(srcGPU, src_HSV, COLOR_BGR2HSV);
		cv::cuda::split(src_HSV, HSV_chan);
		cv::cuda::GpuMat rectified = rectify_GPU(HSV_chan[1], bright_chan, mcd);

		// Estimate the atmospheric light
		cv::Mat src_gray;
		cv::cvtColor(src, src_gray, COLOR_BGR2GRAY);
		vector<uchar> A;
		A = lightEstimation_GPU(src_gray, size, bright_chan, newGPU_chan);

		// Compute the transmittance image
		cv::cuda::GpuMat trans = transmittance(rectified, A);

		// Refine the transmittance image
		cv::Mat filtered;
		trans.download(transmit);
		cv::cuda::guidedFilter(src_gray, transmit, filtered, 30, 0.001, -1);

		// Dehaze the image channels
		std::vector<cv::cuda::GpuMat> chan_dehazed;
		chan_dehazed.push_back(newGPU_chan[0]);
		chan_dehazed.push_back(newGPU_chan[1]);
		chan_dehazed.push_back(newGPU_chan[2]);

		trans.upload(filtered);
		dstGPU = dehaze(chan_dehazed, A, trans);

		dstGPU.download(dst);
	}
#endif

// CPU Implementation
if (!CUDA) {
	// Split the RGB channels (BGR for OpenCV)
	std::vector<cv::Mat_<uchar>> src_chan, new_chan;
	split(src, src_chan);

	// Compute the new channels for the dehazing process
	new_chan.push_back(255 - src_chan[0]);
	new_chan.push_back(255 - src_chan[1]);
	new_chan.push_back(src_chan[2]);

	// Compute the bright channel image
	int size = sqrt(src.total()) / 50;						// Making the size bigger creates halos around objects
	cv::Mat bright_chan = brightChannel(new_chan, size);

	// Compute the maximum color difference
	cv::Mat mcd = maxColDiff(src_chan);

	// Rectify the bright channel image
	cv::Mat src_HSV, S;
	cv::cvtColor(src, src_HSV, COLOR_BGR2HSV);
	extractChannel(src_HSV, S, 1);
	cv::Mat rectified = rectify(S, bright_chan, mcd);

	// Estimate the atmospheric light
	cv::Mat src_gray;
	cv::cvtColor(src, src_gray, COLOR_BGR2GRAY);
	std::vector<uchar> A;
	A = lightEstimation(src_gray, size, bright_chan, new_chan);

	// Compute the transmittance image
	cv::Mat trans = transmittance(rectified, A);

	// Refine the transmittance image
	cv::Mat filtered;
	guidedFilter(src_gray, trans, filtered, 30, 0.001, -1);

	// Dehaze the image channels
	std::vector<cv::Mat_<float>> chan_dehazed;
	chan_dehazed.push_back(new_chan[0]);
	chan_dehazed.push_back(new_chan[1]);
	chan_dehazed.push_back(new_chan[2]);
	dst = dehaze(chan_dehazed, A, filtered);
}

//  End time measurement (Showing time results is optional)
if (Time) {
	t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
	std::cout << endl << "Execution Time" << implementation << ": " << t << " ms " << endl;
	
	// Name for the output csv file where the execution time will be saved
	std::size_t pos;
	if (OutputFile.find(92) != std::string::npos) pos = OutputFile.find_last_of(92);	// If string contains '\' search for the last one
	else pos = OutputFile.find_last_of('/');											// If does not containn '\' search for the last '/'
	std::string Output = OutputFile.substr(0, pos + 1);
	std::string ext = "execution_time.csv";
	Output.insert(pos + 1, ext);
	ofstream file;
	file.open(Output, std::ios::app);
	file << endl << OutputFile << ";" << src.rows << ";" << src.cols << ";" << t;
}

std::cout << endl << "Saving processed image" << endl;
imwrite(OutputFile, dst);

if (Show) {
	std::cout << endl << "Showing image comparison" << endl;
	cv::Mat comparison;
	hconcat(src, dst, comparison);
	namedWindow("Comparison", WINDOW_KEEPRATIO);
	imshow("Comparison", comparison);
}

waitKey(0);
return 0;
}