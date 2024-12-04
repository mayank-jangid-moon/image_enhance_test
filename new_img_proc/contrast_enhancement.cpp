/*********************************************************************/
/* Project: uw-img-proc									             */
/* Module:  contrastenhancement				  		                 */
/* File: 	contrastenhancement.h					         	     */
/* Created:	01/03/2018				                                 */
/* Description:
	C++ Module for contrast enhancement using histogram stretching   
	and equalization												 */
 /********************************************************************/

 /********************************************************************/
 /* Created by:                                                      */
 /* Geraldine Barreto (@geraldinebc)                                 */
/*********************************************************************/

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

// C++ namespaces
using namespace cv;
using namespace cuda;
using namespace std;

/// CUDA specific libraries
#if USE_GPU
#include <opencv2/cudafilters.hpp>
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#endif

/*
	@brief		Enhances an image with the Simplest Color Balance method
	@function	cv::Mat simplestColorBalance(cv::Mat src, float percent)
*/
cv::Mat simplestColorBalance(cv::Mat src, float percent);

/*
	@brief		Stretches the histogram of one image channel in a specific direction (right 0, both sides 1 or left 2)
	@function	cv::Mat histStretch(cv::Mat src, float percent, int direction);
*/
cv::Mat histStretch(cv::Mat src, float percent, int direction);

/*
	@brief		Enhances the contrast of an image using the Integrated Color Model by Iqbal et al. based on histogram stretching
	@function	cv::Mat ICM(cv::Mat src, float percent)
*/
cv::Mat ICM(cv::Mat src, float percent);

/*
	@brief		Enhances the contrast of an image using histogram stretching and the Von Kries Hypotesis
	@function	cv::Mat UCM(cv::Mat src, float percent);
*/
cv::Mat UCM(cv::Mat src, float percent);

/*
	@brief		Equalizes the histogram of one image channel using Rayleigh's distribution
	@function	cv::Mat rayleighEqualization(cv::Mat src)
*/
cv::Mat rayleighEqualization(cv::Mat src);


/*
	@brief		Computes the histogram of a single channel
	@function	void getHistogram(cv::Mat *channel, cv::Mat *hist)
*/
void getHistogram(cv::Mat *channel, cv::Mat *hist);


#if USE_GPU

	cv::cuda::GpuMat simplestColorBalance_GPU(cv::cuda::GpuMat srcGPU, float percent);

	cv::cuda::GpuMat histStretch_GPU(cv::cuda::GpuMat srcGPU, float percent, int direction);

	cv::cuda::GpuMat ICM_GPU(cv::cuda::GpuMat srcGPU, float percent);

	cv::cuda::GpuMat UCM_GPU(cv::cuda::GpuMat srcGPU, float percent);

	cv::cuda::GpuMat rayleighEqualization_GPU(cv::cuda::GpuMat srcGPU);

//	void getHistogram_GPU(cv::cuda::GpuMat *channel, cv::Mat *hist);

#endif

/*********************************************************************/
/* Project: uw-img-proc									             */
/* Module:  contrastenhancement				  		                 */
/* File: 	contrastenhancement.cpp						             */
/* Created:	01/03/2018				                                 */
/* Description:
	C++ Module for contrast enhancement using histogram stretching   
	and equalization												 */
 /********************************************************************/

 /********************************************************************/
 /* Created by:                                                      */
 /* Geraldine Barreto (@geraldinebc)                                 */
/*********************************************************************/


cv::Mat simplestColorBalance(cv::Mat src, float percent) {			// Simplest Color Balance
	vector<Mat_<uchar>> channel;
	split(src, channel);
	cv::Mat flat, result[3];
	for (int i = 0; i < 3; i++) {
		flat = channel[i].clone();													// Clone the matrix of each color channel
		flat = flat.reshape(0, 1);													// Reshape the matrix to one column
		cv::sort(flat, flat, SORT_EVERY_ROW + SORT_ASCENDING);						// Sort values from low to high
		int min = flat.at<uchar>(0, floor(flat.cols * percent / 100.0));			// Minimum boundary
		int max = flat.at<uchar>(0, ceil(flat.cols * (1.0 - percent / 100.0)));		// Maximum boundary
		result[i] = (channel[i] - min) * 255.0 / (max - min);						// Pixel remapping								// CHANGE 255 to max
	}
	cv::Mat balanced;
	merge(result, 3, balanced);
	return balanced;
}

void getHistogram(cv::Mat *channel, cv::Mat *hist) {								// Computes the histogram of a single channel
	int histSize = 256;
	float range[] = { 0, 256 };														// The histograms ranges from 0 to 255
	const float* histRange = { range };
	cv::calcHist(channel, 1, 0, Mat(), *hist, 1, &histSize, &histRange, true, false);
}

void getHistogram_GPU(cv::cuda::GpuMat *channel, cv::Mat *hist) {								// Computes the histogram of a single channel
/*	int histSize = 256;
	float range[] = { 0, 256 };														// The histograms ranges from 0 to 255
	const float* histRange = { range };
	cv::cuda::calcHist(channel, *hist);*/
}


cv::Mat histStretch(cv::Mat src, float percent, int direction) {
	cv::Mat histogram;
	getHistogram(&src, &histogram);
	float percent_sum = 0.0, channel_min = -1.0, channel_max = -1.0;
	float percent_min = percent / 100.0, percent_max = 1.0 - percent_min;
	int i = 0;
	
	while (percent_sum < percent_max * src.total()) {
		if (percent_sum < percent_min * src.total()) channel_min++;
		percent_sum += histogram.at<float>(i, 0);
		channel_max++;
		i++;
	}
	
	cv::Mat dst;
	if (direction == 0) dst = (src - channel_min) * (255.0 - channel_min) / (channel_max - channel_min) + channel_min;	// Stretches the channel towards the Upper side
	else if (direction == 2) dst = (src - channel_min) * channel_max / (channel_max - channel_min);						// Stretches the channel towards the Lower side
	else dst = (src - channel_min) * 255.0 / (channel_max - channel_min);												// Stretches the channel towards both sides
	dst.convertTo(dst, CV_8UC1);
	return dst;
}

cv::Mat ICM(cv::Mat src, float percent) {												// Integrated Color Model
	vector<Mat_<uchar>> channel;
	split(src, channel);
	Mat chan[3], result;
	for (int i = 0; i < 3; i++) chan[i] = histStretch(channel[i], percent, 1);			// Histogram stretching of each color channel
	merge(chan, 3, result);
	Mat HSV, hsv[3], dst;
	cv::cvtColor(result, HSV, COLOR_BGR2HSV);												// Conversion to the HSV color model
	split(HSV, hsv);
	for (int i = 1; i < 3; i++) hsv[i] = histStretch(hsv[i], percent, 1);				// Histogram stretching of the Saturation and Value Channels
	merge(hsv, 3, HSV);
	cv::cvtColor(HSV, dst, COLOR_HSV2BGR);													// Conversion to the BGR color model
	return dst;
}

// For the lowest channel the best is towards the lower (2) side or else it causes red blobs

cv::Mat UCM(cv::Mat src, float percent) {
	vector<Mat_<uchar>> channel;
	split(src, channel);
	float means[3] = { mean(channel[0])[0], mean(channel[1])[0], mean(channel[2])[0] };	// Means of each channel
	cv::Mat c, sorted;
	c = Mat(1, 3, CV_32F, means);
	sortIdx(c, sorted, SORT_EVERY_ROW + SORT_ASCENDING);								// Sorts means from low to high
	vector<Mat_<float>> sorted_chan;
	for(int i = 0; i < 3; i++) sorted_chan.push_back(channel[sorted.at<int>(0, i)]);	// Reorders channels according to their mean

	float A = means[sorted.at<int>(0, 2)] / means[sorted.at<int>(0, 0)];				// Gain factor for the equalization (Von Kries Hypothesis)
	float B = means[sorted.at<int>(0, 2)] / means[sorted.at<int>(0, 1)];

	sorted_chan[0] = A * sorted_chan[0];												// Channels equalization
	sorted_chan[1] = B * sorted_chan[1];
	for (int i = 0; i < 3; i++) cv::threshold(sorted_chan[i], sorted_chan[i], 255, 255, 2);	// Limit the max value to 255
	
	cv::Mat result_chan[3], result;														// Histogram stretching according to the channel intensity
	for (int i = 0; i < 3; i++) result_chan[sorted.at<int>(0, i)] = histStretch(sorted_chan[i], percent, i);
	merge(result_chan, 3, result);

	cv::Mat HSV, hsv[3], dst;
	cv::cvtColor(result, HSV, COLOR_BGR2HSV);												// Conversion to HSV color space
	split(HSV, hsv);
	for (int i = 1; i < 3; i++) hsv[i] = histStretch(hsv[i], percent, 1);				// Histogram stretching of channels S and V
	merge(hsv, 3, HSV);
	cv::cvtColor(HSV, dst, COLOR_HSV2BGR);
	return dst;
}

cv::Mat rayleighEqualization(cv::Mat src) {
	cv::Mat histogram;
	getHistogram(&src, &histogram);
	cv::Mat accumulated_hist = histogram.clone();										// Uses the image histogram to create the cumulative distribution function
	for (int i = 1; i < 256; i++) accumulated_hist.at<float>(i,0) += accumulated_hist.at<float>(i-1,0);
	cv::Mat cdf_norm = accumulated_hist / accumulated_hist.at<float>(255, 0);			// Normalizes the CDF (cumulative distribution function)

	cv::Mat dst(src.rows, src.cols, CV_8U);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {											// Maps the image histogram to follow Rayleigh's distribution
		if (cdf_norm.at<float>(src.at<uchar>(i, j), 0) >= 0.95) dst.at<uchar>(i, j) = 255.0 * cdf_norm.at<float>(src.at<uchar>(i, j), 0);
		else dst.at<uchar>(i,j) = 255.0 * sqrt(0.32 * log(1.0 / (1.0 - cdf_norm.at<float>(src.at<uchar>(i,j),0))));
		}
	}
	return dst;
}

#if USE_GPU
/*
cv::Mat simplestColorBalance(cv::Mat src, float percent) {			// Simplest Color Balance
	vector<Mat_<uchar>> channel;
	split(src, channel);
	cv::Mat flat, result[3];
	for (int i = 0; i < 3; i++) {
		flat = channel[i].clone();													// Clone the matrix of each color channel
		flat = flat.reshape(0, 1);													// Reshape the matrix to one column
		cv::sort(flat, flat, SORT_EVERY_ROW + SORT_ASCENDING);						// Sort values from low to high
		int min = flat.at<uchar>(0, floor(flat.cols * percent / 100.0));			// Minimum boundary
		int max = flat.at<uchar>(0, ceil(flat.cols * (1.0 - percent / 100.0)));		// Maximum boundary
		result[i] = (channel[i] - min) * 255.0 / (max - min);						// Pixel remapping								// CHANGE 255 to max
	}
	cv::Mat balanced;
	merge(result, 3, balanced);
	return balanced;
}*/

cv::cuda::GpuMat simplestColorBalance_GPU(cv::cuda::GpuMat srcGPU, float percent) {	// Simplest Color Balance
	std::vector<cv::cuda::GpuMat> channel;
	split(srcGPU, channel);
	cv::cuda::GpuMat flat, result[3];

	for (int i = 0; i < 3; i++) {
//		flat = srcGPU.clone();														// Clone the matrix of each color channel
		cv::Mat _tmp_flat(channel[i].clone());
//		flat = channel[i].clone();													// Clone the matrix of each color channel
		_tmp_flat = _tmp_flat.reshape(0, 1);													// Reshape the matrix to one column
		cv::sort(_tmp_flat, _tmp_flat, SORT_EVERY_ROW + SORT_ASCENDING);						// Sort values from low to high
		float min = _tmp_flat.at<uchar>(0, floor(_tmp_flat.cols * percent / 100.0));			// Minimum boundary
		float max = _tmp_flat.at<uchar>(0, ceil(_tmp_flat.cols * (1.0 - percent / 100.0)));	// Maximum boundary
		cv::cuda::subtract(result[i], min, result[i]);								// Pixel remapping
		cv::cuda::multiply(result[i], 255.0 / (max - min), result[i]);
	}
	cv::cuda::GpuMat dst;
	cv::cuda::merge(result, 3, dst);
	return dst;
}

cv::cuda::GpuMat histStretch_GPU(cv::cuda::GpuMat srcGPU, float percent, int direction) {
	cv::cuda::GpuMat hist;
	cv::Mat histogram;
	cv::cuda::calcHist(srcGPU, hist);
	hist.download(histogram);
	float percent_sum = 0.0, channel_min = -1.0, channel_max = -1.0;
	float percent_min = percent / 100.0, percent_max = 1.0 - percent_min;
	int i = 0;

	// In order to retrieve the total number of elements in a GPU stored matrix, we cannot use the total() method
	// available for cv::Mat
	// GpuMat doesn't provide per element access (standard iterators will fail, such as <at>)

	cv::Mat _temp(srcGPU);

	while (percent_sum < percent_max * _temp.total()) {
		if (percent_sum < percent_min * _temp.total()) channel_min++;
		percent_sum += histogram.at<float>(i, 0);
		channel_max++;
		i++;
	}

	cv::cuda::GpuMat dst;
	if (direction = 0) {
		cv::cuda::subtract(srcGPU, channel_min, dst);										// Stretches the channel towards the Upper side
		cv::cuda::multiply(dst, (255.0 - channel_min) / (channel_max - channel_min), dst);
		cv::cuda::add(dst, channel_min, dst);
	}
	else if (direction = 2) {
		cv::cuda::subtract(srcGPU, channel_min, dst);										// Stretches the channel towards the Lower side
		cv::cuda::multiply(dst, channel_max / (channel_max - channel_min), dst);
	}
	else {							
		cv::cuda::subtract(srcGPU, channel_min, dst);										// Stretches the channel towards both sides
		cv::cuda::multiply(dst, 255.0 / (channel_max - channel_min), dst);
	}
	dst.convertTo(dst, CV_8U);
	return dst;
}

cv::cuda::GpuMat ICM_GPU(cv::cuda::GpuMat srcGPU, float percent) {						// Integrated Color Model
	cv::cuda::GpuMat channel[3];
	cv::cuda::split(srcGPU, channel);
	cv::cuda::GpuMat chan[3], result;
	for (int i = 0; i < 3; i++) chan[i] = histStretch_GPU(channel[i], percent, 1);		// Histogram stretching of each color channel
	cv::cuda::merge(chan, 3, result);
	cv::cuda::GpuMat HSV, hsv[3], dst;
	cv::cuda::cvtColor(result, HSV, COLOR_BGR2HSV);										// Conversion to the HSV color model
	cv::cuda::split(HSV, hsv);
	for (int i = 1; i < 3; i++) hsv[i] = histStretch_GPU(hsv[i], percent, 1);			// Histogram stretching of the Saturation and Value Channels
	cv::cuda::merge(hsv, 3, HSV);
	cv::cuda::cvtColor(HSV, dst, COLOR_HSV2BGR);										// Conversion to the BGR color model
	return dst;
}

cv::cuda::GpuMat UCM_GPU(cv::cuda::GpuMat srcGPU, float percent) {
	cv::cuda::GpuMat channel[3];
	cv::cuda::split(srcGPU, channel);
	float means[3] = { mean(channel[0])[0], mean(channel[1])[0], mean(channel[2])[0] };	// Means of each channel
	//else cv::cuda::meanStdDev 
	cv::Mat c, sorted;
	c = Mat(1, 3, CV_32F, means);
	sortIdx(c, sorted, SORT_EVERY_ROW + SORT_ASCENDING);								// Sorts means from low to high
	vector<GpuMat> sorted_chan;
	for (int i = 0; i < 3; i++) sorted_chan.push_back(channel[sorted.at<int>(0, i)]);	// Reorders channels according to their mean

	float A = means[sorted.at<int>(0, 2)] / means[sorted.at<int>(0, 0)];				// Gain factor for the equalization (Von Kries Hypothesis)
	float B = means[sorted.at<int>(0, 2)] / means[sorted.at<int>(0, 1)];

	cv::cuda::multiply(sorted_chan[0], A, sorted_chan[0]);								// Channels equalization
	cv::cuda::multiply(sorted_chan[1], B, sorted_chan[1]);
	for (int i = 0; i < 3; i++) cv::cuda::threshold(sorted_chan[i], sorted_chan[i], 255, 255, 2);	// Limit the max value to 255

	cv::cuda::GpuMat result_chan[3], result;											// Histogram stretching according to the channel intensity
	for (int i = 0; i < 3; i++) result_chan[sorted.at<int>(0, i)] = histStretch_GPU(sorted_chan[i], percent, i);
	cv::cuda::merge(result_chan, 3, result);

	cv::cuda::GpuMat HSV, hsv[3], dst;
	cv::cuda::cvtColor(result, HSV, COLOR_BGR2HSV);										// Conversion to HSV color space
	cv::cuda::split(HSV, hsv);
	for (int i = 1; i < 3; i++) hsv[i] = histStretch_GPU(hsv[i], percent, 1);			// Histogram stretching of channels S and V
	cv::cuda::merge(hsv, 3, HSV);
	cv::cuda::cvtColor(HSV, dst, COLOR_HSV2BGR);
	return dst;
}

cv::cuda::GpuMat rayleighEqualization_GPU(cv::cuda::GpuMat srcGPU) {
	cv::Mat histogram, src;

	cv::cuda::calcHist(srcGPU, histogram);
	// TODO: THERE IS A DIRECT METHOD TO COMPUTE HISTOGRAM IN GPU (but no range can be specified)
	//getHistogram_GPU(&srcGPU, &histogram);
	

	cv::Mat accumulated_hist = histogram.clone();										// Uses the image histogram to create the cumulative distribution function
	for (int i = 1; i < 256; i++) accumulated_hist.at<float>(i, 0) += accumulated_hist.at<float>(i - 1, 0);
	cv::Mat cdf_norm = accumulated_hist / accumulated_hist.at<float>(255, 0);			// Normalizes the CDF (cumulative distribution function)
	
	srcGPU.download(src);
	cv::Mat dst(src.rows, src.cols, CV_8U);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {											// Maps the image histogram to follow Rayleigh's distribution
			if (cdf_norm.at<float>(src.at<uchar>(i, j), 0) >= 0.95) dst.at<uchar>(i, j) = 255.0 * cdf_norm.at<float>(src.at<uchar>(i, j), 0);
			else dst.at<uchar>(i, j) = 255.0 * sqrt(0.32 * log(1.0 / (1.0 - cdf_norm.at<float>(src.at<uchar>(i, j), 0))));
		}
	}
	cv::cuda::GpuMat dstGPU;
	dstGPU.upload(dst);
	return dstGPU;
}
#endif

/*********************************************************************/
/* Project: uw-img-proc									             */
/* Module:  contrastenhancement				  		                 */
/* File: 	main.cpp								         	     */
/* Created:	01/03/2018				                                 */
/* Description:
	C++ Module for contrast enhancement using histogram stretching   
	and equalization												 */
 /********************************************************************/

 /********************************************************************/
 /* Created by:                                                      */
 /* Geraldine Barreto (@geraldinebc)                                 */
/*********************************************************************/

#define ABOUT_STRING "Contrast Enhancement Module based on Histogram Stretching and Equalization"

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
		"{@input  |<none> | Input image file}"									// Input image is the first argument (positional)
		"{@output |<none> | Output image file}"									// Output prefix is the second argument (positional)
		"{m       |r      | Method}"											// Enhancement method to use
		"{show    |       | Show result (ON: 1, OFF: 0)}"						// Show the resulting image (optional)
		"{cuda    |       | Use CUDA or not (CUDA ON: 1, CUDA OFF: 0)}"         // Use CUDA (optional)
		"{time    |       | Show time measurements or not (ON: 1, OFF: 0)}"		// Show time measurements (optional)
		"{help h usage ?  |       | Print help message}";						// Show help (optional)

	CommandLineParser cvParser(argc, argv, keys);
	cvParser.about(ABOUT_STRING);												// Adds "about" information to the parser method

	//**************************************************************************
	std::cout << ABOUT_STRING << endl;
	std::cout << "Built with OpenCV " << CV_VERSION << endl;

	// If the number of arguments is lower than 3, or contains "help" keyword, then we show the help
	if (argc < 5 || cvParser.has("help")) {
		std::cout << endl << "C++ implementation of contrast enhancement module using histogram stretching" << endl;
		std::cout << endl << "Arguments are:" << endl;
		std::cout << "\t*Input: Input image name with path and extension" << endl;
		std::cout << "\t*Output: Output image name with path and extension" << endl;
		std::cout << "\t*-show=0 or -show=1 (ON: 1, OFF: 0)" << endl;
		std::cout << "\t*-cuda=0 or -cuda=1 (ON: 1, OFF: 0)" << endl;
		std::cout << "\t*-time=0 or -time=1 (ON: 1, OFF: 0)" << endl;
		std::cout << "\t*Argument 'm=<method>' is a string containing a list of the desired method to use" << endl;
		std::cout << endl << "Complete options of evaluation metrics are:" << endl;
		std::cout << "\t-m=S for Simplest Color Balance" << endl;
		std::cout << "\t-m=I for Integrated Color Model " << endl;
		std::cout << "\t-m=U for Unsupervised Color Correction Method" << endl;
		std::cout << "\t-m=E for Histogram Equalization" << endl;
		std::cout << "\t-m=R for Rayleigh Equalization" << endl;
		std::cout << endl << "Example:" << endl;
		std::cout << "\tinput.jpg output.jpg -cuda=0 -time=0 -show=0 -m=I" << endl;
		std::cout << "\tThis will open 'input.jpg' enhance the contrast using the ICM Method and save it in 'output.jpg'" << endl << endl;
		return 0;
	}

	int CUDA = 0;										// Default option (running with CPU)
	int Time = 0;                                       // Default option (not showing time)
	int Show = 0;										// Default option (not showing results)

	std::string InputFile = cvParser.get<cv::String>(0); // String containing the input file path+name+extension from cvParser function
	std::string OutputFile = cvParser.get<cv::String>(1);// String containing the input file path+name+extension from cvParser function
	std::string method = cvParser.get<cv::String>("m");	 // Gets argument -m=x, where 'x' is the enhancement method
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

	std::cout << endl << "************************************************************************" << endl;
	std::cout << endl << "Input: " << InputFile << endl;
	std::cout << "Output: " << OutputFile << endl;

	cv::Mat src, dst;
	src = imread(InputFile, IMREAD_COLOR);

	if (src.empty()){
		std::cout << "Error occured when loading the image" << endl << endl;
		return -1;
	}

	// Start time measurement
	if (Time) t = (double)getTickCount();

	// GPU Implementation
#if USE_GPU
	if (CUDA) {
		cv::cuda::GpuMat srcGPU, channelsGPU[3], dstGPU;
		srcGPU.upload(src);

		switch (method[0]) {

			case 'S':	// Simplest Color Balance
				std::cout << endl << "Applying contrast enhancement using Simplest Color Balance" << endl;
				dstGPU = simplestColorBalance_GPU(srcGPU, 0.5);
				break;

			case 'I':	// Integrated Color Model
				std::cout << endl << "Applying contrast enhancement using an Integrated Color Model" << endl;
				dstGPU = ICM_GPU(srcGPU, 0.5);
				break;

			case 'U':	// Unsupervised Color Correction Method
				std::cout << endl << "Applying contrast enhancement using an Unsupervised Color Correction Method" << endl;
				dstGPU = UCM_GPU(srcGPU, 0.2);
				break;

			case 'E':	// Normal Equalization
				std::cout << endl << "Applying contrast enhancement using Normal Equalization" << endl;
				cv::cuda::split(srcGPU, channelsGPU);
				for (int i = 0; i < 3; i++) cv::cuda::equalizeHist(channelsGPU[i], channelsGPU[i]);
				cv::cuda::merge(channelsGPU, 3, dstGPU);
				break;

			case 'R':	// Rayleigh Equalization
				std::cout << endl << "Applying contrast enhancement using Rayleigh Equalization" << endl;
				cv::cuda::split(srcGPU, channelsGPU);
				for (int i = 0; i < 3; i++) channelsGPU[i] = rayleighEqualization_GPU(channelsGPU[i]);
				cv::cuda::merge(channelsGPU, 3, dstGPU);
				break;

			default:	// Unrecognized Option
				std::cout << "Option " << method[0] << " not recognized" << endl;
			break;
		}
		dstGPU.download(dst);
	}
#endif

	// CPU Implementation
	if (! CUDA) {

		cv::Mat balanced;
		vector<Mat_<uchar>> channels;

		switch (method[0]) {

			case 'S':	// Simplest Color Balance
				std::cout << endl << "Applying contrast enhancement using Simplest Color Balance" << endl;
				dst = simplestColorBalance(src, 0.5);
			break;

			case 'I':	// Integrated Color Model
				std::cout << endl << "Applying contrast enhancement using an Integrated Color Model" << endl;
				dst = ICM(src, 0.5);
			break;

			case 'U':	// Unsupervised Color Correction Method
				std::cout << endl << "Applying contrast enhancement using an Unsupervised Color Correction Method" << endl;
				dst = UCM(src, 0.2);
			break;

			case 'E':	// Normal Equalization
				std::cout << endl << "Applying contrast enhancement using Normal Equalization" << endl;
				split(src, channels);
				for (int i = 0; i < 3; i++) cv::equalizeHist(channels[i], channels[i]);
				merge(channels, dst);
			break;

			case 'R':	// Rayleigh Equalization
				std::cout << endl << "Applying contrast enhancement using Rayleigh Equalization" << endl;
				split(src, channels);
				for (int i = 0; i < 3; i++) channels[i] = rayleighEqualization(channels[i]);
				merge(channels, dst);
			break;

			default:	// Unrecognized Option
				std::cout << "Option " << method[0] << " not recognized" << endl;
			break;
		}
	}

	//  End time measurement (Showing time results is optional)
	if (Time) {
		t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
		std::cout << endl << "Execution Time" << implementation << ": " << t << " ms " << endl;

		// Name for the output csv file where the time will be saved
		std::size_t pos;
		if (OutputFile.find(92) != std::string::npos) pos = OutputFile.find_last_of(92);	// If string contains '\' search for the last one
		else pos = OutputFile.find_last_of('/');											// If does not contain '\' search for the last '/'
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
		Mat comparison;
		hconcat(src, dst, comparison);
		namedWindow("Comparison", WINDOW_KEEPRATIO);
		imshow("Comparison", comparison);
	}

	waitKey(0);
	return 0;
}