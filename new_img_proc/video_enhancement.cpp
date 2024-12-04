/*********************************************************************/
/* Project: uw-img-proc									             */
/* Module:  videoenhancement				  		                 */
/* File: 	videoenhancement.h					         	     */
/* Created:	20/09/2019				                                 */
/* Description:
	C++ Module for underwater video enhancement						 */
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
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

// C++ namespaces
using namespace cv;
using namespace cuda;
using namespace std;
using namespace ximgproc;

/// CUDA specific libraries
#if USE_GPU
#include <opencv2/cudafilters.hpp>
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#endif

/*
	@brief		Stretches the histogram of one image channel in a specific direction (right 0, both sides 1 or left 2)
	@function	cv::Mat histStretch(cv::Mat prev, cv::Mat src, float percent, int direction);
*/
cv::Mat histStretch(cv::Mat prev, cv::Mat src, float percent, int direction);

/*
	@brief		Enhances the contrast of an image using the Integrated Color Model by Iqbal et al. based on histogram stretching
	@function	cv::Mat ICM(cv::Mat prev, cv::Mat src, float percent)
*/
cv::Mat ICM(cv::Mat prev, cv::Mat src, float percent);

/*
	@brief		Corrects the color of an image using Grey World Assumption and histogram strething
	@function	cv::Mat UCM(cv::Mat src, float percent);
*/
cv::Mat colorcorrection(cv::Mat src);

/*
	@brief		Dehazes an underwater image using the Bright channel Prior
	@function	cv::Mat dehazing(cv::Mat prev, cv::Mat src)
*/
cv::Mat dehazing(cv::Mat prev, cv::Mat src);

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
cv::Mat rectify(cv::Mat S, cv::Mat bc, cv::Mat mcd);

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

/*********************************************************************/
/* Project: uw-img-proc									             */
/* Module:  videoenhancement				  		                 */
/* File: 	videoenhancement.cpp						             */
/* Created:	20/09/2019				                                 */
/* Description:
	C++ Module for underwater video enhancement						 */
/********************************************************************/

 /********************************************************************/
 /* Created by:                                                      */
 /* Geraldine Barreto (@geraldinebc)                                 */
/*********************************************************************/


void getHistogram(cv::Mat *channel, cv::Mat *hist) {								// Computes the histogram of a single channel
	int histSize = 256;
	float range[] = { 0, 256 };														// The histograms ranges from 0 to 255
	const float* histRange = { range };
	calcHist(channel, 1, 0, Mat(), *hist, 1, &histSize, &histRange, true, false);
}

cv::Mat histStretch(cv::Mat prev, cv::Mat src, float percent, int direction) {
	cv::Mat sum;
	addWeighted(prev, 0.7, src, 0.3, 0, sum);

	cv::Mat histogram;
	getHistogram(&sum, &histogram);
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

cv::Mat ICM(cv::Mat prev, cv::Mat src, float percent) {												// Integrated Color Model
	vector<Mat_<uchar>> chann, channel;
	split(prev, chann);
	split(src, channel);
	Mat chan[3], result;
	for (int i = 0; i < 3; i++) chan[i] = histStretch(chann[i], channel[i], percent, 1);			// Histogram stretching of each color channel
	merge(chan, 3, result);
	Mat HSV, hsv[3], dst;
	cvtColor(result, HSV, COLOR_BGR2HSV);												// Conversion to the HSV color model
	split(HSV, hsv);
	for (int i = 1; i < 3; i++) hsv[i] = histStretch(hsv[i], hsv[i], percent, 1);		// Histogram stretching of the Saturation and Value Channels
	merge(hsv, 3, HSV);
	cvtColor(HSV, dst, COLOR_HSV2BGR);													// Conversion to the BGR color model
	return dst;
}

cv::Mat colorcorrection(cv::Mat src) {														// Corrects the color
	cv::Mat LAB, lab[3], dst;
	cvtColor(src, LAB, COLOR_BGR2Lab);														// Conversion to the CIELAB color space
	split(LAB, lab);
	lab[0] = histStretch(lab[0], lab[0], 1, 1);											// Histogram stretching
	lab[1] = 127.5 * lab[1] / mean(lab[1])[0];												// Grey World Assumption
	lab[2] = 127.5 * lab[2] / mean(lab[2])[0];
	merge(lab, 3, LAB);
	cvtColor(LAB, dst, COLOR_Lab2BGR);														// Conversion to the BGR color space
	return dst;
}
cv::Mat dehazing(cv::Mat prev, cv::Mat src) {											// Dehazed an underwater image
	cv::Mat sum;
	addWeighted(prev, 0.7, src, 0.3, 0, sum);

	vector<Mat_<uchar>> sum_chan, new_chan;
	split(sum, sum_chan);

	new_chan.push_back(255 - sum_chan[0]);												// Compute the new channels for the dehazing process
	new_chan.push_back(255 - sum_chan[1]);
	new_chan.push_back(sum_chan[2]);

	int size = sqrt(src.total()) / 40;													// Making the size bigger creates halos around objects
	cv::Mat bright_chan = brightChannel(new_chan, size);								// Compute the bright channel image
	cv::Mat mcd = maxColDiff(sum_chan);													// Compute the maximum color difference

	cv::Mat sum_HSV, S;
	cv::cvtColor(sum, sum_HSV, COLOR_BGR2HSV);
	extractChannel(sum_HSV, S, 1);
	cv::Mat rectified = rectify(S, bright_chan, mcd);									// Rectify the bright channel image

	cv::Mat sum_gray;
	cv::cvtColor(sum, sum_gray, COLOR_BGR2GRAY);
	vector<uchar> A;
	A = lightEstimation(sum_gray, size, bright_chan, new_chan);							// Estimate the atmospheric light
	cv::Mat trans = transmittance(rectified, A);										// Compute the transmittance image

	cv::Mat filtered;
	guidedFilter(sum_gray, trans, filtered, 30, 0.001, -1);								// Refine the transmittance image

	vector<Mat_<uchar>> src_chan;
	vector<Mat_<float>> chan_dehazed;
	split(src, src_chan);
	chan_dehazed.push_back(255 - src_chan[0]);											// Compute the new channels for the dehazing process
	chan_dehazed.push_back(255 - src_chan[1]);
	chan_dehazed.push_back(src_chan[2]);
	cv::Mat dst = dehaze(chan_dehazed, A, filtered);									// Dehaze the image channels
	return dst;
}

cv::Mat brightChannel(std::vector<cv::Mat_<uchar>> channels, int size) {				// Generates the Bright Channel Image
	cv::Mat maxRGB = max(max(channels[0], channels[1]), channels[2]);					// Maximum Color Image
	cv::Mat element, bright_chan;
	element = getStructuringElement(MORPH_RECT, Size(size, size), Point(-1, -1));		// Maximum filter
	dilate(maxRGB, bright_chan, element);												// Dilates the maxRGB image
	return bright_chan;
}

cv::Mat maxColDiff(std::vector<cv::Mat_<uchar>> channels) {								// Generates the Maximum Color Difference Image
	vector<float> means;
	means.push_back(mean(channels[0])[0]);
	means.push_back(mean(channels[1])[0]);
	means.push_back(mean(channels[2])[0]);
	cv::Mat sorted;
	sortIdx(means, sorted, SORT_EVERY_ROW + SORT_ASCENDING);							// Orders the mean of the channels from low to high

	cv::Mat cmin = channels[sorted.at<int>(0, 0)];
	cv::Mat cmid = channels[sorted.at<int>(0, 1)];
	cv::Mat cmax = channels[sorted.at<int>(0, 2)];

	cv::Mat a, b, mcd;
	a = max(cmax - cmin, 0);															// Calculates the maximum values for the MCD image
	b = max(cmid - cmin, 0);
	mcd = 255 - max(a, b);
	return mcd;
}

cv::Mat rectify(cv::Mat S, cv::Mat bc, cv::Mat mcd) {									// Rectifies the Bright Channel Image
	double lambda;
	minMaxLoc(S, NULL, &lambda);														// Maximum value of the Saturation channel
	lambda = lambda / 255.0;															// Normalization for the next step
	cv::Mat correct;
	addWeighted(bc, lambda, mcd, 1.0 - lambda, 0.0, correct);
	return correct;
}

std::vector<uchar> lightEstimation(cv::Mat src_gray, int size, cv::Mat bright_chan, std::vector<Mat_<uchar>> channels) {
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
	return A;
}

cv::Mat transmittance(cv::Mat correct, vector<uchar> A) {						// Computes the Transmittance Image
	correct.convertTo(correct, CV_32F);
	cv::Mat t[3], acc(correct.size(), CV_32F, Scalar(0));
	for (int i = 0; i < 3; i++) {
		t[i] = 255.0 * ((correct - A[i]) / (255.0 - A[i]));
		accumulate(t[i], acc);
	}
	cv::Mat trans = acc / 3;
	trans.convertTo(trans, CV_8U);
	return trans;
}

cv::Mat dehaze(vector<Mat_<float>> channels, vector<uchar> A, cv::Mat trans) {	// Restores the Underwater Image using the Bright Channel Prior
	trans.convertTo(trans, CV_32F, 1.0 / 255.0);
	channels[0] = 255.0 - ((channels[0] - (A[0] * (1.0 - trans))) / trans);
	channels[1] = 255.0 - ((channels[1] - (A[1] * (1.0 - trans))) / trans);
	channels[2] = (channels[2] - A[2]) / trans + A[2];
	cv::Mat dehazed, dst;
	merge(channels, dehazed);
	dehazed.convertTo(dst, CV_8U);
	return dst;
}


/*********************************************************************/
/* Project: uw-img-proc									             */
/* Module:  videoenhancement				  		                 */
/* File: 	main.cpp								         	     */
/* Created:	20/09/2019				                                 */
/* Description:
	C++ Module for underwater video enhancement						 */
 /********************************************************************/

 /********************************************************************/
 /* Created by:                                                      */
 /* Geraldine Barreto (@geraldinebc)                                 */
/*********************************************************************/

#define ABOUT_STRING "Video Enhancement Module"


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
		"{@input  |<none> | Input image file}"									// Input video is the first argument (positional)
		"{m       |r      | Method}"											// Enhancement method to use
		"{comp    |       | Save video comparison (ON: 1, OFF: 0)}"				// Show video comparison (optional)
		"{cuda    |       | Use CUDA or not (CUDA ON: 1, CUDA OFF: 0)}"         // Use CUDA (optional)
		"{time    |       | Show time measurements or not (ON: 1, OFF: 0)}"		// Show time measurements (optional)
		"{help h usage ?  |       | Print help message}";						// Show help (optional)

	CommandLineParser cvParser(argc, argv, keys);
	cvParser.about(ABOUT_STRING);												// Adds "about" information to the parser method

	//**************************************************************************
	std::cout << ABOUT_STRING << endl;
	std::cout << "\nBuilt with OpenCV" << CV_VERSION << endl;

	// If the number of arguments is lower than 3, or contains "help" keyword, then we show the help
	if (argc < 5 || cvParser.has("help")) {
		std::cout << endl << "C++ implementation of video enhancement module" << endl;
		std::cout << endl << "Arguments are:" << endl;
		std::cout << "\t*Input: Input video name with path and extension" << endl;
		std::cout << "\t*-comp=0 or -comp=1 (ON: 1, OFF: 0)" << endl;
		std::cout << "\t*-cuda=0 or -cuda=1 (ON: 1, OFF: 0)" << endl;
		std::cout << "\t*-time=0 or -time=1 (ON: 1, OFF: 0)" << endl;
		std::cout << "\t*Argument 'm=<method>' is a string containing a list of the desired method to use" << endl;
		std::cout << endl << "Complete options are:" << endl;
		std::cout << "\t-m=C for Color Correction" << endl;
		std::cout << "\t-m=E for Histogram Equalization" << endl;
		std::cout << "\t-m=S for Histogram Stretching" << endl;
		std::cout << "\t-m=D for Dehazing" << endl;
		std::cout << endl << "Example:" << endl;
		std::cout << "\tv1.mp4 -cuda=0 -time=0 -comp=0 -m=E" << endl;
		std::cout << "\tThis will open 'v1.mp4' enhance the video using histogram equalization and save it in 'v1_E.avi'" << endl << endl;
		return 0;
	}

	int CUDA = 0;										// Default option (running with CPU)
	int Time = 0;                                       // Default option (not showing time)
	int Comp = 0;										// Default option (not showing results)

	std::string InputFile = cvParser.get<cv::String>(0); // String containing the input file path+name+extension from cvParser function
	std::string method = cvParser.get<cv::String>("m");	 // Gets argument -m=x, where 'x' is the enhancement method
	std::string implementation;							 // CPU or GPU implementation
	Comp = cvParser.get<int>("comp");					 // Gets argument -comp=x, where 'x' defines if the comparison video will be saved or not
	Time = cvParser.get<int>("time");	                 // Gets argument -time=x, where 'x' defines if execution time will show or not

	// Check if any error occurred during parsing process
	if (!cvParser.check()) {
		cvParser.printErrors();
		return -1;
	}

	int nCuda = -1;    //Defines number of detected CUDA devices. By default, -1 acting as error value

	std::size_t filename;
	if (InputFile.find('.') != std::string::npos) filename = InputFile.find_last_of('.');	// Find the last '.'
	std::string ext_out = '_' + method + ".avi";
	std::string OutputFile = InputFile.substr(0, filename);
	OutputFile.insert(filename, ext_out);

	std::cout << endl << "************************************************************************" << endl;
	std::cout << endl << "Input: " << InputFile << endl;
	std::cout << "Output: " << OutputFile << endl;

	// Opend the video file
	cv::VideoCapture cap(InputFile);
	if (!cap.isOpened()) {
		std::cout << "\nUnable to open the video \n";
		return -1;
	}

	// Get the width/height frame count and the FPS of the video
	int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
	int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
	int n_frames = int(cap.get(CAP_PROP_FRAME_COUNT));
	double FPS = cap.get(CAP_PROP_FPS);

	// Open a video file for writing the output
	cv::VideoWriter out(OutputFile,cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), FPS, cv::Size(width, height));
	if (!out.isOpened()) {
		std::cout << "\nError! Unable to open video file for the output video \n\n" << std::endl;
		return -1;
	}

	std::string ext_comp = '_' + method + "_comp.avi";
	std::string Comparison = InputFile.substr(0, filename);
	Comparison.insert(filename, ext_comp);

	// Open a video file for writing the comparison
	cv::VideoWriter comp(Comparison, cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), FPS, cv::Size(2 * width, height));
	if (!comp.isOpened()) {
		std::cout << "\nError! Unable to open video file for the comparison video \n\n" << std::endl;
		return -1;
	}

	// Start time measurement
	if (Time) t = (double)getTickCount();

	// CPU Implementation
	if (! CUDA) {

		cv::Mat image, image_out, comparison, top;
		vector<cv::Mat> frames;
		cv::Mat sum(cv::Size(width, height), CV_32FC3, Scalar());
		cv::Mat avgImg(cv::Size(width, height), CV_32FC3, Scalar());
		vector<Mat_<uchar>> channels;

		int i = 0, j = 0;
		float n;
		if (n_frames / FPS < 7) n = n_frames / FPS * 0.5;
		else n = FPS * 7;

		switch (method[0]) {

			case 'C':	// Color Correction
				std::cout << endl << "Applying video enhancement using the Gray World Assumption" << endl;
				for (i = 0; i < n_frames - 1; i++) {
					cap >> image;
					image_out = colorcorrection(image);
					out << image_out;
					if (Comp) {
						hconcat(image, image_out, comparison);
						comp << comparison;
					}
				}
				std::cout << "\nProcessed video saved\n";
				break;

			case 'E':	// Histogram Equalization
				std::cout << endl << "Applying video enhancement using Histogram Equalization" << endl;
				for (i = 0; i < n_frames - 1; i++) {
					cap >> image;
					split(image, channels);
					for (j = 0; j < 3; j++) equalizeHist(channels[j], channels[j]);
					merge(channels, image_out);
					out << image_out;
					if (Comp) {
						hconcat(image, image_out, comparison);
						comp << comparison;
					}
				}
				std::cout << "\nProcessed video saved\n";
			break;

			case 'H':	// Histogram Stretching
				std::cout << endl << "Applying video enhancement using Histogram Stretching" << endl;
				while (true) {
					if (frames.size() < n) {
						cap >> image;
						if (image.empty()) {
							for (int i = 0; i < (n - 1) / 2; i++) {
								image_out = ICM(avgImg, frames[(n - 1) / 2 + i], 0.3);
								out << image_out;
								if (Comp) {
									hconcat(frames[(n - 1) / 2 + i], image_out, comparison);
									comp << comparison;
								}
							}
							std::cout << "\nProcessed video saved\n";
							break;
						}
						frames.push_back(image);
						image.convertTo(image, CV_32FC3);
						accumulate(image, sum);
					}
					else {
						sum.convertTo(avgImg, CV_8UC3, 1.0 / n);
						for (j; j < (n - 1) / 2; j++) {
							image_out = ICM(avgImg, frames[j], 0.5);
							out << image_out;
							if (Comp) {
								hconcat(frames[j], image_out, comparison);
								comp << comparison;
							}
						}
						image_out = ICM(avgImg, frames[(n - 1) / 2], 0.5);
						out << image_out;
						if (Comp) {
							hconcat(frames[(n - 1) / 2], image_out, comparison);
							comp << comparison;
						}
						frames.erase(frames.begin());
						frames[0].convertTo(top, CV_32FC3);
						subtract(sum, top, sum);
					}
				}
			break;

			case 'D':	// Dehazing
				std::cout << endl << "Applying video enhancement using the Bright Channel Prior" << endl;
				while (true) {
					if (frames.size() < n) {
						cap >> image;
						if (image.empty()) {
							for (int i = 0; i < (n - 1) / 2; i++) {
								image_out = dehazing(avgImg, frames[(n - 1) / 2 + i]);
								out << image_out;
								if (Comp) {
									hconcat(frames[(n - 1) / 2 + i], image_out, comparison);
									comp << comparison;
								}
							}
							std::cout << "\nProcessed video saved\n";
							break;
						}
						frames.push_back(image);
						image.convertTo(image, CV_32FC3);
						accumulate(image, sum);
					}
					else {
						sum.convertTo(avgImg, CV_8UC3, 1.0 / n);
						for (j; j < (n - 1) / 2; j++) {
							image_out = dehazing(avgImg, frames[j]);
							out << image_out;
							if (Comp) {
								hconcat(frames[j], image_out, comparison);
								comp << comparison;
							}
						}
						image_out = dehazing(avgImg, frames[(n - 1) / 2]);
						out << image_out;
						if (Comp) {
							hconcat(frames[(n - 1) / 2], image_out, comparison);
							comp << comparison;
						}
						frames.erase(frames.begin());
						frames[0].convertTo(top, CV_32FC3);
						subtract(sum, top, sum);
					}
				}
			break;

			default:	// Unrecognized Option
				std::cout << "Option " << method[0] << " not recognized" << endl;
			break;
		}

		// When everything done, release the video capture object
		cap.release();
	}

	// End time measurement (Showing time results is optional)
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
		file << endl << OutputFile << ";" << width << ";" << height << ";" << t;
	}

	waitKey(0);
	return 0;
}

