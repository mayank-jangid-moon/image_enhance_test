/********************************************************************/
/* Project: uw-img-proc									            */
/* Module: 	fusion										            */
/* File: 	fusion.h										        */
/* Created:	18/02/2019				                                */
/* Description:
	C++ module for underwater image enhancement using a fusion based
	strategy
 /*******************************************************************/

 /*******************************************************************/
 /* Created by:                                                     */
 /* Geraldine Barreto (@geraldinebc)                                */
 /*******************************************************************/

#define ABOUT_STRING "Fusion Enhancement Module"

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

/// CUDA specific libraries
#if USE_GPU
#include <opencv2/cudafilters.hpp>
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#endif

/*
	@brief		Corrects ununinform illumination using a homomorphic filter
	@function	illuminationCorrection(cv::Mat src)
*/
cv::Mat illuminationCorrection(cv::Mat src);

/*
	@brief		Computes the Normalized Discrete Fourier Transform
	@function	void fft(const cv::Mat &src, cv::Mat &dst)
*/
void fft(const cv::Mat &src, cv::Mat &dst);

/*
	@brief		Creates an Emphasis Highpass Gaussian Filter
	@function	cv::Mat gaussianFilter(cv::Mat img, float sigma, float high, float low)
*/
cv::Mat gaussianFilter(cv::Mat img, float sigma, float high, float low);

/*
	@brief		Rearranges the quadrants of a zero centered filter
	@function	void dftShift(Mat &fImage)
*/
void dftShift(Mat &fImage);

void getHistogram(cv::Mat *channel, cv::Mat *hist);

cv::Mat histStretch(cv::Mat src, float percent, int direction);

/*
	@brief		Corrects the hue shift and ununiform illumination of the underwater image
	@function	cv::Mat hueIllumination(cv::Mat src)
*/
cv::Mat hueIllumination(cv::Mat src);

/*
	@brief		Enhances the contrast of the image using histogram stretching
	@function	cv::Mat IUCM(vector<Mat_<uchar>> channel, float percent)
*/
cv::Mat ICM(vector<Mat_<uchar>> channel, float percent);

/*
	@brief		Dehazes an underwater image using the Bright channel Prior
	@function	cv::Mat dehazing(cv::Mat src)
*/
cv::Mat dehazing(cv::Mat src);

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

/*
	@brief		Creates a kernel for a 5x5 Gaussian Filter
	@function	cv::Mat filter_mask()
*/
cv::Mat filter_mask();

/*
	@brief		Computes a laplaican contrast weight map
	@function	cv::Mat laplacian_contrast(cv::Mat img)
*/
cv::Mat laplacian_contrast(cv::Mat img);

/*
	@brief		Computes a local contrast weight map
	@function	cv::Mat local_contrast(cv::Mat img, cv::Mat kernel)
*/
cv::Mat local_contrast(cv::Mat img, cv::Mat kernel);

/*
	@brief		Computes a saliency weight map
	@function	cv::Mat saliency(cv::Mat img, cv::Mat kernel)
*/
cv::Mat saliency(cv::Mat img, cv::Mat kernel);

/*
	@brief		Computes a exposedness weight map
	@function	cv::Mat exposedness(cv::Mat img)
*/
cv::Mat exposedness(cv::Mat img);

/*
	@brief		Normalizes two weigh maps
	@function	vector<Mat> weight_norm(cv::Mat w1, cv::Mat w2)
*/
vector<Mat> weight_norm(cv::Mat w1, cv::Mat w2);

/*
	@brief		Creates a laplacian pyramid
	@function	vector<Mat_<float>> laplacian_pyramid(cv::Mat img, int levels)
*/
vector<Mat_<float>> laplacian_pyramid(cv::Mat img, int levels);

/*
	@brief		Fuses two pyramids
	@function	Mat_<float> pyramid_fusion(Mat_<float> *pyramid, int levels)
*/
Mat pyramid_fusion(Mat *pyramid, int levels);

#if USE_GPU
	cv::cuda::GpuMat illuminationCorrection_GPU(cv::cuda::GpuMat srcGPU);

	void fft_GPU(const cv::cuda::GpuMat &srcGPU, cv::cuda::GpuMat &dst);

	cv::Mat gaussianFilter_GPU(cv::Mat img, float sigma, float high, float low);

	cv::cuda::GpuMat histStretch_GPU(cv::cuda::GpuMat srcGPU, float percent, int direction);

	cv::cuda::GpuMat ICM_GPU(cv::cuda::GpuMat srcGPU, float percent);

	cv::cuda::GpuMat dehazing_GPU(cv::cuda::GpuMat srcGPU);

	cv::cuda::GpuMat brightChannel_GPU(std::vector<cv::cuda::GpuMat> channels, int size);

	cv::cuda::GpuMat maxColDiff_GPU(std::vector<cv::cuda::GpuMat> channels);

	cv::cuda::GpuMat rectify_GPU(cv::cuda::GpuMat S, cv::cuda::GpuMat bc, cv::cuda::GpuMat mcd);

	std::vector<uchar> lightEstimation_GPU(cv::Mat src_gray, int size, cv::cuda::GpuMat bright_chan, std::vector<cv::Mat> channels);

	cv::cuda::GpuMat transmittance_GPU(cv::cuda::GpuMat correct, std::vector<uchar> A);

	cv::cuda::GpuMat dehaze_GPU(std::vector<cv::cuda::GpuMat> channels, std::vector<uchar> A, cv::cuda::GpuMat trans);
#endif


/********************************************************************/
/* Project: uw-img-proc									            */
/* Module: 	fusion										            */
/* File: 	fusion.cpp										        */
/* Created:	18/02/2019				                                */
/* Description:
	C++ module for underwater image enhancement using a fusion based
	strategy
 /*******************************************************************/

 /*******************************************************************/
 /* Created by:                                                     */
 /* Geraldine Barreto (@geraldinebc)                                */
 /*******************************************************************/


cv::Mat illuminationCorrection(cv::Mat src) {												// Homomorphic Filter
	Mat imgTemp1 = Mat::zeros(src.size(), CV_32FC1);
	normalize(src, imgTemp1, 0, 1, NORM_MINMAX, CV_32FC1);									// Normalize the channel
	imgTemp1 = imgTemp1 + 0.000001;
	log(imgTemp1, imgTemp1);																// Calculate the logarithm

	cv::Mat fftimg;
	fft(imgTemp1, fftimg);																	// Fourier transform

	cv::Mat_<float> filter = gaussianFilter(fftimg, 0.7, 1.0, 0.1);							// Gaussian Emphasis High-Pass Filter
	cv::Mat bimg;
	cv::Mat bchannels[] = { cv::Mat_<float>(filter), cv::Mat::zeros(filter.size(), CV_32F) };
	cv::merge(bchannels, 2, bimg);

	dftShift(bimg);																			// Shift the filter
	cv::mulSpectrums(fftimg, bimg, fftimg, 0);												// Apply the filter to the image in frequency domain

	cv::Mat ifftimg;					
	cv::dft(fftimg, ifftimg, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);						// Apply the inverse Fourier transform

	cv::Mat expimg = Mat::zeros(ifftimg.size(), CV_32FC1);									// Calculate the exponent
	cv::exp(ifftimg, expimg);

	cv::Mat dst;
	dst = cv::Mat(expimg, cv::Rect(0, 0, src.cols, src.rows));								// Eliminate the padding from the image
	normalize(dst, dst, 0, 255, NORM_MINMAX, CV_8U);										// Normalize the results
	return dst;
}

void fft(const cv::Mat &src, cv::Mat &dst) {												// Fast Fourier Transform
	cv::Mat padded;
	int m = cv::getOptimalDFTSize(src.rows);
	int n = cv::getOptimalDFTSize(src.cols);
	cv::copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, cv::BORDER_REPLICATE);// Resize to optimal size

	cv::Mat plane[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };	// Add complex column to store the imaginary result
	cv::Mat imgComplex;
	cv::merge(plane, 2, imgComplex);
	cv::dft(imgComplex, dst);																// Aply the Discrete Fourier Transform
	dst = dst / dst.total();																// Normalize
}

cv::Mat gaussianFilter(cv::Mat img, float sigma, float high, float low) {
	cv::Mat radius(img.rows, img.cols, CV_32F);
	cv::Mat expo(img.rows, img.cols, CV_32F);
	cv::Mat filter(img.rows, img.cols, CV_32F);
	int cx = img.rows / 2;
	int cy = img.cols / 2;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			radius.at<float>(i, j) = pow(i - cx, 2) + pow(j - cy, 2);
		}
	}
	exp(-radius / (2 * pow(sigma, 2)), expo);
	filter = (high - low) * (1 - expo) + low;												// High-pass Emphasis Filter
	return filter;
}

void dftShift(Mat &fImage) {																// Rearranges the quadrants
	Mat tmp, q0, q1, q2, q3;
	int cx = fImage.cols / 2;
	int cy = fImage.rows / 2;

	q0 = fImage(Rect(0, 0, cx, cy));
	q1 = fImage(Rect(cx, 0, cx, cy));
	q2 = fImage(Rect(0, cy, cx, cy));
	q3 = fImage(Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);																			// Each quadrant its replaced for its diagonally opposite quadrant
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

cv::Mat hueIllumination(cv::Mat src) {														// Corrects the color and illumination
	cv::Mat LAB, lab[3], dst;
	cvtColor(src, LAB, COLOR_BGR2Lab);														// Conversion to the Lab color model
	split(LAB, lab);
	lab[0] = illuminationCorrection(lab[0]);												// Correction of ununiform illumination
	lab[1] = 127.5 * lab[1] / mean(lab[1])[0];												// Grey World Assumption
	lab[2] = 127.5 * lab[2] / mean(lab[2])[0];
	merge(lab, 3, LAB);
	cvtColor(LAB, dst, COLOR_Lab2BGR);														// Conversion to the BGR color model
	return dst;
}

void getHistogram(cv::Mat *channel, cv::Mat *hist) {										// Computes the histogram of a single channel
	int histSize = 256;
	float range[] = { 0, 256 };																// The histograms ranges from 0 to 255
	const float* histRange = { range };
	calcHist(channel, 1, 0, Mat(), *hist, 1, &histSize, &histRange, true, false);
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
	return dst;
}

cv::Mat ICM(vector<Mat_<uchar>> channel, float percent) {								// Integrated Color Model
	Mat chan[3], result;
	for (int i = 0; i < 3; i++) chan[i] = histStretch(channel[i], percent, 1);			// Histogram stretching of each color channel
	merge(chan, 3, result);
	Mat HSV, hsv[3], dst;
	cvtColor(result, HSV, COLOR_BGR2HSV);												// Conversion to the HSV color model
	split(HSV, hsv);
	for (int i = 1; i < 3; i++) hsv[i] = histStretch(hsv[i], percent, 1);				// Histogram stretching of the Saturation and Value Channels
	merge(hsv, 3, HSV);
	cvtColor(HSV, dst, COLOR_HSV2BGR);													// Conversion to the BGR color model
	return dst;
}

cv::Mat dehazing(cv::Mat src) {															// Dehazed an underwater image
	vector<Mat_<uchar>> src_chan, new_chan;
	split(src, src_chan);

	new_chan.push_back(255 - src_chan[0]);												// Compute the new channels for the dehazing process
	new_chan.push_back(255 - src_chan[1]);
	new_chan.push_back(src_chan[2]);

	int size = sqrt(src.total()) / 40;													// Making the size bigger creates halos around objects
	cv::Mat bright_chan = brightChannel(new_chan, size);								// Compute the bright channel image
	cv::Mat mcd = maxColDiff(src_chan);													// Compute the maximum color difference

	cv::Mat src_HSV, S;
	cv::cvtColor(src, src_HSV, COLOR_BGR2HSV);
	extractChannel(src_HSV, S, 1);
	cv::Mat rectified = rectify(S, bright_chan, mcd);									// Rectify the bright channel image

	cv::Mat src_gray;
	cv::cvtColor(src, src_gray, COLOR_BGR2GRAY);
	vector<uchar> A;
	A = lightEstimation(src_gray, size, bright_chan, new_chan);							// Estimate the atmospheric light
	cv::Mat trans = transmittance(rectified, A);										// Compute the transmittance image

	cv::Mat filtered;
	guidedFilter(src_gray, trans, filtered, 30, 0.001, -1);								// Refine the transmittance image

	vector<Mat_<float>> chan_dehazed;
	chan_dehazed.push_back(new_chan[0]);
	chan_dehazed.push_back(new_chan[1]);
	chan_dehazed.push_back(new_chan[2]);
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

cv::Mat filter_mask() {
	float h[5] = { 1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0 };
	cv::Mat kernel = Mat(5, 5, CV_32F);
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) kernel.at<float>(i, j) = h[i] * h[j];
	}
	return kernel;
}

cv::Mat laplacian_contrast(cv::Mat img) {
	img.convertTo(img, CV_32F);
	cv::Mat laplacian = Mat(img.rows, img.cols, CV_32F);
	Laplacian(img, laplacian, img.depth());
	convertScaleAbs(laplacian, laplacian);
	return laplacian;
}

cv::Mat local_contrast(cv::Mat img, cv::Mat kernel) {
	img.convertTo(img, CV_32F);
	cv::Mat blurred = Mat(img.rows, img.cols, CV_32F);
	filter2D(img, blurred, img.depth(), kernel);
	cv::Mat contrast = Mat(img.rows, img.cols, CV_32F);
	contrast = abs(img.mul(img) - blurred.mul(blurred));
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			contrast.at<float>(i, j) = sqrt(contrast.at<float>(i, j));
		}
	}
	return contrast;
}

cv::Mat saliency(cv::Mat img, cv::Mat kernel) {
	cv::Mat blurred, img_Lab;
	blurred = Mat(img.rows, img.cols, CV_32F);
	filter2D(img, blurred, img.depth(), kernel);
	cvtColor(blurred, img_Lab, COLOR_BGR2Lab);
	cv::Mat chan_lab[3], l, a, b;
	split(img_Lab, chan_lab);
	chan_lab[0].convertTo(l, CV_32F);
	chan_lab[1].convertTo(a, CV_32F);
	chan_lab[2].convertTo(b, CV_32F);
	l = mean(l).val[0] - l;
	a = mean(a).val[0] - a;
	b = mean(b).val[0] - b;
	cv::Mat saliency = Mat::zeros(img.rows, img.cols, CV_32F);
	accumulateSquare(l, saliency);
	accumulateSquare(a, saliency);
	accumulateSquare(b, saliency);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			saliency.at<float>(i, j) = sqrt(saliency.at<float>(i, j));
		}
	}
	return saliency;
}

cv::Mat exposedness(cv::Mat img) {
	img.convertTo(img, CV_32F, 1.0 / 255.0);
	cv::Mat exposedness = Mat(img.rows, img.cols, CV_32F);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			exposedness.at<float>(i, j) = exp(-1.0 * pow(img.at<float>(i, j) - 0.5, 2.0) / (2.0 * pow(0.25, 2.0)));
		}
	}
	return exposedness;
}

vector<Mat> weight_norm(cv::Mat w1, cv::Mat w2) {
	w1.convertTo(w1, CV_32F);
	w2.convertTo(w2, CV_32F);
	vector<Mat> norm;
	norm.push_back(w1 / (w1 + w2));
	norm.push_back(w2 / (w1 + w2));
	return norm;
}

vector<Mat_<float>> laplacian_pyramid(cv::Mat img, int levels) {
	vector<Mat_<float>> l_pyr;
	Mat_<float> downsampled, upsampled, lap_pyr, current_layer = img;
	for (int i = 0; i < levels - 1; i++) {
		pyrDown(current_layer, downsampled);
		pyrUp(downsampled, upsampled, current_layer.size());
		subtract(current_layer, upsampled, lap_pyr);
		l_pyr.push_back(lap_pyr);
		current_layer = downsampled;
	}
	l_pyr.push_back(current_layer);
	return l_pyr;
}

Mat pyramid_fusion(Mat *pyramid, int levels) {
	for (int i = levels-1; i > 0; i--) {
		Mat_<float> upsampled;
		pyrUp(pyramid[i], upsampled, pyramid[i - 1].size());
		add(pyramid[i - 1], upsampled, pyramid[i - 1]);
	}
	pyramid[0].convertTo(pyramid[0], CV_8U);
	return pyramid[0];
}

#if USE_GPU
cv::cuda::GpuMat illuminationCorrection_GPU(cv::cuda::GpuMat srcGPU) {						// Homomorphic Filter
	cv::cuda::GpuMat imgTemp1 = Mat::zeros(srcGPU.size(), CV_32FC1);
	cv::cuda::normalize(srcGPU, imgTemp1, 0, 1, NORM_MINMAX, CV_32FC1);						// Normalize the channel
	cv::cuda::add(imgTemp1, imgTemp1, 0.000001);
	cv::cuda::log(imgTemp1, imgTemp1);														// Calculate the logarithm

	cv::cuda::GpuMat fftimg;
	fft_GPU(imgTemp1, fftimg);																// Fourier transform

	cv::Mat filter = gaussianFilter_GPU(fftimg, 0.7, 1.0, 0.5);								// Gaussian Emphasis High-Pass Filter
	cv::Mat bimg;
	cv::Mat bchannels[] = { cv::Mat_<float>(filter), cv::Mat::zeros(filter.size(), CV_32F) };
	merge(bchannels, 2, bimg);

	dftShift(bimg);																			// Shift the filter
	cv::cuda::GpuMat bimgGPU;
	bimgGPU.upload(bimg);
	cv::cuda::mulSpectrums(fftimg, bimgGPU, fftimg, 0);										// Apply the filter to the image in frequency domain

	cv::cuda::GpuMat ifftimg;
	cv::cuda::dft(fftimg, ifftimg, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);					// Apply the inverse Fourier transform
	cv::cuda::GpuMat expimg = Mat::zeros(ifftimg.size(), CV_32FC1);							// Calculate the exponent
	cv::cuda::exp(ifftimg, expimg);

	cv::cuda::GpuMat dst;
	dst = cv::cuda::GpuMat(expimg, cv::Rect(0, 0, srcGPU.cols, srcGPU.rows));					// Eliminate the padding from the image
	cv::cuda::normalize(dst, dst, 0, 255, NORM_MINMAX, CV_8U);								// Normalize the results
	return dst;
}

void fft_GPU(const cv::cuda::GpuMat &srcGPU, cv::cuda::GpuMat &dst) {						// Fast Fourier Transform
	cv::cuda::GpuMat padded, gpu_dft, dft_mag;
	int m = getOptimalDFTSize(srcGPU.rows);													// Optimal Size to calculate the Discrete Fourier Transform 
	int n = getOptimalDFTSize(srcGPU.cols);
	cv::cuda::copyMakeBorder(srcGPU, padded, 0, m - srcGPU.rows, 0, n - srcGPU.cols, BORDER_REPLICATE);	// Resize to optimal FFT size
	dst = cv::cuda::GpuMat(m, n, CV_32FC2);
	cv::cuda::dft(padded, dst, padded.size());												// Aply the Discrete Fourier Transform
	cv::cuda::divide(dst, dst.total(), dst);												// Normalize
}

cv::Mat gaussianFilter_GPU(cv::Mat img, float sigma, float high, float low) {
	cv::Mat radius(img.rows, img.cols, CV_32F);
	int cx = img.rows / 2;
	int cy = img.cols / 2;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			radius.at<float>(i, j) = - (pow(i - cx, 2) + pow(j - cy, 2)) / (2 * pow(sigma, 2));
		}
	}
	cv::cuda::GpuMat radius_GPU, expo, filterGPU, dst;
	radius_GPU.upload(radius);
	cv::cuda::exp(radius_GPU ), expo);
	cv::cuda::subtract(1, expo, dst);
	cv::cuda::addWeighted(dst, high - low, 0, 0, low, filterGPU);								// High-pass Emphasis Filter
	cv::Mat filter;
	filterGPU.download(filter);
	return filter;
}

cv::cuda::GpuMat histStretch_GPU(cv::cuda::GpuMat srcGPU, float percent, int direction) {
	cv::cuda::GpuMat hist;
	cv::Mat histogram;
	cv::cuda::calcHist(srcGPU, hist);
	hist.download(histogram);
	float percent_sum = 0.0, channel_min = -1.0, channel_max = -1.0;
	float percent_min = percent / 100.0, percent_max = 1.0 - percent_min;
	int i = 0;

	while (percent_sum < percent_max * srcGPU.total()) {
		if (percent_sum < percent_min * srcGPU.total()) channel_min++;
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
	for (int i = 0; i < 3; i++) chan[i] = histStretchGPU(channel[i], percent, 1);		// Histogram stretching of each color channel
	cv::cuda::merge(chan, 3, result);
	cv::cuda::GpuMat HSV, hsv[3], dst;
	cv::cuda::cvtColor(result, HSV, COLOR_BGR2HSV);										// Conversion to the HSV color model
	cv::cuda::split(HSV, hsv);
	for (int i = 1; i < 3; i++) hsv[i] = histStretchGPU(hsv[i], percent, 1);			// Histogram stretching of the Saturation and Value Channels
	cv::cuda::merge(hsv, 3, HSV);
	cv::cuda::cvtColor(HSV, dst, COLOR_HSV2BGR);										// Conversion to the BGR color model
	return dst;
}

cv::cuda::GpuMat dehazing_GPU(cv::cuda::GpuMat srcGPU) {
	cv::cuda::GpuMat new_chanGPU, dstGPU;

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
}

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
		cv::cuda::multiply(sub, 255.0 / (255.0 - A[i]), t[i]);
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
	cv::cuda::subtract(255.0, B[1], C[1]);
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

//Mat lookUpTable(1, 256, CV_32F);
//for (int i = 0; i < 256; ++i) lookUpTable.at<float>(0,i) = 255.0 * sqrt(0.32 * log(255.0/(255.0-i)));
//LUT(x, lookUpTable, dst);
//dst.convertTo(dst, CV_8U);


/********************************************************************/
/* Project: uw-img-proc									            */
/* Module: 	fusion										            */
/* File: 	main.cpp										        */
/* Created:	18/02/2019				                                */
/* Description:
	C++ module for underwater image enhancement using a fusion based
	strategy
 /*******************************************************************/

 /*******************************************************************/
 /* Created by:                                                     */
 /* Geraldine Barreto (@geraldinebc)                                */
 /*******************************************************************/

#define ABOUT_STRING "Fusion Enhancement Module"


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
		"{show    |       | Show image comparison or not (ON: 1,OFF: 0)}"		// Show image comparison (optional)
		"{cuda    |       | Use CUDA or not (ON: 1, OFF: 0)}"			        // Use CUDA (if available) (optional)
		"{time    |       | Show time measurements or not (ON: 1, OFF: 0)}"		// Show time measurements (optional)
		"{help h usage ?  |       | Print help message}";						// Show help (optional)

	CommandLineParser cvParser(argc, argv, keys);
	cvParser.about(ABOUT_STRING);	// Adds "about" information to the parser method

	//**************************************************************************
	std::cout << ABOUT_STRING << endl;
	std::cout << endl << "Built with OpenCV " << CV_VERSION << endl;

	// If the number of arguments is lower than 4, or contains "help" keyword, then we show the help
	if (argc < 5 || cvParser.has("help")) {
		std::cout << endl << "C++ Module for fusion enhancement" << endl;
		std::cout << endl << "Arguments are:" << endl;
		std::cout << "\t*Input: Input image name with path and extension" << endl;
		std::cout << "\t*Output: Output image name with path and extension" << endl;
		std::cout << "\t*-cuda=0 or -cuda=1 (ON: 1, OFF: 0)" << endl;
		std::cout << "\t*-time=0 or -time=1 (ON: 1, OFF: 0)" << endl;
		std::cout << "\t*-show=0 or -show=1 (ON: 1, OFF: 0)" << endl;
		std::cout << endl << "Example:" << endl;
		std::cout << "\timg1.jpg img2.jpg -cuda=0 -time=0 -show=0 -d=S -m=F" << endl;
		std::cout << "\tThis will open 'input.jpg' enhance the image and save it in 'output.jpg'" << endl << endl;
		return 0;
	}

	int CUDA = 0;                                   // Default option (running with CPU)
	int Time = 0;                                   // Default option (not showing time)
	int Show = 0;                                   // Default option (not showing comparison)

	std::string InputFile = cvParser.get<cv::String>(0);	// String containing the input file path+name+extension from cvParser function
	std::string OutputFile = cvParser.get<cv::String>(1);	// String containing the input file path+name+extension from cvParser function
	std::string implementation;								// CPU or GPU implementation
	Show = cvParser.get<int>("show");						// Gets argument -show=x, where 'x' defines if the matches will show or not
	Time = cvParser.get<int>("time");						// Gets argument -time=x, where 'x' defines if execution time will show or not

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

	cv::Mat input, dst;
	input = imread(InputFile, cv::IMREAD_COLOR);

	if (input.empty()) {
		std::cout << "Error occured when loading the image" << endl << endl;
		return -1;
	}

	// Start time measurement
	t = (double)getTickCount();

	// GPU Implementation
#if USE_GPU
	if (CUDA) {
		GpuMat srcGPU;
		srcGPU.upload(src);
	}
#endif

	std::cout << endl << "Applying fusion enhancement" << endl;

	// CPU Implementation
	if (!CUDA) {
		cv::Mat HSV, LAB, chanHSV[3], chanLAB[3];
		cvtColor(input, HSV, cv::COLOR_BGR2HSV);
		split(HSV, chanHSV);
		cvtColor(input, LAB, cv::COLOR_BGR2Luv);
		split(LAB, chanLAB);
		Scalar meanS, stddevS, meanV, stddevV;
		meanStdDev(chanHSV[1], meanS, stddevS);
		meanStdDev(chanHSV[2], meanV, stddevV);

		Mat src[2];
		vector<Mat_<uchar>> channels;
		split(input, channels);
		bool flag = 0;

		// Histogram Stretching or Hue and Illumination Correction
		if ((meanV[0] <= 115 & (mean(chanLAB[1])[0] <= 110 | mean(chanLAB[2])[0] <= 110)) | (meanS[0] >= 240 & stddevS[0] <= 15)) {
			src[0] = hueIllumination(input);
			flag = 1;
		}
		else src[0] = ICM(channels, 0.5);

		// Dehazing or Hue and Illumination Correction
		if (stddevV[0] >= 65 | (meanV[0] >= 160 & (mean(chanLAB[1])[0] <= 100 | mean(chanLAB[2])[0] <= 100))) {
			if (flag == 1) dst = src[0];
			else src[1] = hueIllumination(input);
		}
		else src[1] = dehazing(input);

		if (dst.empty()) {
			cv::Mat Lab[2], L[2];
			cvtColor(src[0], Lab[0], COLOR_BGR2Lab);
			extractChannel(Lab[0], L[0], 0);
			cvtColor(src[1], Lab[1], COLOR_BGR2Lab);
			extractChannel(Lab[1], L[1], 0);

			cv::Mat kernel = filter_mask();

			// Normalized weights
			vector<Mat> w1_norm, w2_norm, w3_norm, w4_norm;
			w1_norm = weight_norm(laplacian_contrast(L[0]), laplacian_contrast(L[1]));
			w2_norm = weight_norm(local_contrast(L[0], kernel), local_contrast(L[1], kernel));
			w3_norm = weight_norm(saliency(src[0], kernel), saliency(src[1], kernel));
			w4_norm = weight_norm(exposedness(L[0]), exposedness(L[1]));

			// Weight sum of each input
			cv::Mat w_norm[2];
			w_norm[0] = (w1_norm[0] + w2_norm[0] + w3_norm[0] + w4_norm[0]) / 4;
			w_norm[1] = (w1_norm[1] + w2_norm[1] + w3_norm[1] + w4_norm[1]) / 4;

			// Gaussian pyramids of the weights
			int levels = 5;
			vector<Mat> pyramid_g0, pyramid_g1;
			buildPyramid(w_norm[0], pyramid_g0, levels - 1);
			buildPyramid(w_norm[1], pyramid_g1, levels - 1);

			cv::Mat channels_0[3], channels_1[3];
			split(src[0], channels_0);
			split(src[1], channels_1);

			// Laplacian pyramids of the inputs channels (BGR)
			vector<Mat_<float>> pyramid_l0_b = laplacian_pyramid(channels_0[0], levels);
			vector<Mat_<float>> pyramid_l0_g = laplacian_pyramid(channels_0[1], levels);
			vector<Mat_<float>> pyramid_l0_r = laplacian_pyramid(channels_0[2], levels);

			vector<Mat_<float>> pyramid_l1_b = laplacian_pyramid(channels_1[0], levels);
			vector<Mat_<float>> pyramid_l1_g = laplacian_pyramid(channels_1[1], levels);
			vector<Mat_<float>> pyramid_l1_r = laplacian_pyramid(channels_1[2], levels);

			// Fusion of the inputs with their respective weights
			Mat chan_b[5], chan_g[5], chan_r[5];
			for (int i = 0; i < 5; i++) {
				pyramid_g0[i].convertTo(pyramid_g0[i], CV_32F);
				pyramid_g1[i].convertTo(pyramid_g1[i], CV_32F);
				add(pyramid_l0_b[i].mul(pyramid_g0[i]), pyramid_l1_b[i].mul(pyramid_g1[i]), chan_b[i]);
				add(pyramid_l0_g[i].mul(pyramid_g0[i]), pyramid_l1_g[i].mul(pyramid_g1[i]), chan_g[i]);
				add(pyramid_l0_r[i].mul(pyramid_g0[i]), pyramid_l1_r[i].mul(pyramid_g1[i]), chan_r[i]);
			}

			// Pyramid reconstruction
			cv::Mat channel[3];
			channel[0] = pyramid_fusion(chan_b, levels);
			channel[1] = pyramid_fusion(chan_g, levels);
			channel[2] = pyramid_fusion(chan_r, levels);
			merge(channel, 3, dst);
			
			//Mat test;
			//vconcat(src[0], src[1], test);
			//imwrite(OutputFile + "_t.jpg", test);

			//Mat weight1, weight2, weights;
			//hconcat(w1_norm[0], w2_norm[0], weight1);
			//hconcat(weight1, w3_norm[0], weight1);
			//hconcat(weight1, w4_norm[0], weight1);
			//hconcat(weight1, w_norm[0], weight1);
			//hconcat(w1_norm[1], w2_norm[1], weight2);
			//hconcat(weight2, w3_norm[1], weight2);
			//hconcat(weight2, w4_norm[1], weight2);
			//hconcat(weight2, w_norm[1], weight2);
			//vconcat(weight1, weight2, weights);
			//imwrite(OutputFile + "_w.jpg", 255.0*weights);
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
		file << endl << OutputFile << ";" << input.rows << ";" << input.cols << ";" << t;
	}

	std::cout << endl << "Saving processed image" << endl;
	imwrite(OutputFile, dst);

	if (Show) {
		std::cout << endl << "Showing image comparison" << endl;
		Mat comparison;
		hconcat(input, dst, comparison);
		namedWindow("Comparison", WINDOW_KEEPRATIO);
		imshow("Comparison", comparison);
	}

	waitKey(0);
	return 0;
}