// OCR3a.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<vector>

// global variables ///////////////////////////////////////////////////////////////////////////////
const int MIN_CONTOUR_AREA = 100;

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

///////////////////////////////////////////////////////////////////////////////////////////////////
int main() {

	cv::Mat matTrainingNumbers;
	cv::Mat matGrayscale;
	cv::Mat matBlurred;
	cv::Mat matThresh;
	cv::Mat matThreshCopy;

	std::vector<std::vector<cv::Point> > ptContours;
	std::vector<cv::Vec4i> v4iHierarchy;

	//std::vector<int> intClassifications;
	cv::Mat matClassificationInts;
	cv::Mat matImages;


	// int intValidChars2[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };

	std::vector<int> intValidChars = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };

	matTrainingNumbers = cv::imread("training_numbers.png");			// open image

	if (matTrainingNumbers.empty()) {									// if unable to open image
		std::cout << "error: image not read from file\n\n";		// show error message on command line
		return(0);												// and exit program
	}

	cv::cvtColor(matTrainingNumbers, matGrayscale, CV_BGR2GRAY);		// convert to grayscale

	cv::GaussianBlur(matGrayscale,			// input image
		matBlurred,							// output image
		cv::Size(5, 5),						// smoothing window width and height in pixels
		0);									// sigma value, determines how much the image will be blurred, zero makes function choose the sigma value

	cv::adaptiveThreshold(matBlurred, matThresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);

	cv::imshow("matThresh", matThresh);

	matThreshCopy = matThresh.clone();

	cv::findContours(matThreshCopy, ptContours, v4iHierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < ptContours.size(); i++) {
		if (cv::contourArea(ptContours[i]) > MIN_CONTOUR_AREA) {
			cv::Rect boundingRect = cv::boundingRect(ptContours[i]);

			boundingRect.x;
			boundingRect.y;
			boundingRect.width;
			boundingRect.height;

			cv::rectangle(matTrainingNumbers, boundingRect, cv::Scalar(0, 0, 255), 2);

			cv::Mat matROI = matThresh(boundingRect);

			cv::Mat matROIResized;
			cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));

			cv::imshow("matROI", matROI);
			cv::imshow("matROIResized", matROIResized);
			cv::imshow("matTrainingNumbers", matTrainingNumbers);

			int intChar = cv::waitKey(0);

			if (intChar == 27) {
				// exit program
			}
			else if (std::find(intValidChars.begin(), intValidChars.end(), intChar) != intValidChars.end()) {
				matClassificationInts.push_back(intChar);


				cv::Mat matImageFloat;
				

				matROIResized.convertTo(matImageFloat, CV_32FC1);
				cv::Mat matImageReshaped = matImageFloat.reshape(1, 1);

				matImages.push_back(matImageReshaped);


			}

		}
	}

	/*
	for (int i = 0; i < matClassificationInts.size().height; i++) {			// show contents of classifications for debugging
		std::cout << matClassificationInts.at<int>(i, 0) << "\n";
	}
	*/

	std::cout << "training complete\n\n";
																	// save classifications to file
	cv::Mat matClassificationIntsReshaped = matClassificationInts.reshape(1, 1);

	cv::Mat matClassificationFloats;

	matClassificationInts.convertTo(matClassificationFloats, CV_32FC1);

	cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::WRITE);

	if (fsClassifications.isOpened() == false) {
		std::cout << "error, unable to open training classifications file, exiting program\n\n";
		return(0);
	}

	fsClassifications << "classifications" << matClassificationFloats;
	fsClassifications.release();


																	// save images to file
	cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::WRITE);

	if (fsTrainingImages.isOpened() == false) {
		std::cout << "error, unable to open training images file, exiting program\n\n";
		return(0);
	}

	fsTrainingImages << "images" << matImages;
	fsTrainingImages.release();

	return(0);
}

