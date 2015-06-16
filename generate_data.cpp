// OCR3b.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<sstream>

// global variables ///////////////////////////////////////////////////////////////////////////////
const int MIN_CONTOUR_AREA = 100;

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

///////////////////////////////////////////////////////////////////////////////////////////////////
class ContourWithData {
public:
	std::vector<cv::Point> ptContour;			// contour
	cv::Rect boundingRect;						// bounding rect for contour
	float fltArea;								// area of contour

	///////////////////////////////////////////////////////////////////////////////////////////////
	bool checkIfContourIsValid() {									// obviously in a production grade program
		if (fltArea < MIN_CONTOUR_AREA) return false;				// we would have a much more robust function for 
		return true;												// identifying if a contour is valid !!
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {		// this function allows us to sort
		return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);													// the contours from left to right
	}

};

///////////////////////////////////////////////////////////////////////////////////////////////////
int main() {
	std::vector<ContourWithData> allContoursWithData;			// declare empty vectors,
	std::vector<ContourWithData> validContoursWithData;			// we will fill these shortly

			// read in training classifications
	
	cv::Mat matClassificationFloats;	// we will read the classification numbers into this variable as though it is a vector

	cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);		// open the classifications file

	if (fsClassifications.isOpened() == false) {
		std::cout << "error, unable to open training classifications file, exiting program\n\n";
		return(0);
	}

	fsClassifications["classifications"] >> matClassificationFloats;
	fsClassifications.release();

													// read in training images
	cv::Mat matTrainingImages;

	cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);

	if (fsTrainingImages.isOpened() == false) {
		std::cout << "error, unable to open training images file, exiting program\n\n";
		return(0);
	}

	fsTrainingImages["images"] >> matTrainingImages;
	fsTrainingImages.release();

	cv::KNearest kNearest = cv::KNearest();

																	// pass in the training images and classifications,
	kNearest.train(matTrainingImages, matClassificationFloats);		// note these both have to be of type Mat (a single Mat)
																	// even though in reality they are multiple images / numbers

	cv::Mat matTestingNumbers = cv::imread("test_numbers.png");

	if (matTestingNumbers.empty()) {								// if unable to open image
		std::cout << "error: image not read from file\n\n";			// show error message on command line
		return(0);													// and exit program
	}

	cv::Mat matGrayscale;
	cv::Mat matBlurred;
	cv::Mat matThresh;

	cv::cvtColor(matTestingNumbers, matGrayscale, CV_BGR2GRAY);		// convert to grayscale

	cv::GaussianBlur(matGrayscale,			// input image
		matBlurred,							// output image
		cv::Size(5, 5),						// smoothing window width and height in pixels
		0);									// sigma value, determines how much the image will be blurred, zero makes function choose the sigma value

	cv::adaptiveThreshold(matBlurred, matThresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);

	cv::imshow("matThresh", matThresh);

	cv::Mat matThreshCopy;

	matThreshCopy = matThresh.clone();

	std::vector<std::vector<cv::Point> > ptContours;
	std::vector<cv::Vec4i> v4iHierarchy;

	cv::findContours(matThreshCopy, ptContours, v4iHierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < ptContours.size(); i++) {
		ContourWithData contourWithData;
		contourWithData.ptContour = ptContours[i];
		contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);
		contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);
		allContoursWithData.push_back(contourWithData);
	}
	
	for (int i = 0; i < allContoursWithData.size(); i++) {					// for all contours
		if (allContoursWithData[i].checkIfContourIsValid()) {				// check if valid
			validContoursWithData.push_back(allContoursWithData[i]);		// if so, append to valid contour list
		}
	}
															// sort contours from left to right
	std::sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectXPosition);

	std::string strFinalString;

	for (int i = 0; i < validContoursWithData.size(); i++) {			// for each contour
		
																		// draw a green rect around the current char
		cv::rectangle(matTestingNumbers, validContoursWithData[i].boundingRect, cv::Scalar(0, 255, 0), 2);

		cv::Mat matROI = matThresh(validContoursWithData[i].boundingRect);

		cv::Mat matROIResized;
		cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));

		cv::Mat matROIFloat;

		matROIResized.convertTo(matROIFloat, CV_32FC1);

		float fltCurrentChar = kNearest.find_nearest(matROIFloat.reshape(1, 1), 1);
		
		strFinalString = strFinalString + char(int(fltCurrentChar));
	}
	
	std::cout << "\n\n" << strFinalString << "\n\n";		// show the full string
	
	cv::imshow("matTestingNumbers", matTestingNumbers);
	
	cv::waitKey(0);
	
	return(0);
}


