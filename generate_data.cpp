// generate_data.cpp

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

	cv::Mat matTrainingNumbers;		// input image
	cv::Mat matGrayscale;			// 
	cv::Mat matBlurred;				// declare various images
	cv::Mat matThresh;				//
	cv::Mat matThreshCopy;			//

	std::vector<std::vector<cv::Point> > ptContours;		// declare contours vector
	std::vector<cv::Vec4i> v4iHierarchy;					// declare contours hierarchy

	cv::Mat matClassificationInts;		// these are our training classifications, note we will have to perform some conversions before writing to file later
	
												// these are our training images, due to the data types that the KNN object KNearest requires,
	cv::Mat matTrainingImages;					// we have to declare a single Mat, then append to it as though it's a vector,
												// also we will have to perform some conversions before writing to file later

								// possible chars we are interested in are digits 0 through 9, put these in vector intValidChars
	std::vector<int> intValidChars = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };

	matTrainingNumbers = cv::imread("training_numbers.png");			// read in training numbers image

	if (matTrainingNumbers.empty()) {							// if unable to open image
		std::cout << "error: image not read from file\n\n";		// show error message on command line
		return(0);												// and exit program
	}

	cv::cvtColor(matTrainingNumbers, matGrayscale, CV_BGR2GRAY);		// convert to grayscale

	cv::GaussianBlur(matGrayscale,			// input image
		matBlurred,							// output image
		cv::Size(5, 5),						// smoothing window width and height in pixels
		0);									// sigma value, determines how much the image will be blurred, zero makes function choose the sigma value

										// filter image from grayscale to black and white
	cv::adaptiveThreshold(matBlurred,							// input image
						  matThresh,							// output image
						  255,									// make pixels that pass the threshold full white
						  cv::ADAPTIVE_THRESH_GAUSSIAN_C,		// use gaussian rather than mean, seems to give better results
						  cv::THRESH_BINARY_INV,				// invert so foreground will be white, background will be black
						  11,									// size of a pixel neighborhood used to calculate threshold value
						  2);									// constant subtracted from the mean or weighted mean

	cv::imshow("matThresh", matThresh);			// show threshold image for reference

	matThreshCopy = matThresh.clone();			// make a copy of the thresh image, this in necessary b/c findContours modifies the image

	cv::findContours(matThreshCopy,					// input image, make sure to use a copy since the function will modify this image in the course of finding contours
					 ptContours,					// output contours
					 v4iHierarchy,					// output hierarchy
					 cv::RETR_EXTERNAL,				// retrieve the outermost contours only
					 cv::CHAIN_APPROX_SIMPLE);		// compress horizontal, vertical, and diagonal segments and leave only their end points

	for (int i = 0; i < ptContours.size(); i++) {						// for each contour
		if (cv::contourArea(ptContours[i]) > MIN_CONTOUR_AREA) {			// if contour is big enough to consider
			cv::Rect boundingRect = cv::boundingRect(ptContours[i]);			// get the bounding rect

			cv::rectangle(matTrainingNumbers, boundingRect, cv::Scalar(0, 0, 255), 2);		// draw red rectangle around each contour as we ask user for input

			cv::Mat matROI = matThresh(boundingRect);			// get ROI image of bounding rect

			cv::Mat matROIResized;
			cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));		// resize image, this will be more consistent for recognition and storage
			
			cv::imshow("matROI", matROI);								// show ROI image for reference
			cv::imshow("matROIResized", matROIResized);					// show resized ROI image for reference
			cv::imshow("matTrainingNumbers", matTrainingNumbers);		// show training numbers image, this will now have red rectangles drawn on it

			int intChar = cv::waitKey(0);			// get key press

			if (intChar == 27) {		// if esc key was pressed
				return(0);				// exit program
			} else if (std::find(intValidChars.begin(), intValidChars.end(), intChar) != intValidChars.end()) {  // else if the char is in the list of chars we are looking for . . .
				
				matClassificationInts.push_back(intChar);		// append classification char to integer list of chars (we will convert later before writing to file)

				cv::Mat matImageFloat;
				matROIResized.convertTo(matImageFloat, CV_32FC1);		// convert Mat to float
				
				cv::Mat matImageReshaped = matImageFloat.reshape(1, 1);		// flatten

				matTrainingImages.push_back(matImageReshaped);		// add to Mat as though it was a vector, this is necessary due to the
																	// data types that KNearest.train accepts
			}	// end if
		}	// end if
	}	// end for

	std::cout << "training complete\n\n";

			// save classifications to file ///////////////////////////////////////////////////////

	cv::Mat matClassificationIntsReshaped = matClassificationInts.reshape(1, 1);		// flatten

	cv::Mat matClassificationFloats;
	matClassificationInts.convertTo(matClassificationFloats, CV_32FC1);			// convert ints to floats

	cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::WRITE);			// open the classifications file

	if (fsClassifications.isOpened() == false) {														// if the file was not opened successfully
		std::cout << "error, unable to open training classifications file, exiting program\n\n";		// show error message
		return(0);																						// and exit program
	}

	fsClassifications << "classifications" << matClassificationFloats;		// write classifications into classifications section of classifications file
	fsClassifications.release();											// close the classifications file
	
			// save training images to file ///////////////////////////////////////////////////////

	cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::WRITE);			// open the training images file

	if (fsTrainingImages.isOpened() == false) {													// if the file was not opened successfully
		std::cout << "error, unable to open training images file, exiting program\n\n";			// show error message
		return(0);																				// and exit program
	}
	
	fsTrainingImages << "images" << matTrainingImages;		// write training images into images section of images file
	fsTrainingImages.release();								// close the training images file

	return(0);
}

