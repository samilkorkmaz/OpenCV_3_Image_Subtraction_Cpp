// ImageSubtractionCpp.sln
// main.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>

#include "Blob.h"

// global variables ///////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

void showContours(cv::Mat image, std::vector<std::vector<cv::Point> > contours, cv::String name) {
	cv::Mat imgContours(image.size(), CV_8UC3, SCALAR_BLACK);
	cv::drawContours(imgContours, contours, -1, SCALAR_WHITE, -1);
	cv::imshow(name, imgContours);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
int main(void) {
	cv::VideoCapture capVideo(OPENCV_FOLDER"..\\..\\..\\sources\\samples\\data\\vtest.avi");
	//cv::VideoCapture capVideo("C:/Users/sam/Downloads/YBSYeniHatTest3005.mp4");
	if (!capVideo.isOpened()) { // if unable to open video file
		std::cout << "\nerror reading video file" << std::endl << std::endl;      
		getchar();                    
		return(-1);                                                              
	}
	if (capVideo.get(cv::CAP_PROP_FRAME_COUNT) < 2) {
		std::cout << "\nerror: video file must have at least two frames";
		getchar();
		return(-2);
	}
	cv::Mat frame1;
	cv::Mat frame2;
	capVideo.read(frame1);
	capVideo.read(frame2);

	const char ESC_KEY = 27; 
	char keyPressed = 0;
	while (capVideo.isOpened() && keyPressed != ESC_KEY) {
		std::vector<Blob> blobs;

		cv::Mat frame1Copy = frame1.clone();
		cv::Mat frame2Copy = frame2.clone();

		//convert to grayscale:
		cv::cvtColor(frame1Copy, frame1Copy, cv::COLOR_BGR2GRAY);
		cv::cvtColor(frame2Copy, frame2Copy, cv::COLOR_BGR2GRAY);

		//blur both frames to prevent small differences like ribbons moving with wind from being detected
		cv::Size blurSize = cv::Size(5, 5);
		double sigmaX = 0;
		cv::GaussianBlur(frame1Copy, frame1Copy, blurSize, sigmaX);
		cv::GaussianBlur(frame2Copy, frame2Copy, blurSize, sigmaX);

		cv::Mat frameDiff;
		cv::absdiff(frame1Copy, frame2Copy, frameDiff); //difference between two frames

		double tresh = 30; double maxVal = 255;
		cv::Mat frameThresh;
		cv::threshold(frameDiff, frameThresh, tresh, maxVal, cv::THRESH_BINARY); //diff values that are above threshold of 30

		cv::imshow("frameThresh", frameThresh); //show differences that are above treshold

		//remove noise and settle down imperfections:
		cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
		cv::dilate(frameThresh, frameThresh, structuringElement5x5); //increase size, filling holes
		//cv::imshow("imgThreshDilate1", imgThresh);
		
		//get contours of treshold image:
		cv::Mat imgThreshCopy = frameThresh.clone();
		std::vector<std::vector<cv::Point> > treshContours;
		cv::findContours(imgThreshCopy, treshContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		//show contours of threshold image:
		showContours(frameThresh, treshContours, "contours");

		//form convex hulls:
		std::vector< std::vector<cv::Point> > convexHullsContours(treshContours.size());
		for (unsigned int i = 0; i < treshContours.size(); i++) {
			cv::convexHull(treshContours[i], convexHullsContours[i]);
		}
		for (auto &convexHull : convexHullsContours) {
			Blob possibleBlob(convexHull);
			//if convexHull satisfies the following geometric criteria, add it to list
			if (possibleBlob.boundingRect.area() > 100 &&
				possibleBlob.dblAspectRatio >= 0.2 &&
				possibleBlob.dblAspectRatio <= 1.2 &&
				possibleBlob.boundingRect.width > 15 &&
				possibleBlob.boundingRect.height > 20 &&
				possibleBlob.dblDiagonalSize > 30.0) {
				blobs.push_back(possibleBlob);
			}
		}

		//show satistactory convex hulls:
		convexHullsContours.clear();
		for (auto &blob : blobs) {
			convexHullsContours.push_back(blob.contour);
		}
		showContours(frameThresh, convexHullsContours, "convexHulls");

		//get boxes around convex hulls:
		frame2Copy = frame2.clone(); // get another copy of frame2 since we changed the previous frame2 copy in the processing above
		for (auto &blob : blobs) {                                                  
			cv::rectangle(frame2Copy, blob.boundingRect, SCALAR_RED, 2);      // draw a red box around the blob
			cv::circle(frame2Copy, blob.centerPosition, 3, SCALAR_GREEN, -1); // draw a filled-in green circle at the center
		}
		cv::imshow("boxes", frame2Copy);

		// now we prepare for the next iteration
		frame1 = frame2.clone();  // move frame1 up to where frame2 is
		if ((capVideo.get(cv::CAP_PROP_POS_FRAMES) + 1) < capVideo.get(cv::CAP_PROP_FRAME_COUNT)) { // if there is at least one more frame
			capVideo.read(frame2);  // read next frame
		}
		else {                                                  
			std::cout << "end of video\n";                      
			break;  // and jump out of while loop
		}
		keyPressed = cv::waitKey(1);
	}

	if (keyPressed != ESC_KEY) { // if the user did not press esc (i.e. we reached the end of the video)
		cv::waitKey(0);  // hold the windows open to allow the "end of video" message to show
	}
	// note that if the user did press esc, we don't need to hold the windows open, we can simply let the program end which will close the windows
	return(0);
}
