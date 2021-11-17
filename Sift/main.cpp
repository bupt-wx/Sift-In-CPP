#include <Windows.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv.hpp>

#include "sift.h"

using namespace std;
using namespace cv;

int main() 
{
	//Read in picture
	Mat src1 = imread("../pictures/1.jpg");
	Mat src2 = imread("../pictures/2.jpg");
	if (!src1.data || !src2.data)
	{
		cout << "loading failed" << endl;
		system("pause");
		return -1;
	}
	if (src1.channels() != 3 || src2.channels() != 3) return -2;

	vector<keypoints> features1, features2;

	//Feature point detection and description
	Sift(src1, features1, 1.6);
	Sift(src2, features2, 1.6);

	Mat src1_copy, src2_copy;
	src1_copy = src1.clone();
	src2_copy = src2.clone();
	////Draw key points
	DrawKeyPoints(src1_copy, features1);
	DrawKeyPoints(src2_copy, features2);
	////Draw sift features
	DrawSiftFeatures(src1_copy, features1);
	DrawSiftFeatures(src2_copy, features2);


	//∆¥Ω”∆•≈‰
	int row1 = src1.rows, col1 = src1.cols, row2 = src2.rows, col2 = src2.cols;
	int row = max(row1, row2), col = col1 + col2 + 100;
	Mat splicing_mat;
	Composite(splicing_mat, src1, src2);
	//imshow("splicing_mat", splicing_mat);

	vector<matchpoints> match;
	match = Compute_Match(features1, features2);
	cout << "row1: " << row1 << " col1: " << col1 << endl;
	//cout << match.size() << endl;
	for (matchpoints mp : match)
	{
		line(splicing_mat, Point2i(mp.p1.x*0.5, mp.p1.y*0.5), Point2i(mp.p2.x*0.5 + col1 + 100, mp.p2.y*0.5), Scalar(rand() & 255, rand() & 255, rand() & 255), 1, 8, 0);
		//Scalar(rand() & 255, rand() & 255, rand() & 255)
	}
	imshow("splicing_mat", splicing_mat);

	vector<Point2f> imagePoints1, imagePoints2;
	for (matchpoints mp : match)
	{
		imagePoints1.push_back(Point2i(mp.p1.x, mp.p1.y));
		imagePoints2.push_back(Point2i(mp.p2.x, mp.p2.y));
	}
	Mat homo = findHomography(imagePoints2, imagePoints1, CV_RANSAC);
	cout << "±‰ªªæÿ’ÛŒ™£∫\n" << homo << endl;
	Mat stitchedImage = Mat::zeros(col1 + col2, row, CV_8UC3);
	warpPerspective(src2, stitchedImage, homo, Size(col1+col2, row));
	//Mat splicing_resut = Blend_Image(src1, imageTransform2);
	Mat half(stitchedImage, Rect(0, 0, col1, row1));
	src1.copyTo(half);
	imshow("imageTransform2", stitchedImage);
	waitKey(0);

	return 0;
}