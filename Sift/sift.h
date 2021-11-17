#pragma once

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

using namespace std;
using namespace cv;

typedef double pixel_t; //定义像素类型

#define SIGMA 1.6
#define INTERVALS 3
#define INIT_SIGMA 0.5

#define RATIO 10
#define DXTHRESHOLD 0.03
#define IMG_BORDER 5
#define MAX_INTERPOLATION_STEPS 5

#define ORI_HIST_BINS 36
#define ORI_SIGMA_TIMES 1.5
#define ORI_WINDOW_RADIUS 3.0*ORI_SIGMA_TIMES
#define ORI_SMOOTH_TIMES 2
#define ORI_PEAK_RATIO 0.8
#define DESCR_SCALE_ADJUST 3
#define DESCR_HIST_BINS 8
#define DESCR_WINDOW_WIDTH 4
#define DESCR_MAG_THR 0.2
#define INT_DESCR_FCTR 512.0 

#define FEATURE_ELEMENT_LENGTH 128

#define MAX_LOSS 200

//关键点/特征点的结构体声明
struct keypoints
{
	int octave;
	int interval;
	double offset_interval;

	int x;
	int y;
	double scale;

	double dx;
	double dy;

	double offset_x;
	double offset_y;

	double val;
	double octave_scale;
	double ori;
	int    descr_length;
	double descriptor[FEATURE_ELEMENT_LENGTH];
};

//匹配点的结构体声明
struct matchpoints 
{
	Point2i p1, p2;
	matchpoints(Point2i pt1, Point2i pt2)
	{
		p1 = pt1;
		p2 = pt2;
	}
};

void Sift(const Mat &src, vector<keypoints>& features, double sigma = SIGMA, int intervals = INTERVALS);

void CreateInitSmoothGray(const Mat &src, Mat &dst, double);
void ConvertToGray(const Mat& src, Mat& dst);

void UpSample(const Mat& src, Mat& dst);
void DownSample(const Mat& src, Mat& dst);
void Sub(const Mat& a, const Mat& b, Mat & c);

void GaussianSmooth(const Mat &src, Mat &dst, double sigma);
void GaussianPyramid(const Mat &src, vector<Mat>&gauss_pyr, int octaves, int intervals, double sigma);
void DogPyramid(const vector<Mat>& gauss_pyr, vector<Mat>& dog_pyr, int octaves, int intervals);

void write_pyr(const vector<Mat>& pyr, const char* dir);

void DetectionLocalExtrema(const vector<Mat>& dog_pyr, vector<keypoints>& extrema, int octaves, int intervals);
bool isExtremum(int x, int y, const vector<Mat>& dog_pyr, int index);
keypoints * InterploationExtremum(int x, int y, const vector<Mat>& dog_pyr, int index, int octave, int interval, double dxthreshold = DXTHRESHOLD);
bool passEdgeResponse(int x, int y, const vector<Mat>& dog_pyr, int index, double r = RATIO);

void GetOffsetX(int x, int y, const vector<Mat>& dog_pyr, int index, double *offset_x);
double GetFabsDx(int x, int y, const vector<Mat>& dog_pyr, int index, const double* offset_x);
void Hessian3D(int x, int y, const vector<Mat>& dog_pyr, int index, double *H);
bool Inverse3D(const double *H, double *H_inve);
void DerivativeOf3D(int x, int y, const vector<Mat>& dog_pyr, int index, double *dx);

void CalculateScale(vector<keypoints>& features, double sigma = SIGMA, int intervals = INTERVALS);
void HalfFeatures(vector<keypoints>& features);

void OrientationAssignment(vector<keypoints>& extrema, vector<keypoints>& features, const vector<Mat>& gauss_pyr);
double* CalculateOrientationHistogram(const Mat& gauss, int x, int y, int bins, int radius, double sigma);
bool CalcGradMagOri(const Mat& gauss, int x, int y, double& mag, double& ori);

void GaussSmoothOriHist(double *hist, int n);
double DominantDirection(double *hist, int n);
void CalcOriFeatures(const keypoints& keypoint, vector<keypoints>& features, const double *hist, int n, double mag_thr);

void CopyKeypoint(const keypoints& src, keypoints& dst);
void DescriptorRepresentation(vector<keypoints>& features, const vector<Mat>& gauss_pyr, int bins, int width);
double*** CalculateDescrHist(const Mat& gauss, int x, int y, double octave_scale, double ori, int bins, int width);
void InterpHistEntry(double ***hist, double xbin, double ybin, double obin, double mag, int bins, int d);
void HistToDescriptor(double ***hist, int width, int bins, keypoints& feature);

void NormalizeDescr(keypoints& feat);
bool FeatureCmp(keypoints& f1, keypoints& f2);

void DrawKeyPoints(Mat &src, vector<keypoints>& features);
void DrawSiftFeatures(Mat &src, vector<keypoints>& features);
void DrawSiftFeature(Mat& src, keypoints& feat, CvScalar color);

void Composite(Mat& splicing_mat, const Mat& src1, const Mat& src2);
vector<matchpoints> Compute_Match(const vector<keypoints>& features1, const vector<keypoints>& features2, float maxloss = MAX_LOSS);