#include "sift.h"
#include <opencv.hpp>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

/***Sift算法模块
1.src-准备进行特征点检测的原始图片；2.features-用来存储检测出来的关键点；
3.sigma-sigma值；4.intervals-关键点所在的层数
***/
void Sift(const Mat &src, vector<keypoints>& features, double sigma, int intervals)
{
	cout << "[Step_1]:Create -1 octave gaussian pyramid image" << endl;
	Mat init_gray;
	CreateInitSmoothGray(src, init_gray, sigma);
	int octaves = log((double)min(init_gray.rows, init_gray.cols)) / log(2.0) - 2;//计算高斯金字塔的层数
	cout << octaves << endl;
	//return;
	cout << "[1]The height and width of init_gray_img = " << init_gray.rows << "*" << init_gray.cols << endl;
	
	cout << "[Step_2]Building gaussian pyramid..." << endl;
	vector<Mat> gauss_pyr;
	GaussianPyramid(init_gray, gauss_pyr, octaves, intervals, sigma);
	//write_pyr(gauss_pyr, "gausspyramid");

	cout << "[Step_3]Building difference of gaussian pyramid..." << endl;
	vector<Mat> dog_pyr;
	DogPyramid(gauss_pyr, dog_pyr, octaves, intervals);
	//write_pyr(dog_pyr, "dogpyramid");

	cout << "[Step_4]Detecting local extrema..." << endl;
	vector<keypoints> extrema;
	DetectionLocalExtrema(dog_pyr, extrema, octaves, intervals);
	cout << "[3]keypoints count: " << extrema.size() << "--" << endl;
	cout << "[4]extrema detection finished: " << endl;

	cout << "[Step_5]CalculateScale..." << endl;
	CalculateScale(extrema, sigma, intervals);
	HalfFeatures(extrema);

	cout << "[Step_6]Orientation assignment..." << endl;
	OrientationAssignment(extrema, features, gauss_pyr);
	cout << "[6]features count: " << features.size() << endl;

	cout << "[Step_7]DescriptorRepresentation..." << endl;
	DescriptorRepresentation(features, gauss_pyr, DESCR_HIST_BINS, DESCR_WINDOW_WIDTH);
	sort(features.begin(), features.end(), FeatureCmp);
	cout << "finished" << endl;

}

/***创建初始灰度图像函数
用于创建高斯金字塔的-1层图像，初始图像先将原图像灰度化，再扩大一倍后，使用高斯模糊平滑（保留原始图像的信息）
***/
void CreateInitSmoothGray(const Mat& src, Mat& dst, double sigma=SIGMA)
{
	Mat gray;
	Mat up;

	ConvertToGray(src, gray);
	UpSample(gray, up);

	double sigma_init = sqrt(sigma*sigma - (INIT_SIGMA * 2)*(INIT_SIGMA * 2));

	GaussianSmooth(up, dst, sigma_init);
}

/***图像灰度化函数
将彩色图像转为灰度图像
***/
void ConvertToGray(const Mat& src, Mat& dst)
{
	Size size = src.size();
	if (dst.empty())
		dst.create(size, CV_64F);

	uchar* src_data = src.data;
	pixel_t* dst_data = (pixel_t*)dst.data;

	int dst_step = dst.step / sizeof(dst_data[0]);

	for (int j = 0; j < src.cols; j++)
	{
		for (int i = 0; i < src.rows; i++)
		{
			double b = (src_data + src.step*i + src.channels()*j)[0] / 255.0;
			double g = (src_data + src.step*i + src.channels()*j)[1] / 255.0;
			double r = (src_data + src.step*i + src.channels()*j)[2] / 255.0;
			*(dst_data + dst_step * i + dst.channels()*j) = (r + g + b) / 3.0;
		}
	}
}

/***线性插值放大函数
***/
void UpSample(const Mat& src, Mat& dst)
{
	if (src.channels() != 1) return;
	dst.create(src.rows * 2, src.cols * 2, src.type());

	pixel_t* src_data = (pixel_t*)src.data;
	pixel_t* dst_data = (pixel_t*)dst.data;

	int src_step = src.step / sizeof(src_data[0]);
	int dst_step = dst.step / sizeof(dst_data[0]);

	int m = 0, n = 0;
	for (int j = 0; j < src.cols - 1; j++, n += 2)
	{
		m = 0;
		for (int i = 0; i < src.rows - 1; i++, m += 2)
		{
			double sample = *(src_data + src_step * i + src.channels()*j);
			*(dst_data + dst_step * m + dst.channels()*n) = sample;

			double rs = *(src_data + src_step * i + src.channels()*j) + (*(src_data + src_step * (i + 1) + src.channels()*j));
			*(dst_data + dst_step * (m + 1) + dst.channels()*n) = rs / 2;
			double cs = *(src_data + src_step * i + src.channels()*j) + (*(src_data + src_step * i + src.channels()*(j + 1)));
			*(dst_data + dst_step * m + dst.channels()*(n + 1)) = cs / 2;

			double center = (*(src_data + src_step * (i + 1) + src.channels()*j))
				+ *(src_data + src_step * i + src.channels()*j)
				+ *(src_data + src_step * (i + 1) + src.channels()*(j + 1))
				+ (*(src_data + src_step * i + src.channels()*(j + 1)));

			*(dst_data + dst_step * (m + 1) + dst.channels()*(n + 1)) = center / 4;
		}
	}
	if (dst.rows < 3 || dst.cols < 3)
		return;
	//针对最后两行两列
	for (int k = dst.rows - 1; k >= 0; k--)
	{
		*(dst_data + dst_step * k + dst.channels()*(dst.cols - 2)) = *(dst_data + dst_step * k + dst.channels()*(dst.cols - 3));
		*(dst_data + dst_step * k + dst.channels()*(dst.cols - 1)) = *(dst_data + dst_step * k + dst.channels()*(dst.cols - 3));
	}
	for (int k = dst.cols - 1; k >= 0; k--)
	{
		*(dst_data + dst_step * (dst.rows - 2) + dst.channels()*k) = *(dst_data + dst_step * (dst.rows - 3) + dst.channels()*k);
		*(dst_data + dst_step * (dst.rows - 1) + dst.channels()*k) = *(dst_data + dst_step * (dst.rows - 3) + dst.channels()*k);
	}
}

/***高斯平滑函数
***/
void GaussianSmooth(const Mat &src, Mat &dst, double sigma)
{
	GaussianBlur(src, dst, Size(0, 0), sigma, sigma);
}

/***图像高斯金字塔构建函数
***/
void GaussianPyramid(const Mat &src, vector<Mat>&gauss_pyr, int octaves, int intervals, double sigma)
{
	double *sigmas = new double[intervals + 3];
	double k = pow(2.0, 1.0 / intervals);
	sigmas[0] = sigma;

	double sigma_prev;
	double sigma_post;
	
	for (int i = 1; i < intervals + 3; i++)
	{
		sigma_prev = pow(k, i - 1)*sigma;
		sigma_post = sigma_prev * k;
		sigmas[i] = sqrt(sigma_post*sigma_post - sigma_prev * sigma_prev);
	}

	for (int o = 0; o < octaves; o++)
	{
		for (int i = 0; i < intervals + 3; i++)
		{
			Mat mat;
			if (o == 0 && i == 0)
			{
				src.copyTo(mat);
			}
			else if (i == 0)
			{
				DownSample(gauss_pyr[(o - 1)*(intervals + 3) + intervals], mat);
			}
			else
			{
				GaussianSmooth(gauss_pyr[o*(intervals + 3) + i - 1], mat, sigmas[i]);
			}
			gauss_pyr.push_back(mat);
		}
	}
	delete[] sigmas;
}


/***隔点下采样
***/
void DownSample(const Mat& src, Mat& dst)
{
	if (src.channels() != 1)
		return;
	if (src.cols <= 1 || src.rows <= 1)
	{
		src.copyTo(dst);
		return;
	}

	dst.create((int)(src.rows / 2), (int)(src.cols / 2), src.type());

	pixel_t* src_data = (pixel_t*)src.data;
	pixel_t* dst_data = (pixel_t*)dst.data;

	int src_step = src.step / sizeof(src_data[0]);
	int dst_step = dst.step / sizeof(src_data[0]);

	int m = 0, n = 0;
	for (int j = 0; j < src.cols; j += 2, n++)
	{
		m = 0;
		for (int i = 0; i < src.rows; i += 2, m++)
		{
			pixel_t sample = *(src_data + src_step * i + src.channels()*j);
			if (m < dst.rows&&n < dst.cols)
			{
				*(dst_data + dst_step * m + dst.channels()*n) = sample;
			}
		}
	}
}


/***保存金字塔图像
***/
void cv64f_to_cv8U(const Mat& src, Mat& dst)
{
	double* data = (double*)src.data;
	int step = src.step / sizeof(*data);

	if (!dst.empty()) return;
	dst.create(src.size(), CV_8U);

	uchar* dst_data = dst.data;

	for (int i = 0, m = 0; i < src.cols; i++, m++)
	{
		for (int j = 0, n = 0; j < src.rows; j++, n++)
		{
			*(dst_data + dst.step*j + i) = (uchar)(*(data + step * j + i) * 255);
		}
	}
}

void writecv64f(const char* filename, const Mat& mat)
{
	Mat dst;
	cv64f_to_cv8U(mat, dst);
	imwrite(filename, dst);
}

const char* GetFileName(const char* dir, int i)
{
	char *name = new char[50];
	//sprintf(name, "../results/%s/%d.jpg", dir, i);
	return name;
}

void write_pyr(const vector<Mat>& pyr, const char* dir)
{
	for (int i = 0; i < pyr.size(); i++)
	{
		GetFileName(dir, i);
		//writecv64f(GetFileName(dir, i), pyr[i]);
	}
}

/***构造高斯差分金字塔函数
***/
void DogPyramid(const vector<Mat>& gauss_pyr, vector<Mat>& dog_pyr, int octaves, int intervals)
{
	for (int o = 0; o < octaves; o++)
	{
		for (int i = 1; i < intervals + 3; i++)
		{
			Mat mat;
			Sub(gauss_pyr[o*(intervals + 3) + i], gauss_pyr[o*(intervals + 3) + i - 1], mat);
			dog_pyr.push_back(mat);
		}
	}
}

/***图像差分函数
***/
void Sub(const Mat& a, const Mat& b, Mat & c)
{
	if (a.rows != b.rows || a.cols != b.cols || a.type() != b.type())
		return;
	if (!c.empty())
		return;
	c.create(a.size(), a.type());

	pixel_t* ap = (pixel_t*)a.data;
	pixel_t* ap_end = (pixel_t*)a.dataend;
	pixel_t* bp = (pixel_t*)b.data;
	pixel_t* cp = (pixel_t*)c.data;

	while (ap != ap_end)
	{
		*cp++ = *ap++ - *bp++;
	}
}

/***空间极值点的检测（关键点的初步探查）
***/
void DetectionLocalExtrema(const vector<Mat>& dog_pyr, vector<keypoints>& extrema, int octaves, int intervals)
{
	double thresh = 0.5*DXTHRESHOLD / intervals;
	for (int o = 0; o < octaves; o++)
	{
		for (int i = 1; i < (intervals + 2) - 1; i++)
		{
			int index = o * (intervals + 2) + i;
			pixel_t *data = (pixel_t*)dog_pyr[index].data;
			int step = dog_pyr[index].step / sizeof(data[0]);

			for (int y = IMG_BORDER; y < dog_pyr[index].rows - IMG_BORDER; y++)
			{
				for (int x = IMG_BORDER; x < dog_pyr[index].cols - IMG_BORDER; x++)
				{
					pixel_t val = *(data + y * step + x);
					if (fabs(val) > thresh)
					{
						if (isExtremum(x, y, dog_pyr, index))
						{
							keypoints *extrmum = InterploationExtremum(x, y, dog_pyr, index, o, i);
							if (extrmum)
							{
								if (passEdgeResponse(extrmum->x, extrmum->y, dog_pyr, index))
								{
									extrmum->val = *(data + extrmum->y*step + extrmum->x);
									extrema.push_back(*extrmum);
								}
								delete extrmum;
							}
						}
					}
				}
			}

		}
	}
}

/***空间极值点的初步检测
***/
bool isExtremum(int x, int y, const vector<Mat>& dog_pyr, int index)
{
	pixel_t *data = (pixel_t*)dog_pyr[index].data;
	int step = dog_pyr[index].step / sizeof(data[0]);
	pixel_t  val = *(data + y * step + x);

	if (val > 0)
	{
		for (int i = -1; i <= 1; i++)
		{
			int stp = dog_pyr[index + i].step / sizeof(data[0]);
			for (int j = -1; j <= 1; j++)
			{
				for (int k = -1; k <= 1; k++)
				{
					if (val < *((pixel_t*)dog_pyr[index + i].data + stp * (y + j) + (x + k)))
					{
						return false;
					}
				}
			}
		}
	}
	else
	{
		for (int i = -1; i <= 1; i++)
		{
			int stp = dog_pyr[index + i].step / sizeof(pixel_t);
			for (int j = -1; j <= 1; j++)
			{
				for (int k = -1; k <= 1; k++)
				{
					if (val > *((pixel_t*)dog_pyr[index + i].data + stp * (y + j) + (x + k)))
					{
						return false;
					}
				}
			}
		}
	}
	return true;
}

/***修正极值点，删除不稳定的极值点
***/
keypoints * InterploationExtremum(int x, int y, const vector<Mat>& dog_pyr, int index, int octave, int interval, double dxthreshold)
{
	double offset_x[3] = { 0 };
	const Mat &mat = dog_pyr[index];
	int idx = index;
	int intvl = interval;
	int i = 0;

	while (i < MAX_INTERPOLATION_STEPS)
	{
		GetOffsetX(x, y, dog_pyr, idx, offset_x);
		//如果offset_x的任一维度大于0.5，这意味着极值更靠近不同的采样点,就需要用周围的点代替
		if (fabs(offset_x[0]) < 0.5 && fabs(offset_x[1]) < 0.5 && fabs(offset_x[2]) < 0.5)
			break;
		x += cvRound(offset_x[0]);
		y += cvRound(offset_x[1]);
		interval += cvRound(offset_x[2]);

		idx = index - intvl + interval;
		if (interval<1 || interval>INTERVALS || x >= mat.cols - 1 || x < 2 || y >= mat.rows - 1 || y < 2)
		{
			return NULL;
		}
		i++;
	}
	if (i >= MAX_INTERPOLATION_STEPS)
		return NULL;
	//拒绝不稳定的极值点
	if (GetFabsDx(x, y, dog_pyr, idx, offset_x) < dxthreshold / INTERVALS)
		return NULL;

	keypoints *keypoint = new keypoints;
	keypoint->x = x;
	keypoint->y = y;
	keypoint->offset_x = offset_x[0];
	keypoint->offset_y = offset_x[1];
	keypoint->interval = interval;
	keypoint->offset_interval = offset_x[2];
	keypoint->octave = octave;
	keypoint->dx = (x + offset_x[0])*pow(2.0, octave);
	keypoint->dy = (y + offset_x[1])*pow(2.0, octave);

	return keypoint;
}

/***计算x^
***/
void GetOffsetX(int x, int y, const vector<Mat>& dog_pyr, int index, double *offset_x)
{
	double H[9], H_inve[9] = { 0 };
	Hessian3D(x, y, dog_pyr, index, H);
	Inverse3D(H, H_inve);
	double dx[3];
	DerivativeOf3D(x, y, dog_pyr, index, dx);
	
	for (int i = 0; i < 3; i++)
	{
		offset_x[i] = 0.0;
		for (int j = 0; j < 3; j++)
		{
			offset_x[i] += H_inve[i * 3 + j] * dx[j];
		}
		offset_x[i] = -offset_x[i];
	}
}

/***3维D(x)求二阶偏导，即求Hessian矩阵
***/
#define At(index, x, y) (PyrAt(dog_pyr, (index), (x), (y)))
#define Hat(i,j) (*(H+i*3+j))
double PyrAt(const vector<Mat>& pyr, int index, int x, int y)
{
	pixel_t *data = (pixel_t*)pyr[index].data;
	int step = pyr[index].step / sizeof(data[0]);
	pixel_t val = *(data + y * step + x);

	return val;
}
void Hessian3D(int x, int y, const vector<Mat>& dog_pyr, int index, double *H)
{
	double val, Dxx, Dyy, Dss, Dxy, Dxs, Dys;
	
	val = At(index, x, y);

	Dxx = At(index, x + 1, y) + At(index, x - 1, y) - 2 * val;
	Dyy = At(index, x, y + 1) + At(index, x, y - 1) - 2 * val;
	Dss = At(index + 1, x, y) + At(index - 1, x, y) - 2 * val;

	Dxy = (At(index, x + 1, y + 1) + At(index, x - 1, y - 1) - At(index, x + 1, y - 1) - At(index, x - 1, y + 1)) / 4.0;
	Dxs = (At(index + 1, x + 1, y) + At(index - 1, x - 1, y) - At(index - 1, x + 1, y) - At(index + 1, x - 1, y)) / 4.0;
	Dys = (At(index + 1, x, y + 1) + At(index - 1, x, y - 1) - At(index + 1, x, y - 1) - At(index - 1, x, y + 1)) / 4.0;

	Hat(0, 0) = Dxx;
	Hat(1, 1) = Dyy;
	Hat(2, 2) = Dss;

	Hat(1, 0) = Hat(0, 1) = Dxy;
	Hat(2, 0) = Hat(0, 2) = Dxs;
	Hat(1, 2) = Hat(2, 1) = Dys;
}

/***3*3阶矩阵求逆
***/
#define HIat(i, j) (*(H_inve+(i)*3 + (j)))
bool Inverse3D(const double *H, double *H_inve)
{
	double A = Hat(0, 0)*Hat(1, 1)*Hat(2, 2) + Hat(0, 1)*Hat(1, 2)*Hat(2, 0) + Hat(0, 2)*Hat(1, 0)*Hat(2, 1)
		- Hat(0, 0)*Hat(1, 2)*Hat(2, 1) - Hat(0, 1)*Hat(1, 0)*Hat(2, 2) - Hat(0, 2)*Hat(1, 1)*Hat(2, 0);
	if (fabs(A) < 1e-10)return false;

	HIat(0, 0) = Hat(1, 1)*Hat(2, 2) - Hat(2, 1)*Hat(1, 2);
	HIat(0, 1) = -(Hat(0, 1)*Hat(2, 2) - Hat(2, 1)*Hat(0, 2));
	HIat(0, 2) = Hat(0, 1)*Hat(1, 2) - Hat(0, 2)*Hat(1, 1);

	HIat(1, 0) = Hat(1, 2)*Hat(2, 0) - Hat(2, 2)*Hat(1, 0);
	HIat(1, 1) = -(Hat(0, 2)*Hat(2, 0) - Hat(0, 0)*Hat(2, 2));
	HIat(1, 2) = Hat(0, 2)*Hat(1, 0) - Hat(0, 0)*Hat(1, 2);

	HIat(2, 0) = Hat(1, 0)*Hat(2, 1) - Hat(1, 1)*Hat(2, 0);
	HIat(2, 1) = -(Hat(0, 0)*Hat(2, 1) - Hat(0, 1)*Hat(2, 0));
	HIat(2, 2) = Hat(0, 0)*Hat(1, 1) - Hat(0, 1)*Hat(1, 0);

	for (int i = 0; i < 9; i++)
	{
		*(H_inve + i) /= A;
	}
	return true;
}

/***3维D(x)一阶偏导
***/
void DerivativeOf3D(int x, int y, const vector<Mat>& dog_pyr, int index, double *dx)
{
	double Dx = (At(index, x + 1, y) - At(index, x - 1, y)) / 2.0;
	double Dy = (At(index, x, y + 1) - At(index, x, y - 1)) / 2.0;
	double Ds = (At(index + 1, x, y) - At(index - 1, x, y)) / 2.0;

	dx[0] = Dx;
	dx[1] = Dy;
	dx[2] = Ds;
}

/***计算|D(x^)|
***/
double GetFabsDx(int x, int y, const vector<Mat>& dog_pyr, int index, const double* offset_x)
{
	//|D(x^)|=D + 0.5 * dx * offset_x; dx=(Dx, Dy, Ds)^T
	double dx[3];
	DerivativeOf3D(x, y, dog_pyr, index, dx);

	double term = 0.0;
	for (int i = 0; i < 3; i++)
		term += dx[i] * offset_x[i];

	pixel_t *data = (pixel_t*)dog_pyr[index].data;
	int step = dog_pyr[index].step / sizeof(data[0]);
	pixel_t val = *(data + y * step + x);

	return fabs(val + 0.5*term);
}

/***消除边缘响应点-(Tr(H)^2/Det(H))<((r+1)^2/r)? 是保留，否则剔除
***/
#define DAt(x, y) (*(data+(y)*step+(x)))
bool passEdgeResponse(int x, int y, const vector<Mat>& dog_pyr, int index, double r)
{
	pixel_t *data = (pixel_t *)dog_pyr[index].data;
	int step = dog_pyr[index].step / sizeof(data[0]);
	pixel_t val = *(data + y * step + x);

	double Dxx;
	double Dyy;
	double Dxy;
	double Tr_h;
	double Det_h;
			
	Dxx = DAt(x + 1, y) + DAt(x - 1, y) - 2 * val;
	Dyy = DAt(x, y + 1) + DAt(x, y - 1) - 2 * val;
	Dxy = (DAt(x + 1, y + 1) + DAt(x - 1, y - 1) - DAt(x - 1, y + 1) - DAt(x + 1, y - 1)) / 4.0;
	Tr_h = Dxx + Dyy;
	Det_h = Dxx * Dyy - Dxy * Dxy;

	if (Det_h <= 0)return false;
	if (Tr_h*Tr_h / Det_h < ((r + 1)*(r + 1) / r)) return true;
	return false;
}

/***尺度计算函数
***/
void CalculateScale(vector<keypoints>& features, double sigma, int intervals)
{
	double intvl = 0;
	for (int i = 0; i < features.size(); i++)
	{
		intvl = features[i].interval + features[i].offset_interval;
		features[i].scale = sigma * pow(2.0, features[i].octave + intvl / intervals);
		features[i].octave_scale = sigma * pow(2.0, intvl / intervals);
	}
}
/***对扩大后的图像特征进行缩放
***/
void HalfFeatures(vector<keypoints>& features)
{
	for (int i = 0; i < features.size(); i++)
	{
		features[i].dx /= 2;
		features[i].dy /= 2;
		features[i].scale /= 2;
	}
}

/***关键点方向匹配
***/
void OrientationAssignment(vector<keypoints>& extrema, vector<keypoints>& features, const vector<Mat>& gauss_pyr)
{
	int n = extrema.size();
	double *hist;
	
	for (int i = 0; i < n; i++)
	{
		hist = CalculateOrientationHistogram(gauss_pyr[extrema[i].octave*(INTERVALS + 3) + extrema[i].interval],
			extrema[i].x, extrema[i].y, ORI_HIST_BINS, cvRound(ORI_WINDOW_RADIUS*extrema[i].octave_scale),
			ORI_SIGMA_TIMES*extrema[i].octave_scale);//计算梯度的方向直方图

		for (int j = 0; j < ORI_SMOOTH_TIMES; j++)
			GaussSmoothOriHist(hist, ORI_HIST_BINS);//对方向直方图进行高斯平滑
		double highest_peak = DominantDirection(hist, ORI_HIST_BINS);//计算方向直方图中的峰值

		CalcOriFeatures(extrema[i], features, hist, ORI_HIST_BINS, highest_peak*ORI_PEAK_RATIO);

		delete[] hist;
	}
}
/***计算梯度的方向直方图
***/
double* CalculateOrientationHistogram(const Mat& gauss, int x, int y, int bins, int radius, double sigma)
{
	double* hist = new double[bins];
	for (int i = 0; i < bins; i++)
		*(hist + i) = 0.0;

	double mag;
	double ori;
	double weight;

	int bin;
	const double PI2 = 2.0*CV_PI;
	double econs = -1.0 / (2.0*sigma*sigma);

	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			if (CalcGradMagOri(gauss, x + i, y + j, mag, ori))
			{
				weight = exp((i*i + j * j)*econs);
				bin = cvRound(bins*(CV_PI - ori) / PI2);
				bin = bin < bins ? bin : 0;

				hist[bin] += mag * weight;
			}
		}
	}
	return hist;
}

/***计算关键点的梯度和梯度方向
***/
bool CalcGradMagOri(const Mat& gauss, int x, int y, double& mag, double& ori)
{
	if (x > 0 && x < (gauss.cols - 1) && y>0 && y < (gauss.rows - 1))
	{
		pixel_t *data = (pixel_t*)gauss.data;
		int  step = gauss.step / sizeof(*data);
		double dx = *(data + step * y + (x + 1)) - (*(data + step * y + (x - 1)));
		double dy = *(data + step * (y + 1) + x) - (*(data + step * (y - 1) + x));

		mag = sqrt(dx*dx + dy * dy);
		ori = atan2(dy, dx);
		return true;
	}
	else
		return false;
}

/***对梯度方向直方图进行连续两次的高斯平滑
***/
void GaussSmoothOriHist(double *hist, int n)
{
	double prev = hist[n - 1];
	double temp;
	double h0 = hist[0];

	for (int i = 0; i < n; i++)
	{
		temp = hist[i];
		hist[i] = 0.25*prev + 0.5*hist[i] + 0.25*((i + 1) >= n ? h0 : hist[i + 1]);
		prev = temp;
	}
}

/***检测主方向
***/
double DominantDirection(double *hist, int n)
{
	double maxd = hist[0];
	for (int i = 1; i < n; i++)
	{
		if (hist[i] > maxd)
			maxd = hist[i];
	}
	return maxd;
}

/***计算更加精确的关键点主方向--抛物插值
***/
#define Parabola_Interpolate(l,c,r) (0.5*((l)-(r))/((l)-2.0*(c)+(r)))
void CalcOriFeatures(const keypoints& keypoint, vector<keypoints>& features, const double *hist, int n, double mag_thr)
{
	double bin;
	double PI2 = CV_PI * 2.0;
	int l;
	int r;
	for (int i = 0; i < n; i++)
	{
		l = (i == 0) ? (n - 1) : (i - 1);
		r = (i + 1) % n;

		if (hist[i] > hist[l] && hist[i] > hist[r] && hist[i] >= mag_thr)
		{
			bin = i + Parabola_Interpolate(hist[l], hist[i], hist[r]);
			bin = (bin < 0) ? (bin + n) : (bin >= n ? (bin - n) : bin);

			keypoints new_key;
			CopyKeypoint(keypoint, new_key);

			new_key.ori = ((PI2*bin) / n) - CV_PI;
			features.push_back(new_key);
		}
	}
}

/***复制关键点
***/
void CopyKeypoint(const keypoints& src, keypoints& dst)
{
	dst.dx = src.dx;
	dst.dy = src.dy;
	dst.interval = src.interval;
	dst.octave = src.octave;
	dst.octave_scale = src.octave_scale;
	dst.offset_interval = src.offset_interval;
	dst.offset_x = src.offset_x;
	dst.offset_y = src.offset_y;
	dst.ori = src.ori;
	dst.scale = src.scale;
	dst.x = src.x;
	dst.y = src.y;
}

/***关键点描述符建立
***/
void DescriptorRepresentation(vector<keypoints>& features, const vector<Mat>& gauss_pyr, int bins, int width)
{
	double ***hist;
	for (int i = 0; i < features.size(); i++)
	{
		hist = CalculateDescrHist(gauss_pyr[features[i].octave*(INTERVALS + 3) + features[i].interval],//计算描述子的直方图
			features[i].x, features[i].y, features[i].octave_scale, features[i].ori, bins, width);
		HistToDescriptor(hist, width, bins, features[i]);

		for (int j = 0; j < width; j++)
		{
			for (int k = 0; k < width; k++)
			{
				delete[] hist[j][k];
			}
			delete[] hist[j];
		}
		delete[] hist;
	}
}

/***计算描述子的直方图
***/
double*** CalculateDescrHist(const Mat& gauss, int x, int y, double octave_scale, double ori, int bins, int width)
{
	double ***hist = new double**[width];
	for (int i = 0; i < width; i++)
	{
		hist[i] = new double*[width];
		for (int j = 0; j < width; j++)
		{
			hist[i][j] = new double[bins];
		}
	}

	for (int r = 0; r < width; r++)
		for (int c = 0; c < width; c++)
			for (int o = 0; o < bins; o++)
				hist[r][c][o] = 0.0;

	double cos_ori = cos(ori);
	double sin_ori = sin(ori);

	double sigma = 0.5*width;
	double conste = -1.0 / (2 * sigma*sigma);
	double PI2 = CV_PI * 2;
	double sub_hist_width = DESCR_SCALE_ADJUST * octave_scale;

	int radius = (sub_hist_width*sqrt(2.0)*(width + 1)) / 2.0 + 0.5;
	double grad_ori;
	double grad_mag;

	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			double rot_x = (cos_ori*j - sin_ori * i) / sub_hist_width;
			double rot_y = (sin_ori*j + cos_ori * i) / sub_hist_width;
			double xbin = rot_x + width / 2 - 0.5;
			double ybin = rot_y + width / 2 - 0.5;

			if (xbin > -1.0 && xbin < width && ybin > -1.0 && ybin < width)
			{
				if (CalcGradMagOri(gauss, x + j, y + i, grad_mag, grad_ori))
				{
					grad_ori = (CV_PI - grad_ori) - ori;
					while (grad_ori < 0.0)
						grad_ori += PI2;
					while (grad_ori >= PI2)
						grad_ori -= PI2;

					double obin = grad_ori * (bins / PI2);
					double weight = exp(conste*(rot_x*rot_x + rot_y * rot_y));

					InterpHistEntry(hist, xbin, ybin, obin, grad_mag*weight, bins, width);

				}
			}
		}
	}
	return hist;
}

/***对直方图做插值
***/
void InterpHistEntry(double ***hist, double xbin, double ybin, double obin, double mag, int bins, int d)
{
	double d_r, d_c, d_o, v_r, v_c, v_o;
	double** row, *h;
	int r0, c0, o0, rb, cb, ob, r, c, o;

	r0 = cvFloor(ybin);
	c0 = cvFloor(xbin);
	o0 = cvFloor(obin);
	d_r = ybin - r0;
	d_c = xbin - c0;
	d_o = obin - o0;

	for (r = 0; r <= 1; r++)
	{
		rb = r0 + r;
		if (rb >= 0 && rb < d)
		{
			v_r = mag * ((r == 0) ? (1.0 - d_r) : d_r);
			row = hist[rb];
			for (c = 0; c <= 1; c++)
			{
				cb = c0 + c;
				if (cb >= 0 && cb < d)
				{
					v_c = v_r * ((c == 0) ? (1.0 - d_c) : d_c);
					h = row[cb];
					for (o = 0; o <= 1; o++)
					{
						ob = (o0 + o) % bins;
						v_o = v_c * ((o == 0) ? (1 - d_o) : d_o);
						h[ob] += v_o;
					}
				}
			}
		}
	}
}

/***直方图到描述子的转换
***/
void HistToDescriptor(double ***hist, int width, int bins, keypoints& feature)
{
	int int_val, i, r, c, o, k = 0;
	for (r = 0; r < width; r++)
		for (c = 0; c < width; c++)
			for (o = 0; o < bins; o++)
				feature.descriptor[k++] = hist[r][c][o];

	feature.descr_length = k;
	NormalizeDescr(feature);
	for (i = 0; i < k; i++)
		if (feature.descriptor[i] > DESCR_MAG_THR)
			feature.descriptor[i] = DESCR_MAG_THR;
	NormalizeDescr(feature);
	for (i = 0; i < k; i++)
	{
		int_val = INT_DESCR_FCTR * feature.descriptor[i];
		feature.descriptor[i] = min(255, int_val);
	}
}

/***特征向量归一化
***/
void NormalizeDescr(keypoints& feat)
{
	double cur, len_inv, len_sq = 0.0;
	int i, d = feat.descr_length;
	
	for (i = 0; i < d; i++)
	{
		cur = feat.descriptor[i];
		len_sq += cur * cur;
	}
	len_inv = 1.0 / sqrt(len_sq);
	for (i = 0; i < d; i++)
		feat.descriptor[i] *= len_inv;
}

/***特征尺度比较
***/
bool FeatureCmp(keypoints& f1, keypoints& f2)
{
	return f1.scale < f2.scale;
}

/***关键点keypoints绘制
***/
void DrawKeyPoints(Mat &src, vector<keypoints>& features)
{
	int  j = 0;
	for (int i = 0; i < features.size(); i++)
	{
		CvScalar color = { 255, 0, 0};
		circle(src, Point(features[i].dx, features[i].dy), 3, color);
		j++;
	}
}

/***SIFT特征点绘制
***/
void DrawSiftFeatures(Mat &src, vector<keypoints>& features)
{
	CvScalar color = CV_RGB(0, 255, 0);
	for (int i = 0; i < features.size(); i++)
	{
		DrawSiftFeature(src, features[i], color);
	}
}

/***SIFT特征点绘制的具体函数
***/
void DrawSiftFeature(Mat& src, keypoints& feat, CvScalar color)
{
	int len, hlen, blen, start_x, start_y, end_x, end_y, h1_x, h1_y, h2_x, h2_y;
	double scl, ori;
	double scale = 5.0;
	double hscale = 0.75;
	CvPoint start, end, h1, h2;

	start_x = cvRound(feat.dx);
	start_y = cvRound(feat.dy);
	scl = feat.scale;
	ori = feat.ori;
	len = cvRound(scl*scale);
	hlen = cvRound(scl*hscale);
	blen = len - hlen;
	end_x = cvRound(len*cos(ori)) + start_x;
	end_y = cvRound(len*-sin(ori)) + start_y;
	h1_x = cvRound(blen*cos(ori + CV_PI / 18.0)) + start_x;
	h1_y = cvRound(blen*-sin(ori + CV_PI / 18.0)) + start_y;
	h2_x = cvRound(blen*cos(ori - CV_PI / 18.0)) + start_x;
	h2_y = cvRound(blen*-sin(ori - CV_PI / 18.0)) + start_y;

	start = cvPoint(start_x, start_y);
	end = cvPoint(end_x, end_y);
	h1 = cvPoint(h1_x, h1_y);
	h2 = cvPoint(h2_x, h2_y);

	line(src, start, end, color, 1, 8, 0);
	line(src, end, h1, color, 1, 8, 0);
	line(src, end, h2, color, 1, 8, 0);
}

/***为拼接后的图片赋值
***/
void Composite(Mat& splicing_mat, const Mat& src1, const Mat& src2)
{
	int row1 = src1.rows, col1 = src1.cols, row2 = src2.rows, col2 = src2.cols;
	int row = max(row1, row2), col = col1 + col2 + 100;
	splicing_mat = Mat(row, col, CV_8UC3);
	for (int r = 0; r < row1; r++)
	{
		for (int c = 0; c < col1; c++)
		{
			splicing_mat.ptr<uchar>(r)[c * 3 + 0] = src1.ptr<uchar>(r)[c * 3 + 0];
			splicing_mat.ptr<uchar>(r)[c * 3 + 1] = src1.ptr<uchar>(r)[c * 3 + 1];
			splicing_mat.ptr<uchar>(r)[c * 3 + 2] = src1.ptr<uchar>(r)[c * 3 + 2];
		}
	}

	for (int r = 0; r < row2; r++)
	{
		for (int c = 0; c < col2; c++)
		{
			splicing_mat.ptr<uchar>(r)[3 * col1 + 300 + c * 3 + 0] = src2.ptr<uchar>(r)[c * 3 + 0];
			splicing_mat.ptr<uchar>(r)[3 * col1 + 300 + c * 3 + 1] = src2.ptr<uchar>(r)[c * 3 + 1];
			splicing_mat.ptr<uchar>(r)[3 * col1 + 300 + c * 3 + 2] = src2.ptr<uchar>(r)[c * 3 + 2];
		}
	}
}

/***计算匹配点
***/
vector<matchpoints> Compute_Match(const vector<keypoints>& features1, const vector<keypoints>& features2, float maxloss)
{
	double rate_thres = 0.5;
	int i = 0, j = 0;
	vector<matchpoints> mac;
	vector<keypoints> f1 = features1, f2 = features2;
	while (i<features1.size())
	{
		j = 0;
		Point2i m1(0, 0), m2(0, 0);
		double loss1 = 1000000, loss2 = 100000000;
		while (j<features2.size())
		{
			double loss = 0;
			double *descriptor1 = f1[i].descriptor;
			double *descriptor2 = f2[j].descriptor;
			
			for (int k = 0; k < 128; k++)
			{
				loss += pow(*descriptor1++ - *descriptor2++, 2);
			}
			loss = sqrt(loss);
			if (loss <= loss1)
			{
				loss2 = loss1;
				loss1 = loss;
				m2 = m1;
				m1 = Point2i(features2[j].x, features2[j].y);
			}
			else if (loss < loss2)
			{
				loss2 = loss;
				m2 = Point2i(features2[j].x, features2[j].y);
			}
			j++;
		}
		cout << "loss1: " << loss1 << " loss2: " << loss2 << endl;
		if (loss1 < maxloss)
		{
			if (loss1 < (rate_thres*loss2))
			{
				cout << "f1_x: " << features1[i].x << " f1_y: " << features2[i].y << endl;
				mac.push_back(matchpoints(Point2i(features1[i].x, features1[i].y), m1));
			}
		}
		i++;
	}
	return mac;
}
