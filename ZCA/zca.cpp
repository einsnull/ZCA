#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
using namespace std;


#pragma comment(lib,"opencv_core243.lib")
#pragma comment(lib,"opencv_highgui243.lib")
#pragma comment(lib,"opencv_imgproc243.lib")

#include <Eigen/Dense>
using namespace Eigen;

//scalar reciprocal
double reciprocalScalar(double x)
{
	return 1.0 / x;
}

//scalar sqrt
double sqrtScalar(double x)
{
	return sqrt(x);
}

class CA
{
public:
	MatrixXd PCA(MatrixXd &x,double ratio);
	MatrixXd PCAWhite(MatrixXd &x);
	MatrixXd ZCAWhite(MatrixXd &x);
private:
	MatrixXd reciprocal(MatrixXd &z);
	MatrixXd sqrt(MatrixXd &z);
};

MatrixXd CA::reciprocal(MatrixXd &z)
{
	return z.unaryExpr(ptr_fun(reciprocalScalar));
}

MatrixXd CA::sqrt(MatrixXd &z)
{
	return z.unaryExpr(ptr_fun(sqrtScalar));
}

MatrixXd CA::PCA(MatrixXd &x,double ratio)
{
	if(ratio <= 0 || ratio > 1)
	{
		return MatrixXd::Zero(x.rows(),x.cols());
	}
	MatrixXd avg = x.colwise().mean();
	x = x - avg.replicate(x.rows(),1);
	MatrixXd sigma = x * x.transpose() * (1.0 / x.cols());
	JacobiSVD<MatrixXd> svd(sigma, ComputeFullU | ComputeFullV);
	MatrixXd U = svd.matrixU();
	MatrixXd V = svd.matrixV();
	MatrixXd S = svd.singularValues();
	MatrixXd xRot = U.transpose() * x;
	MatrixXd cov = xRot * xRot.transpose() * (1.0 / xRot.cols());
	int k = 0;
	double p = 0;
	double curVar = 0;
	double sumVar = S.array().sum();
	for(k = 0;k < S.rows() && p < ratio;k++)
	{
		k = k + 1;
		curVar += S(k,0);
		p = curVar / sumVar;
	}
	xRot.bottomRows(S.rows() - k) = MatrixXd::Zero(S.rows() - k,xRot.cols());
	MatrixXd xHat = U * xRot;
	return xHat;
}

MatrixXd CA::PCAWhite(MatrixXd &x)
{
	
	MatrixXd avg = x.colwise().mean();
	x = x - (avg.replicate(x.rows(),1));
	MatrixXd sigma = x * x.transpose() * (1.0 / x.cols());
	JacobiSVD<MatrixXd> svd(sigma, ComputeFullU | ComputeFullV);
	MatrixXd U = svd.matrixU();
	MatrixXd V = svd.matrixV();
	MatrixXd S = svd.singularValues();
	double epsilon = 0.1;
	MatrixXd term1 = S + MatrixXd::Ones(S.rows(),S.cols()) * epsilon;
	MatrixXd term2 = reciprocal(sqrt(term1));
	MatrixXd xPCAWhite = term2.asDiagonal() * U.transpose() * x;
	return xPCAWhite;
}

MatrixXd CA::ZCAWhite(MatrixXd &x)
{
	MatrixXd avg = x.colwise().mean();
	x = x - (avg.replicate(x.rows(),1));
	MatrixXd sigma = x * x.transpose() * (1.0 / x.cols());
	JacobiSVD<MatrixXd> svd(sigma, ComputeFullU | ComputeFullV);
	MatrixXd U = svd.matrixU();
	MatrixXd V = svd.matrixV();
	MatrixXd S = svd.singularValues();
	double epsilon = 0.1;
	MatrixXd term1 = S + MatrixXd::Ones(S.rows(),S.cols()) * epsilon;
	MatrixXd term2 = reciprocal(sqrt(term1));
	MatrixXd xPCAWhite = term2.asDiagonal() * U.transpose() * x;
	return U * xPCAWhite;
}


bool loadDataSet(MatrixXd &data);

void buildImage(MatrixXd &theta,int imgWidth,char *szFileName,bool showFlag = false,double ratio = 1);

int main()
{
	MatrixXd x(1,1);
	loadDataSet(x);
	cout << x.rows() << " " << x.cols() << endl;
	MatrixXd showData = x.topRows(200);
	buildImage(showData,12,"data.jpg",false);
	MatrixXd data = x.transpose();
	CA ca;
	MatrixXd xZCAWhite = ca.ZCAWhite(data);
	showData = xZCAWhite.transpose().topRows(200);
	buildImage(showData,12,"zcaWhiteData.jpg",false);
	/*MatrixXd xPCAWhite = ca.PCAWhite(data);
	showData = xPCAWhite.transpose().topRows(200);
	buildImage(showData,12,"pcaWhiteData.jpg",false);*/
	MatrixXd pca = ca.PCA(data,0.99);
	showData = pca.transpose().topRows(200);
	buildImage(showData,12,"pcaData.jpg",false);
	system("pause");
	return 0;
}

bool loadDataSet(MatrixXd &data)
{
	ifstream ifs("rawImages.txt");
	if(!ifs)
	{
		return false;
	}
	cout << "Loading data..." << endl;
	int inputLayerSize;
	ifs >> inputLayerSize;
	int dataSetSize;
	ifs >> dataSetSize;
	data.resize(dataSetSize,inputLayerSize);
	for(int i = 0; i < dataSetSize; i++)
	{
		for(int j = 0; j < inputLayerSize; j++)
		{
			ifs >> data(i,j);
		}
	}
	//data = data - MatrixXd::Ones(data.rows(),data.cols())*0.5;
	ifs.close();
	return true;
}


void buildImage(MatrixXd &theta,int imgWidth,char* szFileName,bool showFlag,double ratio)
{
	int margin = 1;
	int rows = theta.rows();
	int cols = theta.cols();
	if(rows <= 0 || cols <= 0)
	{
		return ;
	}
	/*cout << rows << endl;
	cout << cols << endl;*/
	double pr = sqrt((double)rows);
	int perRow = (int)pr + (pr - (int)pr > 0);
	double tc = (double)rows / (double)perRow;
	int tCols = (int)tc + (tc - (int)tc > 0);
	/*cout << "perRow: " << perRow << endl;
	cout << "tCols: " << tCols << endl;*/
	MatrixXd max = theta.rowwise().maxCoeff();
	MatrixXd min = theta.rowwise().minCoeff();

	int imgHeight = cols/imgWidth;
	IplImage* iplImage = cvCreateImage(
		cvSize(imgWidth * perRow + margin * (perRow+1),imgHeight * tCols + margin * (tCols + 1)),
		IPL_DEPTH_8U,1);
	
	int step = iplImage->widthStep;
	uchar *data = (uchar *)iplImage->imageData;
	int h = iplImage->height;
	int w = iplImage->width;
	for(int x = 0; x < w; x++)
	{
		for(int y = 0; y < h; y++)
		{
			data[y * step + x] = 0;
		}
	}

	for(int i = 0;i < rows;i++)
	{
		
		int n = 0;
		int hIdx = i / perRow;
		int wIdx = i % perRow;

		for(int j = 0;j < imgHeight;j++)
		{
			for(int k = 0;k < imgWidth; k++)
			{
				double per = (theta(i,n) - min(i,0) ) / (max(i,0) - min(i,0));
				//data[j * step + k] = 255;
				int val = (hIdx * imgHeight+j + margin * (hIdx+1)) * step
					+ (wIdx * imgWidth + k + margin * (wIdx+1));
				if(val > step * (imgHeight * tCols + margin * (tCols + 1)))
				{
					cout << "error" << endl;
				}
				data[val] = (uchar)(int)(per * 230.0);
				n ++;
			}
		}
		
	}
	
	cvSaveImage(szFileName,iplImage);

	if(showFlag)
	{
		cvNamedWindow(szFileName,CV_WINDOW_AUTOSIZE);
		IplImage* iplImageShow = cvCreateImage(
			cvSize((int)(iplImage->width * ratio),(int)(iplImage->height * ratio)),IPL_DEPTH_8U,1);
		cvResize(iplImage,iplImageShow,CV_INTER_CUBIC);
		cvShowImage(szFileName,iplImageShow);
		cvWaitKey(100000);
		cvDestroyWindow(szFileName);
		cvReleaseImage(&iplImageShow);
	}
	cvReleaseImage(&iplImage);
}