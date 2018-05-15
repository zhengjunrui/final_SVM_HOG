#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <time.h>
#include "Detection.h"
#include"Gamma.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
using namespace std;
using namespace cv;

#define PosSamNO 589  //正样本个数
#define NegSamNO 588 //负样本个数

//HardExample是SVM分类错误的样本，这些样本需要进行二次训练
//HardExample：负样本个数。如果HardExampleNO大于0，表示处理完初始负样本集后，继续处理HardExample负样本集。
//不使用HardExample时必须设置为0，因为特征向量矩阵和特征类别矩阵的维数初始化时用到这个值

#define HardExampleNO 0  

class MySVM : public CvSVM
{
public:
	//获得SVM的决策函数中的alpha数组
	double * get_alpha_vector()
	{
		return this->decision_func->alpha;
	}

	//获得SVM的决策函数中的rho参数,即偏移量
	float get_rho()
	{
		return this->decision_func->rho;
	}
};

void set_SVM(char* A)
{

	//检测窗口(64,128),block尺寸(16,16),block步长(8,8),cell尺寸(8,8),直方图bin个数9
	HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9); 
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	MySVM svm; // SVM分类器
	double start,end,time;

	string ImgName;//图片名(绝对路径)
	//ifstream finPos("E:\\毕业设计\\代码\\标注文本\\INRIADATAPositiveImageList.txt");	//正样本图片的文件名列表
	ifstream finPos("E:\\毕业设计\\代码\\标注文本\\PositiveImageList.txt");	//正样本图片的文件名列表

	//ifstream finNeg("E:\\毕业设计\\代码\\标注文本\\INRIADATANegativeImageList.txt");	//负样本图片的文件名列表
	ifstream finNeg("E:\\毕业设计\\代码\\标注文本\\NegativeImageList.txt");	//负样本图片的文件名列表

	Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数	
	Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人


	//依次读取正样本图片，生成HOG描述子
	start = clock();
	for(int num=0; num<PosSamNO && getline(finPos,ImgName); num++)
	{
		//cout<<"处理："<<ImgName<<endl;
		ImgName = "E:\\毕业设计\\代码\\素材pos帧_resize\\" + ImgName;//加上正样本的路径名
		//ImgName = "E:\\毕业设计\\代码\\video\\INRIADATA\\normalized_images\\train\\pos\\" + ImgName;//加上正样本的路径名
		Mat src = imread(ImgName);//读取图片
		resize(src,src,Size(64,128)); //将正样本图片缩为64*128 

		//RGB直方图均衡化
		/*Mat srcRGB[3];  
		split(src, srcRGB);
		for (int i = 0; i < 3; i++) 
		{
			equalizeHist(srcRGB[i], srcRGB[i]); 
		}
		merge(srcRGB, 3, src); */
		//src = gamma(src);
		//GaussianBlur(src, src, Size(3, 3), 0);
		cvtColor(src, src,CV_RGB2GRAY);	//对图像灰度化
		//normalize(src,src, 1, 0, NORM_MINMAX );
		//Sobel(src,src,-1,1,1,5);
		//Sobel算子进行运算
	    Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;
	    Sobel(src, grad_x, CV_16S, 1, 0, 3, 1, 1,BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);	//将图片转化成为8位图形
	    Sobel(src, grad_y, CV_16S, 0,  1,3, 1, 1, BORDER_DEFAULT);
		convertScaleAbs(grad_y,abs_grad_y);
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, src);
		//medianBlur(src, src, 3);  //中值滤波操作
		//equalizeHist(src, src); 


		vector<float> descriptors;//HOG描述子向量
		hog.compute(src,descriptors,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
		//cout<<"描述子维数："<<descriptors.size()<<endl;

		//处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
		if( 0 == num )
		{
			DescriptorDim = descriptors.size();//HOG描述子的维数
			//初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
			sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
			//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
			sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
		}

		//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
		for(int i=0; i<DescriptorDim; i++)
		{
			sampleFeatureMat.at<float>(num,i) = descriptors[i];//第num个样本的特征向量中的第i个元素
		}
		sampleLabelMat.at<float>(num,0) = 1;//正样本类别为1，有人
	}
	end = clock();
	time = (end-start)/CLOCKS_PER_SEC;
	cout<<"正样本特征提取完毕！"<<endl;
	cout<<"正样本特征提取所需时间为"<<time<<"秒"<<endl;

	//依次读取负样本图片，生成HOG描述子
	start = clock();
	for(int num=0; num<NegSamNO && getline(finNeg,ImgName); num++)
	{
		//cout<<"处理："<<ImgName<<endl;
		ImgName = "E:\\毕业设计\\代码\\素材neg帧_resize\\" + ImgName;//加上负样本的路径名
		//ImgName = "E:\\毕业设计\\代码\\video\\INRIADATA\\normalized_images\\train\\neg\\" + ImgName;//加上负样本的路径名
		Mat src = imread(ImgName);//读取图片
		//normalize(src,src, 1, 0, NORM_MINMAX,  -1 );

		//RGB直方图均衡化
		/*Mat srcRGB[3];  
		split(src, srcRGB);
		for (int i = 0; i < 3; i++) 
		{
			equalizeHist(srcRGB[i], srcRGB[i]); 
		}
		merge(srcRGB, 3, src); */
		//src = gamma(src);
		//GaussianBlur(src, src, Size(3, 3), 0);
		cvtColor(src, src,CV_RGB2GRAY);
		//normalize(src,src, 1, 0, NORM_MINMAX);	//灰度化后再进行归一化，设值的问题？
		//medianBlur(src, src, 3);  //中值滤波操作
		//Sobel(src,src,-1,1,1,5);
		//Sobel算子进行运算
	    Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;
	    Sobel(src, grad_x, CV_16S, 1, 0, 3, 1, 1,BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);	//将图片转化成为8位图形
	    Sobel(src, grad_y, CV_16S, 0,  1,3, 1, 1, BORDER_DEFAULT);
		convertScaleAbs(grad_y,abs_grad_y);
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, src);
		//如果负样本的大小没有达到64*128，就进行扩大（程序里是对64*128的像素块进行hog提取）
		if(src.rows<64||src.cols<128)
		{
			resize(src,src,Size(64,128));	//将样本图片扩大为64*128 
		}

		//灰度直方图均衡化
		//equalizeHist(src, src); 


		vector<float> descriptors;//HOG描述子向量
		hog.compute(src,descriptors,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
		//cout<<"描述子维数："<<descriptors.size()<<endl;

		//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
		for(int i=0; i<DescriptorDim; i++)
		{
			sampleFeatureMat.at<float>(num+PosSamNO,i) = descriptors[i];
		}
		//第PosSamNO+num个样本的特征向量中的第i个元素
		sampleLabelMat.at<float>(num+PosSamNO,0) = -1;//负样本类别为-1，判断无人
		//cout<<"1"<<endl;	
	}
	end = clock();
	time = (end-start)/CLOCKS_PER_SEC;
	cout<<"负样本特征提取完毕！"<<endl;
	cout<<"负样本特征提取所需时间为"<<time<<"秒"<<endl;

	//处理HardExample负样本
	/*if(HardExampleNO > 0)
	{
		ifstream finHardExample("HardExample_2400PosINRIA_12000NegList.txt");//HardExample负样本的文件名列表
		//依次读取HardExample负样本图片，生成HOG描述子
		for(int num=0; num<HardExampleNO && getline(finHardExample,ImgName); num++)
		{
			cout<<"处理："<<ImgName<<endl;
			ImgName = "D:\\DataSet\\HardExample_2400PosINRIA_12000Neg\\" + ImgName;//加上HardExample负样本的路径名
			Mat src = imread(ImgName);//读取图片
			//resize(src,img,Size(64,128));

			vector<float> descriptors;//HOG描述子向量
			hog.compute(src,descriptors,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
			//cout<<"描述子维数："<<descriptors.size()<<endl;

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for(int i=0; i<DescriptorDim; i++)
			{
				sampleFeatureMat.at<float>(num+PosSamNO+NegSamNO,i) = descriptors[i];
			}
			//第PosSamNO+num个样本的特征向量中的第i个元素
			sampleLabelMat.at<int>(num+PosSamNO+NegSamNO,0) = -1;//负样本类别为-1，无人
		}*/
		

	//输出样本的HOG特征向量矩阵到文件
		/*ofstream fout("E:\\毕业设计\\代码\\标注文本\\SampleFeatureMat.txt");
		for(int i=0; i<PosSamNO+NegSamNO; i++)
		{
			fout<<i<<endl;
			for(int j=0; j<DescriptorDim; j++)
			{	fout<<sampleFeatureMat.at<float>(i,j)<<"  ";
	              		
			}
			fout<<endl;
		}*/

		//训练SVM分类器
		//SVM参数：设SVM类型为C_SVC；线性核函数；松弛因子C=0.01,调参方式选择较多人使用的参数
	//CvSVMParams params;	
	//params.svm_type = CvSVM::C_SVC;
	//params.kernel_type = CvSVM::LINEAR;
		//svm->setDegree(0);
		//svm->setGamma(1);
		//svm->setCoef0(0);
		//svm->setNu(0);
		//svm->setP(0);
		//svm->setC(0.01);
	//params.C = 0.01;

		//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
		
	//params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 50000, FLT_EPSILON);
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	
	//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
	CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);

	cout<<"开始训练SVM分类器"<<endl;
	start = clock();
	svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);
	end = clock();
	time = (end-start)/CLOCKS_PER_SEC;
	cout<<"SVM训练完成！"<<endl;
	cout<<"SVM训练所需时间为"<<time<<"秒"<<endl;

	svm.save(A);//将训练好的SVM模型保存为xml文件
	//svm.save("E:\\毕业设计\\代码\\标注文本\\INRIADATA_SVM_HOG.xml");//将训练好的SVM模型保存为xml文件
	
	//system("pause");
}