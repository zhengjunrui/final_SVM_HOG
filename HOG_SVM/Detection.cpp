#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
using namespace std;
using namespace cv;

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

void Detecting()
{
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	MySVM svm;//SVM分类器	
	
	svm.load("E:\\毕业设计\\代码\\标注文本\\SVM_HOG_20Pos_60Neg.xml");//从XML文件读取训练好的SVM模型
	//svm.load("E:\\毕业设计\\代码\\标注文本\\SVM_HOG.xml");//从XML文件读取训练好的SVM模型

	DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数
	int supportVectorNum = svm.get_support_vector_count();//支持向量的个数
	cout<<"支持向量个数："<<supportVectorNum<<endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数,一维矩阵，放的是决策函数的alpha向量,svm不能定义double
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);	//支持向量矩阵，放的是各个支持向量的特征向量
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_64FC1);//alpha向量乘以支持向量矩阵的结果,一维矩阵

	//将支持向量的数据复制到supportVectorMat矩阵中
	for(int i=0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);	//返回第i个支持向量的数据指针
		for(int j=0; j<DescriptorDim; j++)
		{
			supportVectorMat.at<float>(i,j) = pSVData[j];			
		}
	}

	//将alpha向量的数据复制到alphaMat中
	double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量
	for(int i=0; i<supportVectorNum; i++)
	{
		//设定条件，防止因double转换成float的时候数据溢出
		if(pAlphaData[i]<1e-30)
		{
			pAlphaData[i]=1e-30;
		}
		if(pAlphaData[i]>1e30)
		{
			pAlphaData[i]=1e30;
		}
		alphaMat.at<float>(0,i) = pAlphaData[i];
	}

	//计算-(alphaMat * supportVectorMat),结果放到resultMat中，加负号是因为HOG进行提取的矩阵和计算出来的矩阵结果是相反的（网上的解释）
	resultMat = -1.0 * alphaMat *  supportVectorMat;		
	//cout<<"resultMat:"<<resultMat<<endl;

	//得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
	vector<float> myDetector;

	//将resultMat中的数据复制到数组myDetector第一行中
	for(int i=0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0,i));
	}
	//最后添加偏移量rho，得到检测子
	myDetector.push_back(svm.get_rho());
	cout<<"检测子维数："<<myDetector.size()<<endl;	
	//设置HOGDescriptor的检测子
	HOGDescriptor myHOG;
	myHOG.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//保存检测子参数到文件
	ofstream fout("E:\\毕业设计\\代码\\标注文本\\HOGDetectorForOpenCV.txt");
	for(int i=0; i<myDetector.size(); i++)
	{
		fout<<myDetector[i]<<endl;
	}


	/**************读入图片进行HOG行人检测******************/
	Mat src = imread("E:\\毕业设计\\代码\\pos帧待截图\\19.png");	//图片要小
	vector<Rect> found, found_filtered;//矩形框数组
	cout<<"进行多尺度HOG人体检测"<<endl;
	double start,end,time;
	start = clock();    
	//cvtColor(src, src,CV_RGB2GRAY);
	myHOG.detectMultiScale(src, found, 0, Size(8,8), Size(32,32), 1.05, 1);//对图片进行多尺度行人检测
	cout<<"找到的矩形框个数："<<found.size()<<endl;

	//找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中
	for(int i=0; i < found.size(); i++)
	{
		Rect r = found[i];
		int j=0;
		for(; j < found.size(); j++)
			if(j != i && (r & found[j]) == r)
				break;
		if( j == found.size())
			found_filtered.push_back(r);
	}

	//画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
	for(int i=0; i<found_filtered.size(); i++)
	{
		Rect r = found_filtered[i];
		/*r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);*/
		r.width = cvRound(r.width);
		r.height = cvRound(r.height);
		rectangle(src, r.tl(), r.br(), Scalar(0,0,255), 2);
	}
	end = clock();    
	time = (end-start)/CLOCKS_PER_SEC;
    cout<<"SVM检测所需时间为"<<time<<endl;
	//保存图像
	imwrite("ImgProcessed_normal_SVM.jpg",src);
	namedWindow("src",0);
	imshow("src",src);
	waitKey();//注意：imshow之后必须加waitKey，否则无法显示图像
	system("pause");

	/******************读入单个64*128的测试图并对其HOG描述子进行分类*********************/
	////读取测试图片(64*128大小)，并计算其HOG描述子
	////Mat testImg = imread("person014142.jpg");
	//Mat testImg = imread("noperson000026.jpg");
	//vector<float> descriptor;
	//hog.compute(testImg,descriptor,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
	//Mat testFeatureMat = Mat::zeros(1,3780,CV_32FC1);//测试样本的特征向量矩阵
	////将计算好的HOG描述子复制到testFeatureMat矩阵中
	//for(int i=0; i<descriptor.size(); i++)
	//	testFeatureMat.at<float>(0,i) = descriptor[i];

	////用训练好的SVM分类器对测试图片的特征向量进行分类
	//int result = svm.predict(testFeatureMat);//返回类标
	//cout<<"分类结果："<<result<<endl;
}