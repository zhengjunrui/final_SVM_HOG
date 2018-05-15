#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <conio.h>
#include <time.h>
#include"Gamma.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#define X 30
#define Y 4
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

void Detecting(char* A,char* B,int num1,int num2)
{
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	int num=0;	//储存上一帧的方框个数
	int pre_found[X][Y]={0};
	int pre2_found[X][Y]={0};
	int flag=0;	//前后帧突然出现方框的标记
	MySVM svm;//SVM分类器	
	
	svm.load(A);//从XML文件读取训练好的SVM模型
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

	//保存检测子参数到文件
	ofstream fout("E:\\毕业设计\\代码\\标注文本\\HOGDetectorForOpenCV.txt");
	for(int i=0; i<myDetector.size(); i++)
	{
		fout<<myDetector[i]<<endl;
	}

	
	/**************读入图片进行HOG行人检测******************/
	CvCapture *capture = cvCreateFileCapture(B); 
	IplImage *frame;
	string name;
	int a=1;
	while(1)
	{
		string a1 = to_string(static_cast<long long>(a));
		name = a1 + ".jpg";
		cout<<name<<endl;
		frame = cvQueryFrame(capture);    //（循环）将capture（下一帧图像）加载到内存，frame的格式是IplImage*
		if(!frame)
		{
			cout<<"检测完成"<<endl;
			break;    //capture是null的情况了，说明没有可读帧了，所以跳出	
		}
		Mat src = cvarrToMat(frame); 		

		//Mat src = imread(B);	//图片要小

		if(src.rows>320||src.cols>180)
			resize(src,src,Size(num1,num2));	//缩小图片
		Mat src_Gray;
		namedWindow("原图",1);
		imshow("原图",src);

		//进行RGB的直方图均衡化
		/*Mat srcRGB[3];  
		split(src, srcRGB);
		for (int i = 0; i < 3; i++) 
		{
			equalizeHist(srcRGB[i], srcRGB[i]); 
		}
		Mat src_EquHist;
		merge(srcRGB, 3, src_EquHist); */

		/*namedWindow("after_EquHist",CV_WINDOW_NORMAL);
		imshow("after_EquHist",src_EquHist);*/

		vector<Rect> found, found_filtered;//矩形框数组
		cout<<"进行多尺度HOG人体检测"<<endl;
		double start,end,time;
		start = clock();    
	
		/*增强图像亮度*/
		/*for( int y = 0; y < src.rows; y++ )
		{ 
			for( int x = 0; x < src.cols; x++ )
			{ 
				for( int c = 0; c < 3; c++ )
				{
					src.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( (src.at<Vec3b>(y,x)[c] ) - 50 );
				}
			}
		}*/

		Mat src1;
		src1 = src.clone();	//复制
		//src = gamma(src);
		//imshow("Gamma校正",src);	
		//GaussianBlur(src, src, Size(3, 3), 0);
		/*namedWindow("after_GaussianBlur",CV_WINDOW_NORMAL);
		imshow("after_GaussianBlur",src);*/
		cvtColor(src, src_Gray, CV_RGB2GRAY);	//灰度化图像（预处理步骤）
		//normalize(src , src_Gray, 1, 0, NORM_MINMAX);
		//Sobel算子进行运算
	    Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;
	    Sobel(src_Gray, grad_x, CV_16S, 1, 0, 3, 1, 1,BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);	//将图片转化成为8位图形
	    Sobel(src_Gray, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
		convertScaleAbs(grad_y,abs_grad_y);
		addWeighted(abs_grad_x, 0.3, abs_grad_y, 0.7, 0, src_Gray);
		namedWindow("after_Sobel",1);
		imshow("after_Sobel",src_Gray);
		//medianBlur(src, src_Gray, 3);  //中值滤波操作
		//namedWindow("after_medianBlur",CV_WINDOW_NORMAL);
		//imshow("after_medianBlur",src_Gray);
		//equalizeHist(src_Gray, src_Gray);		//灰度直方图均衡化

		//myHOG.detectMultiScale(src_Gray, found, 0, Size(8,8), Size(32,32), 1.05, 2);//对图片进行多尺度行人检测
		myHOG.detectMultiScale(src_Gray, found, 0, Size(8,8), Size(32,32), 1.05, 1);//对图片进行多尺度行人检测
		cout<<"找到的矩形框个数："<<found.size()<<endl;
		//imshow("src",src);
		//找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中(读图时使用)
		/*for(int i=0; i < found.size(); i++)
		{
			Rect r = found[i];
			int j=0;
			for(; j < found.size(); j++)
				if(j != i && (r & found[j]) == r)
					break;
			if( j == found.size())
				found_filtered.push_back(r);
		}*/

		//帧间处理(读视频时使用)
		for(int i=0; i < found.size(); i++)
		{
			Rect r = found[i];
			int j=0,T=0;
			flag=0;
			if(num!=0)
			{
				//过滤坐标与上一帧坐标偏离太远的坐标


				for(int k=0; pre2_found[k][0]!=0;k++)
				{
					//cout<<1<<endl;
					if(((found[i].x-pre2_found[k][0])*(found[i].x-pre2_found[k][0])<900) && ((found[i].y-pre2_found[k][1])*(found[i].y-pre2_found[k][1])<900))
					{

						for(int K=0; pre_found[K][0]!=0;K++)
						{
							if(((found[i].x-pre_found[K][0])*(found[i].x-pre_found[K][0])<225) && ((found[i].y-pre_found[K][1])*(found[i].y-pre_found[K][1])<225))
							{

								flag=2;	//存在相邻数组
								break;								
							}
						}
					}
				}
				//cout<<"flag:"<<flag<<endl;
			}


			for(j=0 ; j < found.size(); j++)
			{
				if(flag!=2)
					break;
				if((j != i) && ((r & found[j]) == r))
					break;
			}
			if( j == found.size())
				found_filtered.push_back(r);
		
			//system("pause");

			num = found.size();
			//cout<<num<<endl;

		}

		//数组初始化
		for(int I=0; pre2_found[I][0]!=0;I++)
		{
			pre2_found[I][0] = 0;
			pre2_found[I][1] = 0;		
			//cout<<pre2_found[I][0]<<endl<<pre2_found[I][1]<<endl;
			//cout<<endl;
		}		
			for(int I=0; pre_found[I][0]!=0;I++)
		{
			pre2_found[I][0] = pre_found[I][0];
			pre2_found[I][1] = pre_found[I][1];			
			//if(name=="68.jpg")
				//cout<<"存前二个帧的数组："<<endl<<pre2_found[I][0]<<endl<<pre2_found[I][1]<<endl;
			//cout<<endl;
		}			
			for(int I=0; pre_found[I][0]!=0;I++)
		{
			pre_found[I][0] = 0;
			pre_found[I][1] = 0;			
		}
		//方框左上角的坐标一一存入数组
		for(int I=0; I < found.size(); I++)
		{
			pre_found[I][0] = found[I].x;
			pre_found[I][1] = found[I].y;

			//cout<<endl;
		}
		/*-----------------------------------------分割线----------------------------------*/

		//画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
		for(int i=0; i<found_filtered.size(); i++)
		{
			Rect r = found_filtered[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			//r.y += cvRound(r.height*0.07);
			r.y += cvRound(r.height*0.1);
			r.height = cvRound(r.height*0.8);
			//r.height = cvRound(r.height*);
			//r.width = cvRound(r.width);
			//r.height = cvRound(r.height);
			rectangle(src1, r.tl(), r.br(), Scalar(0,0,255), 2);
		}
		end = clock();    
		time = (end-start)/CLOCKS_PER_SEC;
		cout<<"SVM检测所需时间为"<<time<<"秒"<<endl;
		//保存图像
		//imwrite("example.png",src1);
		//imwrite(name,src1);
		//cout<<1<<endl;
		namedWindow("src",1);
		imshow("src",src1);	
		//imwrite("result.png",src_Gray);
		waitKey(1);//注意：imshow之后必须加waitKey，否则无法显示图像	
		a++;
	}
	waitKey();
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

void Default_Detector(char* B)
{
    string name;
	CvCapture *capture = cvCreateFileCapture(B); 
	IplImage *frame;
	int a=1;
	//定义HOG对象，采用默认参数，或者按照下面的格式自己设置  
	HOGDescriptor defaultHog;  
	    //(cv::Size(64, 128), cv::Size(16, 16), cv::Size(8, 8),   
		                        //cv::Size(8, 8),9, 1, -1,   
			                    //cv::HOGDescriptor::L2Hys, 0.2, true,   
				                //cv::HOGDescriptor::DEFAULT_NLEVELS);  
  	//设置SVM分类器，用默认分类器  
	defaultHog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());  
	while(1)
	{
		string a1 = to_string(static_cast<long long>(a));
		name = a1 + ".jpg";
		cout<<name<<endl;
		frame = cvQueryFrame(capture);    //（循环）将capture（下一帧图像）加载到内存，frame的格式是IplImage*
		if(!frame)
		{
			cout<<"检测完成"<<endl;
			break;    //capture是null的情况了，说明没有可读帧了，所以跳出	
		}
		Mat img = cvarrToMat(frame); 
		//Mat img;  
		vector<Rect> people;  
		//img = imread(A,1);  
		if(img.rows>320||img.cols>180)
			resize(img,img,Size(400,225));	//缩小图片

  
		//对图像进行多尺度行人检测，返回结果为矩形框  
		defaultHog.detectMultiScale(img, people,0,Size(8,8),Size(0,0),1.05,2);  
  
		//画长方形，框出行人  
		for (int i = 0; i < people.size(); i++)  
	    {  
	        Rect r = people[i];  
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			//r.y += cvRound(r.height*0.07);
			r.y += cvRound(r.height*0.1);
			r.height = cvRound(r.height*0.8);
	        rectangle(img, r.tl(), r.br(), Scalar(0, 0, 255), 2);  
	    }  
  
	    namedWindow("检测行人", CV_WINDOW_AUTOSIZE);  
		imshow("检测行人", img);  
		a++;
		//imwrite("example.png",img);
		waitKey(1);	
	}
}