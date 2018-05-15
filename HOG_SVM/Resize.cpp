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

#define PosSamNO 2000  //正样本个数
#define NegSamNO 2000 //负样本个数

void resize_and_flip_Pos()
{
	string ImgName;//图片名(绝对路径)
	string ImgName_Resize;	//改变图片大小后的图片路径
	string ImgName_Flip;	//镜像后的图片路径
	ifstream finPos("E:\\毕业设计\\代码\\标注文本\\PositiveImageList.txt");	//正样本图片的文件名列表

	for(int num=0; num<PosSamNO && getline(finPos,ImgName); num++)
	{
		//cout<<"处理："<<ImgName<<endl;
		ImgName_Flip = "E:\\毕业设计\\代码\\素材pos帧_resize\\Flip_" + ImgName;
		ImgName_Resize = "E:\\毕业设计\\代码\\素材pos帧_resize\\" + ImgName;
		ImgName = "E:\\毕业设计\\代码\\素材pos帧\\" + ImgName;//加上正样本的路径名
		//cout<<ImgName_Resize<<endl;
		Mat src = imread(ImgName);//读取图片
		Mat dst;
		//cvtColor(src, src,CV_RGB2GRAY);
		//resize(src,src,Size(64,128)); //将正样本图片缩为64*128 
		resize(src,src,Size(128,256)); //将样本图片缩为128*256
		flip(src, dst, 1); 
		imwrite(ImgName_Resize, src);
		imwrite(ImgName_Flip, dst);
	}
	cout<<"转换正样本大小完成！"<<endl;
	//system("pause");
}

void resize_Pos()
{
	string ImgName;//图片名(绝对路径)
	string ImgName_Resize;	//改变图片大小后的图片路径
	ifstream finPos("E:\\毕业设计\\代码\\标注文本\\PositiveImageList.txt");	//正样本图片的文件名列表

	for(int num=0; num<PosSamNO && getline(finPos,ImgName); num++)
	{
		//cout<<"处理："<<ImgName<<endl;
		ImgName_Resize = "E:\\毕业设计\\代码\\素材pos帧_resize\\" + ImgName;
		ImgName = "E:\\毕业设计\\代码\\素材pos帧\\" + ImgName;//加上正样本的路径名
		//cout<<ImgName_Resize<<endl;
		Mat src = imread(ImgName);//读取图片
		//cvtColor(src, src,CV_RGB2GRAY);
		//resize(src,src,Size(64,128)); //将正样本图片缩为64*128 
		resize(src,src,Size(128,256)); //将样本图片缩为128*256
		imwrite(ImgName_Resize, src);
	}
	cout<<"转换正样本大小完成！"<<endl;
	//system("pause");
}

void resize_and_flip_Neg()
{
	string ImgName;//图片名(绝对路径)
	string ImgName_Resize;	//改变图片大小后的图片路径
	string ImgName_Flip;	//镜像后的图片路径
	ifstream finPos("E:\\毕业设计\\代码\\标注文本\\NegativeImageList.txt");	//正样本图片的文件名列表
	//ifstream finPos("E:\\毕业设计\\代码\\标注文本\\INRIADATANegativeImageList.txt");	//负样本图片的文件名列表

	for(int num=0; num<NegSamNO && getline(finPos,ImgName); num++)
	{
		//cout<<"处理："<<ImgName<<endl;
		ImgName_Resize = "E:\\毕业设计\\代码\\素材neg帧_resize\\" + ImgName;	//防止ImgName在后面被覆盖
		ImgName_Flip = "E:\\毕业设计\\代码\\素材neg帧_resize\\Flip_" + ImgName;
		ImgName = "E:\\毕业设计\\代码\\素材neg帧\\" + ImgName;//加上负样本的路径名
		//ImgName_Resize = "E:\\毕业设计\\代码\\video\\INRIADATA\\normalized_images\\train\\neg\\" + ImgName;	//防止ImgName在后面被覆盖
		//ImgName_Flip = "E:\\毕业设计\\代码\\video\\INRIADATA\\normalized_images\\train\\neg\\Flip_" + ImgName;
		//ImgName = "E:\\毕业设计\\代码\\video\\INRIADATA\\normalized_images\\train\\neg\\" + ImgName;//加上负样本的路径名
		Mat dst;
		Mat src = imread(ImgName);//读取图片	
		 
		//cvtColor(src, src,CV_RGB2GRAY);
		//resize(src,src,Size(64,128)); //将正样本图片缩为64*128 
		if(src.rows>src.cols)
			resize(src,src,Size(128,256)); //将样本图片缩为256*192
		else
			resize(src,src,Size(320,240));
		flip(src, dst, 1); 
		imwrite(ImgName_Resize, src);
		imwrite(ImgName_Flip, dst);
	}
	//system("pause");
	cout<<"转换负样本大小完成！"<<endl;
}