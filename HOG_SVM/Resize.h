#ifndef __RESIZE_H__
#define __RESIZE_H__
/*#include "highgui.h"
#include "cv.h"*/
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

#define PosSamNO 10000  //正样本个数


void resize()
{
	string ImgName;//图片名(绝对路径)
	ifstream finPos("E:\\毕业设计\\代码\\标注文本\\PositiveImageList.txt");	//正样本图片的文件名列表

	for(int num=0; num<PosSamNO && getline(finPos,ImgName); num++)
	{
		cout<<"处理："<<ImgName<<endl;
		ImgName = "E:\\毕业设计\\代码\\素材pos帧1\\" + ImgName;//加上正样本的路径名
		
		Mat src = imread(ImgName);//读取图片	
		//cvtColor(src, src,CV_RGB2GRAY);
		//resize(src,src,Size(64,128)); //将正样本图片缩为64*128 
		resize(src,src,Size(256,192)); //将样本图片缩为256*192
		imwrite(ImgName,src);
	}
	system("pause");
}

#endif