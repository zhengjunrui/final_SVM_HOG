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

#define PosSamNO 10000  //����������


void resize()
{
	string ImgName;//ͼƬ��(����·��)
	ifstream finPos("E:\\��ҵ���\\����\\��ע�ı�\\PositiveImageList.txt");	//������ͼƬ���ļ����б�

	for(int num=0; num<PosSamNO && getline(finPos,ImgName); num++)
	{
		cout<<"����"<<ImgName<<endl;
		ImgName = "E:\\��ҵ���\\����\\�ز�pos֡1\\" + ImgName;//������������·����
		
		Mat src = imread(ImgName);//��ȡͼƬ	
		//cvtColor(src, src,CV_RGB2GRAY);
		//resize(src,src,Size(64,128)); //��������ͼƬ��Ϊ64*128 
		resize(src,src,Size(256,192)); //������ͼƬ��Ϊ256*192
		imwrite(ImgName,src);
	}
	system("pause");
}

#endif