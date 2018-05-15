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

#define PosSamNO 2000  //����������
#define NegSamNO 2000 //����������

void resize_and_flip_Pos()
{
	string ImgName;//ͼƬ��(����·��)
	string ImgName_Resize;	//�ı�ͼƬ��С���ͼƬ·��
	string ImgName_Flip;	//������ͼƬ·��
	ifstream finPos("E:\\��ҵ���\\����\\��ע�ı�\\PositiveImageList.txt");	//������ͼƬ���ļ����б�

	for(int num=0; num<PosSamNO && getline(finPos,ImgName); num++)
	{
		//cout<<"����"<<ImgName<<endl;
		ImgName_Flip = "E:\\��ҵ���\\����\\�ز�pos֡_resize\\Flip_" + ImgName;
		ImgName_Resize = "E:\\��ҵ���\\����\\�ز�pos֡_resize\\" + ImgName;
		ImgName = "E:\\��ҵ���\\����\\�ز�pos֡\\" + ImgName;//������������·����
		//cout<<ImgName_Resize<<endl;
		Mat src = imread(ImgName);//��ȡͼƬ
		Mat dst;
		//cvtColor(src, src,CV_RGB2GRAY);
		//resize(src,src,Size(64,128)); //��������ͼƬ��Ϊ64*128 
		resize(src,src,Size(128,256)); //������ͼƬ��Ϊ128*256
		flip(src, dst, 1); 
		imwrite(ImgName_Resize, src);
		imwrite(ImgName_Flip, dst);
	}
	cout<<"ת����������С��ɣ�"<<endl;
	//system("pause");
}

void resize_Pos()
{
	string ImgName;//ͼƬ��(����·��)
	string ImgName_Resize;	//�ı�ͼƬ��С���ͼƬ·��
	ifstream finPos("E:\\��ҵ���\\����\\��ע�ı�\\PositiveImageList.txt");	//������ͼƬ���ļ����б�

	for(int num=0; num<PosSamNO && getline(finPos,ImgName); num++)
	{
		//cout<<"����"<<ImgName<<endl;
		ImgName_Resize = "E:\\��ҵ���\\����\\�ز�pos֡_resize\\" + ImgName;
		ImgName = "E:\\��ҵ���\\����\\�ز�pos֡\\" + ImgName;//������������·����
		//cout<<ImgName_Resize<<endl;
		Mat src = imread(ImgName);//��ȡͼƬ
		//cvtColor(src, src,CV_RGB2GRAY);
		//resize(src,src,Size(64,128)); //��������ͼƬ��Ϊ64*128 
		resize(src,src,Size(128,256)); //������ͼƬ��Ϊ128*256
		imwrite(ImgName_Resize, src);
	}
	cout<<"ת����������С��ɣ�"<<endl;
	//system("pause");
}

void resize_and_flip_Neg()
{
	string ImgName;//ͼƬ��(����·��)
	string ImgName_Resize;	//�ı�ͼƬ��С���ͼƬ·��
	string ImgName_Flip;	//������ͼƬ·��
	ifstream finPos("E:\\��ҵ���\\����\\��ע�ı�\\NegativeImageList.txt");	//������ͼƬ���ļ����б�
	//ifstream finPos("E:\\��ҵ���\\����\\��ע�ı�\\INRIADATANegativeImageList.txt");	//������ͼƬ���ļ����б�

	for(int num=0; num<NegSamNO && getline(finPos,ImgName); num++)
	{
		//cout<<"����"<<ImgName<<endl;
		ImgName_Resize = "E:\\��ҵ���\\����\\�ز�neg֡_resize\\" + ImgName;	//��ֹImgName�ں��汻����
		ImgName_Flip = "E:\\��ҵ���\\����\\�ز�neg֡_resize\\Flip_" + ImgName;
		ImgName = "E:\\��ҵ���\\����\\�ز�neg֡\\" + ImgName;//���ϸ�������·����
		//ImgName_Resize = "E:\\��ҵ���\\����\\video\\INRIADATA\\normalized_images\\train\\neg\\" + ImgName;	//��ֹImgName�ں��汻����
		//ImgName_Flip = "E:\\��ҵ���\\����\\video\\INRIADATA\\normalized_images\\train\\neg\\Flip_" + ImgName;
		//ImgName = "E:\\��ҵ���\\����\\video\\INRIADATA\\normalized_images\\train\\neg\\" + ImgName;//���ϸ�������·����
		Mat dst;
		Mat src = imread(ImgName);//��ȡͼƬ	
		 
		//cvtColor(src, src,CV_RGB2GRAY);
		//resize(src,src,Size(64,128)); //��������ͼƬ��Ϊ64*128 
		if(src.rows>src.cols)
			resize(src,src,Size(128,256)); //������ͼƬ��Ϊ256*192
		else
			resize(src,src,Size(320,240));
		flip(src, dst, 1); 
		imwrite(ImgName_Resize, src);
		imwrite(ImgName_Flip, dst);
	}
	//system("pause");
	cout<<"ת����������С��ɣ�"<<endl;
}