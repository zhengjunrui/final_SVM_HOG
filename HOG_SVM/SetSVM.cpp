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

#define PosSamNO 589  //����������
#define NegSamNO 588 //����������

//HardExample��SVM����������������Щ������Ҫ���ж���ѵ��
//HardExample�����������������HardExampleNO����0����ʾ�������ʼ���������󣬼�������HardExample����������
//��ʹ��HardExampleʱ��������Ϊ0����Ϊ������������������������ά����ʼ��ʱ�õ����ֵ

#define HardExampleNO 0  

class MySVM : public CvSVM
{
public:
	//���SVM�ľ��ߺ����е�alpha����
	double * get_alpha_vector()
	{
		return this->decision_func->alpha;
	}

	//���SVM�ľ��ߺ����е�rho����,��ƫ����
	float get_rho()
	{
		return this->decision_func->rho;
	}
};

void set_SVM(char* A)
{

	//��ⴰ��(64,128),block�ߴ�(16,16),block����(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9
	HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9); 
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	MySVM svm; // SVM������
	double start,end,time;

	string ImgName;//ͼƬ��(����·��)
	//ifstream finPos("E:\\��ҵ���\\����\\��ע�ı�\\INRIADATAPositiveImageList.txt");	//������ͼƬ���ļ����б�
	ifstream finPos("E:\\��ҵ���\\����\\��ע�ı�\\PositiveImageList.txt");	//������ͼƬ���ļ����б�

	//ifstream finNeg("E:\\��ҵ���\\����\\��ע�ı�\\INRIADATANegativeImageList.txt");	//������ͼƬ���ļ����б�
	ifstream finNeg("E:\\��ҵ���\\����\\��ע�ı�\\NegativeImageList.txt");	//������ͼƬ���ļ����б�

	Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��	
	Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����


	//���ζ�ȡ������ͼƬ������HOG������
	start = clock();
	for(int num=0; num<PosSamNO && getline(finPos,ImgName); num++)
	{
		//cout<<"����"<<ImgName<<endl;
		ImgName = "E:\\��ҵ���\\����\\�ز�pos֡_resize\\" + ImgName;//������������·����
		//ImgName = "E:\\��ҵ���\\����\\video\\INRIADATA\\normalized_images\\train\\pos\\" + ImgName;//������������·����
		Mat src = imread(ImgName);//��ȡͼƬ
		resize(src,src,Size(64,128)); //��������ͼƬ��Ϊ64*128 

		//RGBֱ��ͼ���⻯
		/*Mat srcRGB[3];  
		split(src, srcRGB);
		for (int i = 0; i < 3; i++) 
		{
			equalizeHist(srcRGB[i], srcRGB[i]); 
		}
		merge(srcRGB, 3, src); */
		//src = gamma(src);
		//GaussianBlur(src, src, Size(3, 3), 0);
		cvtColor(src, src,CV_RGB2GRAY);	//��ͼ��ҶȻ�
		//normalize(src,src, 1, 0, NORM_MINMAX );
		//Sobel(src,src,-1,1,1,5);
		//Sobel���ӽ�������
	    Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;
	    Sobel(src, grad_x, CV_16S, 1, 0, 3, 1, 1,BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);	//��ͼƬת����Ϊ8λͼ��
	    Sobel(src, grad_y, CV_16S, 0,  1,3, 1, 1, BORDER_DEFAULT);
		convertScaleAbs(grad_y,abs_grad_y);
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, src);
		//medianBlur(src, src, 3);  //��ֵ�˲�����
		//equalizeHist(src, src); 


		vector<float> descriptors;//HOG����������
		hog.compute(src,descriptors,Size(8,8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
		//cout<<"������ά����"<<descriptors.size()<<endl;

		//�����һ������ʱ��ʼ�����������������������Ϊֻ��֪��������������ά�����ܳ�ʼ��������������
		if( 0 == num )
		{
			DescriptorDim = descriptors.size();//HOG�����ӵ�ά��
			//��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat
			sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
			//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����
			sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
		}

		//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
		for(int i=0; i<DescriptorDim; i++)
		{
			sampleFeatureMat.at<float>(num,i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��
		}
		sampleLabelMat.at<float>(num,0) = 1;//���������Ϊ1������
	}
	end = clock();
	time = (end-start)/CLOCKS_PER_SEC;
	cout<<"������������ȡ��ϣ�"<<endl;
	cout<<"������������ȡ����ʱ��Ϊ"<<time<<"��"<<endl;

	//���ζ�ȡ������ͼƬ������HOG������
	start = clock();
	for(int num=0; num<NegSamNO && getline(finNeg,ImgName); num++)
	{
		//cout<<"����"<<ImgName<<endl;
		ImgName = "E:\\��ҵ���\\����\\�ز�neg֡_resize\\" + ImgName;//���ϸ�������·����
		//ImgName = "E:\\��ҵ���\\����\\video\\INRIADATA\\normalized_images\\train\\neg\\" + ImgName;//���ϸ�������·����
		Mat src = imread(ImgName);//��ȡͼƬ
		//normalize(src,src, 1, 0, NORM_MINMAX,  -1 );

		//RGBֱ��ͼ���⻯
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
		//normalize(src,src, 1, 0, NORM_MINMAX);	//�ҶȻ����ٽ��й�һ������ֵ�����⣿
		//medianBlur(src, src, 3);  //��ֵ�˲�����
		//Sobel(src,src,-1,1,1,5);
		//Sobel���ӽ�������
	    Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;
	    Sobel(src, grad_x, CV_16S, 1, 0, 3, 1, 1,BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);	//��ͼƬת����Ϊ8λͼ��
	    Sobel(src, grad_y, CV_16S, 0,  1,3, 1, 1, BORDER_DEFAULT);
		convertScaleAbs(grad_y,abs_grad_y);
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, src);
		//����������Ĵ�Сû�дﵽ64*128���ͽ������󣨳������Ƕ�64*128�����ؿ����hog��ȡ��
		if(src.rows<64||src.cols<128)
		{
			resize(src,src,Size(64,128));	//������ͼƬ����Ϊ64*128 
		}

		//�Ҷ�ֱ��ͼ���⻯
		//equalizeHist(src, src); 


		vector<float> descriptors;//HOG����������
		hog.compute(src,descriptors,Size(8,8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
		//cout<<"������ά����"<<descriptors.size()<<endl;

		//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
		for(int i=0; i<DescriptorDim; i++)
		{
			sampleFeatureMat.at<float>(num+PosSamNO,i) = descriptors[i];
		}
		//��PosSamNO+num�����������������еĵ�i��Ԫ��
		sampleLabelMat.at<float>(num+PosSamNO,0) = -1;//���������Ϊ-1���ж�����
		//cout<<"1"<<endl;	
	}
	end = clock();
	time = (end-start)/CLOCKS_PER_SEC;
	cout<<"������������ȡ��ϣ�"<<endl;
	cout<<"������������ȡ����ʱ��Ϊ"<<time<<"��"<<endl;

	//����HardExample������
	/*if(HardExampleNO > 0)
	{
		ifstream finHardExample("HardExample_2400PosINRIA_12000NegList.txt");//HardExample���������ļ����б�
		//���ζ�ȡHardExample������ͼƬ������HOG������
		for(int num=0; num<HardExampleNO && getline(finHardExample,ImgName); num++)
		{
			cout<<"����"<<ImgName<<endl;
			ImgName = "D:\\DataSet\\HardExample_2400PosINRIA_12000Neg\\" + ImgName;//����HardExample��������·����
			Mat src = imread(ImgName);//��ȡͼƬ
			//resize(src,img,Size(64,128));

			vector<float> descriptors;//HOG����������
			hog.compute(src,descriptors,Size(8,8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
			//cout<<"������ά����"<<descriptors.size()<<endl;

			//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for(int i=0; i<DescriptorDim; i++)
			{
				sampleFeatureMat.at<float>(num+PosSamNO+NegSamNO,i) = descriptors[i];
			}
			//��PosSamNO+num�����������������еĵ�i��Ԫ��
			sampleLabelMat.at<int>(num+PosSamNO+NegSamNO,0) = -1;//���������Ϊ-1������
		}*/
		

	//���������HOG�������������ļ�
		/*ofstream fout("E:\\��ҵ���\\����\\��ע�ı�\\SampleFeatureMat.txt");
		for(int i=0; i<PosSamNO+NegSamNO; i++)
		{
			fout<<i<<endl;
			for(int j=0; j<DescriptorDim; j++)
			{	fout<<sampleFeatureMat.at<float>(i,j)<<"  ";
	              		
			}
			fout<<endl;
		}*/

		//ѵ��SVM������
		//SVM��������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01,���η�ʽѡ��϶���ʹ�õĲ���
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

		//������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����
		
	//params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 50000, FLT_EPSILON);
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	
	//SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01
	CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);

	cout<<"��ʼѵ��SVM������"<<endl;
	start = clock();
	svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);
	end = clock();
	time = (end-start)/CLOCKS_PER_SEC;
	cout<<"SVMѵ����ɣ�"<<endl;
	cout<<"SVMѵ������ʱ��Ϊ"<<time<<"��"<<endl;

	svm.save(A);//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�
	//svm.save("E:\\��ҵ���\\����\\��ע�ı�\\INRIADATA_SVM_HOG.xml");//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�
	
	//system("pause");
}