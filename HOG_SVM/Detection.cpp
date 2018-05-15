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

void Detecting(char* A,char* B,int num1,int num2)
{
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	int num=0;	//������һ֡�ķ������
	int pre_found[X][Y]={0};
	int pre2_found[X][Y]={0};
	int flag=0;	//ǰ��֡ͻȻ���ַ���ı��
	MySVM svm;//SVM������	
	
	svm.load(A);//��XML�ļ���ȡѵ���õ�SVMģ��
	//svm.load("E:\\��ҵ���\\����\\��ע�ı�\\SVM_HOG.xml");//��XML�ļ���ȡѵ���õ�SVMģ��

	DescriptorDim = svm.get_var_count();//����������ά������HOG�����ӵ�ά��
	int supportVectorNum = svm.get_support_vector_count();//֧�������ĸ���
	cout<<"֧������������"<<supportVectorNum<<endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha���������ȵ���֧����������,һά���󣬷ŵ��Ǿ��ߺ�����alpha����,svm���ܶ���double
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);	//֧���������󣬷ŵ��Ǹ���֧����������������
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_64FC1);//alpha��������֧����������Ľ��,һά����

	//��֧�����������ݸ��Ƶ�supportVectorMat������
	for(int i=0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);	//���ص�i��֧������������ָ��
		for(int j=0; j<DescriptorDim; j++)
		{
			supportVectorMat.at<float>(i,j) = pSVData[j];			
		}
	}

	//��alpha���������ݸ��Ƶ�alphaMat��
	double * pAlphaData = svm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����
	for(int i=0; i<supportVectorNum; i++)
	{
		//�趨��������ֹ��doubleת����float��ʱ���������
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

	//����-(alphaMat * supportVectorMat),����ŵ�resultMat�У��Ӹ�������ΪHOG������ȡ�ľ���ͼ�������ľ��������෴�ģ����ϵĽ��ͣ�
	resultMat = -1.0 * alphaMat *  supportVectorMat;		

	//�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����
	vector<float> myDetector;

	//��resultMat�е����ݸ��Ƶ�����myDetector��һ����
	for(int i=0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0,i));
	}
	//������ƫ����rho���õ������
	myDetector.push_back(svm.get_rho());
	cout<<"�����ά����"<<myDetector.size()<<endl;	
	//����HOGDescriptor�ļ����
	HOGDescriptor myHOG;
	myHOG.setSVMDetector(myDetector);

	//�������Ӳ������ļ�
	ofstream fout("E:\\��ҵ���\\����\\��ע�ı�\\HOGDetectorForOpenCV.txt");
	for(int i=0; i<myDetector.size(); i++)
	{
		fout<<myDetector[i]<<endl;
	}

	
	/**************����ͼƬ����HOG���˼��******************/
	CvCapture *capture = cvCreateFileCapture(B); 
	IplImage *frame;
	string name;
	int a=1;
	while(1)
	{
		string a1 = to_string(static_cast<long long>(a));
		name = a1 + ".jpg";
		cout<<name<<endl;
		frame = cvQueryFrame(capture);    //��ѭ������capture����һ֡ͼ�񣩼��ص��ڴ棬frame�ĸ�ʽ��IplImage*
		if(!frame)
		{
			cout<<"������"<<endl;
			break;    //capture��null������ˣ�˵��û�пɶ�֡�ˣ���������	
		}
		Mat src = cvarrToMat(frame); 		

		//Mat src = imread(B);	//ͼƬҪС

		if(src.rows>320||src.cols>180)
			resize(src,src,Size(num1,num2));	//��СͼƬ
		Mat src_Gray;
		namedWindow("ԭͼ",1);
		imshow("ԭͼ",src);

		//����RGB��ֱ��ͼ���⻯
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

		vector<Rect> found, found_filtered;//���ο�����
		cout<<"���ж�߶�HOG������"<<endl;
		double start,end,time;
		start = clock();    
	
		/*��ǿͼ������*/
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
		src1 = src.clone();	//����
		//src = gamma(src);
		//imshow("GammaУ��",src);	
		//GaussianBlur(src, src, Size(3, 3), 0);
		/*namedWindow("after_GaussianBlur",CV_WINDOW_NORMAL);
		imshow("after_GaussianBlur",src);*/
		cvtColor(src, src_Gray, CV_RGB2GRAY);	//�ҶȻ�ͼ��Ԥ�����裩
		//normalize(src , src_Gray, 1, 0, NORM_MINMAX);
		//Sobel���ӽ�������
	    Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;
	    Sobel(src_Gray, grad_x, CV_16S, 1, 0, 3, 1, 1,BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);	//��ͼƬת����Ϊ8λͼ��
	    Sobel(src_Gray, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
		convertScaleAbs(grad_y,abs_grad_y);
		addWeighted(abs_grad_x, 0.3, abs_grad_y, 0.7, 0, src_Gray);
		namedWindow("after_Sobel",1);
		imshow("after_Sobel",src_Gray);
		//medianBlur(src, src_Gray, 3);  //��ֵ�˲�����
		//namedWindow("after_medianBlur",CV_WINDOW_NORMAL);
		//imshow("after_medianBlur",src_Gray);
		//equalizeHist(src_Gray, src_Gray);		//�Ҷ�ֱ��ͼ���⻯

		//myHOG.detectMultiScale(src_Gray, found, 0, Size(8,8), Size(32,32), 1.05, 2);//��ͼƬ���ж�߶����˼��
		myHOG.detectMultiScale(src_Gray, found, 0, Size(8,8), Size(32,32), 1.05, 1);//��ͼƬ���ж�߶����˼��
		cout<<"�ҵ��ľ��ο������"<<found.size()<<endl;
		//imshow("src",src);
		//�ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����ο����found_filtered��(��ͼʱʹ��)
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

		//֡�䴦��(����Ƶʱʹ��)
		for(int i=0; i < found.size(); i++)
		{
			Rect r = found[i];
			int j=0,T=0;
			flag=0;
			if(num!=0)
			{
				//������������һ֡����ƫ��̫Զ������


				for(int k=0; pre2_found[k][0]!=0;k++)
				{
					//cout<<1<<endl;
					if(((found[i].x-pre2_found[k][0])*(found[i].x-pre2_found[k][0])<900) && ((found[i].y-pre2_found[k][1])*(found[i].y-pre2_found[k][1])<900))
					{

						for(int K=0; pre_found[K][0]!=0;K++)
						{
							if(((found[i].x-pre_found[K][0])*(found[i].x-pre_found[K][0])<225) && ((found[i].y-pre_found[K][1])*(found[i].y-pre_found[K][1])<225))
							{

								flag=2;	//������������
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

		//�����ʼ��
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
				//cout<<"��ǰ����֡�����飺"<<endl<<pre2_found[I][0]<<endl<<pre2_found[I][1]<<endl;
			//cout<<endl;
		}			
			for(int I=0; pre_found[I][0]!=0;I++)
		{
			pre_found[I][0] = 0;
			pre_found[I][1] = 0;			
		}
		//�������Ͻǵ�����һһ��������
		for(int I=0; I < found.size(); I++)
		{
			pre_found[I][0] = found[I].x;
			pre_found[I][1] = found[I].y;

			//cout<<endl;
		}
		/*-----------------------------------------�ָ���----------------------------------*/

		//�����ο���Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,����������Ҫ��һЩ����
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
		cout<<"SVM�������ʱ��Ϊ"<<time<<"��"<<endl;
		//����ͼ��
		//imwrite("example.png",src1);
		//imwrite(name,src1);
		//cout<<1<<endl;
		namedWindow("src",1);
		imshow("src",src1);	
		//imwrite("result.png",src_Gray);
		waitKey(1);//ע�⣺imshow֮������waitKey�������޷���ʾͼ��	
		a++;
	}
	waitKey();
	system("pause");

	/******************���뵥��64*128�Ĳ���ͼ������HOG�����ӽ��з���*********************/
	////��ȡ����ͼƬ(64*128��С)����������HOG������
	////Mat testImg = imread("person014142.jpg");
	//Mat testImg = imread("noperson000026.jpg");
	//vector<float> descriptor;
	//hog.compute(testImg,descriptor,Size(8,8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
	//Mat testFeatureMat = Mat::zeros(1,3780,CV_32FC1);//����������������������
	////������õ�HOG�����Ӹ��Ƶ�testFeatureMat������
	//for(int i=0; i<descriptor.size(); i++)
	//	testFeatureMat.at<float>(0,i) = descriptor[i];

	////��ѵ���õ�SVM�������Բ���ͼƬ�������������з���
	//int result = svm.predict(testFeatureMat);//�������
	//cout<<"��������"<<result<<endl;
}

void Default_Detector(char* B)
{
    string name;
	CvCapture *capture = cvCreateFileCapture(B); 
	IplImage *frame;
	int a=1;
	//����HOG���󣬲���Ĭ�ϲ��������߰�������ĸ�ʽ�Լ�����  
	HOGDescriptor defaultHog;  
	    //(cv::Size(64, 128), cv::Size(16, 16), cv::Size(8, 8),   
		                        //cv::Size(8, 8),9, 1, -1,   
			                    //cv::HOGDescriptor::L2Hys, 0.2, true,   
				                //cv::HOGDescriptor::DEFAULT_NLEVELS);  
  	//����SVM����������Ĭ�Ϸ�����  
	defaultHog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());  
	while(1)
	{
		string a1 = to_string(static_cast<long long>(a));
		name = a1 + ".jpg";
		cout<<name<<endl;
		frame = cvQueryFrame(capture);    //��ѭ������capture����һ֡ͼ�񣩼��ص��ڴ棬frame�ĸ�ʽ��IplImage*
		if(!frame)
		{
			cout<<"������"<<endl;
			break;    //capture��null������ˣ�˵��û�пɶ�֡�ˣ���������	
		}
		Mat img = cvarrToMat(frame); 
		//Mat img;  
		vector<Rect> people;  
		//img = imread(A,1);  
		if(img.rows>320||img.cols>180)
			resize(img,img,Size(400,225));	//��СͼƬ

  
		//��ͼ����ж�߶����˼�⣬���ؽ��Ϊ���ο�  
		defaultHog.detectMultiScale(img, people,0,Size(8,8),Size(0,0),1.05,2);  
  
		//�������Σ��������  
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
  
	    namedWindow("�������", CV_WINDOW_AUTOSIZE);  
		imshow("�������", img);  
		a++;
		//imwrite("example.png",img);
		waitKey(1);	
	}
}