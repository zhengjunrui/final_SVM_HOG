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

void Detecting()
{
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	MySVM svm;//SVM������	
	
	svm.load("E:\\��ҵ���\\����\\��ע�ı�\\SVM_HOG_20Pos_60Neg.xml");//��XML�ļ���ȡѵ���õ�SVMģ��
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
	//cout<<"resultMat:"<<resultMat<<endl;

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
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//�������Ӳ������ļ�
	ofstream fout("E:\\��ҵ���\\����\\��ע�ı�\\HOGDetectorForOpenCV.txt");
	for(int i=0; i<myDetector.size(); i++)
	{
		fout<<myDetector[i]<<endl;
	}


	/**************����ͼƬ����HOG���˼��******************/
	Mat src = imread("E:\\��ҵ���\\����\\pos֡����ͼ\\19.png");	//ͼƬҪС
	vector<Rect> found, found_filtered;//���ο�����
	cout<<"���ж�߶�HOG������"<<endl;
	double start,end,time;
	start = clock();    
	//cvtColor(src, src,CV_RGB2GRAY);
	myHOG.detectMultiScale(src, found, 0, Size(8,8), Size(32,32), 1.05, 1);//��ͼƬ���ж�߶����˼��
	cout<<"�ҵ��ľ��ο������"<<found.size()<<endl;

	//�ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����ο����found_filtered��
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

	//�����ο���Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,����������Ҫ��һЩ����
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
    cout<<"SVM�������ʱ��Ϊ"<<time<<endl;
	//����ͼ��
	imwrite("ImgProcessed_normal_SVM.jpg",src);
	namedWindow("src",0);
	imshow("src",src);
	waitKey();//ע�⣺imshow֮������waitKey�������޷���ʾͼ��
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