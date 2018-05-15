#include <opencv2/highgui/highgui.hpp>      
#include <opencv2/imgproc/imgproc.hpp>  

using namespace cv;  
  
Mat gamma(Mat src)  
{  
    //Mat image = imread(src);  
    Mat imageGamma(src.size(), CV_32FC3);  
	//normalize(src, src, 0, 1, CV_MINMAX); 
    for (int i = 0; i < src.rows; i++)  
    {  
        for (int j = 0; j < src.cols; j++)  
        {  
            /*imageGamma.at<Vec3f>(i, j)[0] = (src.at<Vec3b>(i, j)[0])*(src.at<Vec3b>(i, j)[0])*(src.at<Vec3b>(i, j)[0]);  //3�η��ٶ�����
            imageGamma.at<Vec3f>(i, j)[1] = (src.at<Vec3b>(i, j)[1])*(src.at<Vec3b>(i, j)[1])*(src.at<Vec3b>(i, j)[1]);  
            imageGamma.at<Vec3f>(i, j)[2] = (src.at<Vec3b>(i, j)[2])*(src.at<Vec3b>(i, j)[2])*(src.at<Vec3b>(i, j)[2]); */ 
			
			imageGamma.at<Vec3f>(i, j)[0] = (src.at<Vec3b>(i, j)[0])*(src.at<Vec3b>(i, j)[0]);   //2�η��ٶ�����
            imageGamma.at<Vec3f>(i, j)[1] = (src.at<Vec3b>(i, j)[1])*(src.at<Vec3b>(i, j)[1]);  
            imageGamma.at<Vec3f>(i, j)[2] = (src.at<Vec3b>(i, j)[2])*(src.at<Vec3b>(i, j)[2]); 
        } 
    }  
    //��һ����0~255    
    normalize(imageGamma, imageGamma, 0, 255, CV_MINMAX);  //��һ����Ѵ�������ֵ�Ҷ�ֵѹ��
    //ת����8bitͼ����ʾ    
    convertScaleAbs(imageGamma, imageGamma);  
    //imshow("ԭͼ", src);  
    //imshow("٤��任ͼ����ǿЧ��", imageGamma);  
    //waitKey();   
	return imageGamma;
} 