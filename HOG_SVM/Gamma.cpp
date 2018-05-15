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
            /*imageGamma.at<Vec3f>(i, j)[0] = (src.at<Vec3b>(i, j)[0])*(src.at<Vec3b>(i, j)[0])*(src.at<Vec3b>(i, j)[0]);  //3次方速度增长
            imageGamma.at<Vec3f>(i, j)[1] = (src.at<Vec3b>(i, j)[1])*(src.at<Vec3b>(i, j)[1])*(src.at<Vec3b>(i, j)[1]);  
            imageGamma.at<Vec3f>(i, j)[2] = (src.at<Vec3b>(i, j)[2])*(src.at<Vec3b>(i, j)[2])*(src.at<Vec3b>(i, j)[2]); */ 
			
			imageGamma.at<Vec3f>(i, j)[0] = (src.at<Vec3b>(i, j)[0])*(src.at<Vec3b>(i, j)[0]);   //2次方速度增长
            imageGamma.at<Vec3f>(i, j)[1] = (src.at<Vec3b>(i, j)[1])*(src.at<Vec3b>(i, j)[1]);  
            imageGamma.at<Vec3f>(i, j)[2] = (src.at<Vec3b>(i, j)[2])*(src.at<Vec3b>(i, j)[2]); 
        } 
    }  
    //归一化到0~255    
    normalize(imageGamma, imageGamma, 0, 255, CV_MINMAX);  //归一化后把大量像素值灰度值压低
    //转换成8bit图像显示    
    convertScaleAbs(imageGamma, imageGamma);  
    //imshow("原图", src);  
    //imshow("伽马变换图像增强效果", imageGamma);  
    //waitKey();   
	return imageGamma;
} 