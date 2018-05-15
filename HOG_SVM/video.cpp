#include "highgui.h"
#include "cv.h"
#include "cvaux.h" 
#include <ctype.h>  
#include <stdlib.h>
#include <iostream>
using namespace cv;
using namespace std;

//����Ƶת��֡����ʽ��������ж�����Ľ�ͼ��ע
void video2image(string video,string path,int N) 
{ 
    VideoCapture capture(video); 
    if(!capture.isOpened()) 
    { 
        cerr<<"Failed to open a video"<<endl; 
        return ; 
    } 
 
    Mat frame; 
    int num=1; 
    string filename; 
    char   temp_file[5]; 
 
    for(;;) 
    { 
        capture>>frame; 
            //��ʮ֡��һ��ͼ
            for(int i=0;i<N;i++)
            {
                 capture>>frame;
            }
        
        if(frame.empty()) 
            break; 
        _itoa_s(num,temp_file,4,10); //4��ʾ�ַ�����  10��ʾʮ����  ʵ������ת�ַ���  
        filename = temp_file; 
        filename = path+filename+".png";   
        num++; 
        imwrite(filename,frame); 
    } 
    capture.release(); 
} 