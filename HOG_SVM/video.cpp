#include "highgui.h"
#include "cv.h"
#include "cvaux.h" 
#include <ctype.h>  
#include <stdlib.h>
#include <iostream>
using namespace cv;
using namespace std;

//将视频转成帧的形式，方便进行对人物的截图标注
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
            //隔十帧截一次图
            for(int i=0;i<N;i++)
            {
                 capture>>frame;
            }
        
        if(frame.empty()) 
            break; 
        _itoa_s(num,temp_file,4,10); //4表示字符长度  10表示十进制  实现整型转字符串  
        filename = temp_file; 
        filename = path+filename+".png";   
        num++; 
        imwrite(filename,frame); 
    } 
    capture.release(); 
} 