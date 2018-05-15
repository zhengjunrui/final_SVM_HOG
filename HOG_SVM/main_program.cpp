#include "video.h"
#include "SetSVM.h"
#include "Create_List.h"
#include "Detection.h"
#include "Resize.h"
#include "Gamma.h"
#include <math.h>

using namespace std;
int main()
{
	//video2image("E:\\毕业设计\\代码\\实验室视频20180126\\摔倒\\A(1).mov","E:\\毕业设计\\代码\\pos帧待截图\\") ;
	//video2image("E:\\毕业设计\\代码\\测试视频\\traffic3.mov","E:\\毕业设计\\代码\\pos帧待截图\\",2) ;
	/*Pos_List();
    Neg_List();
	resize_Pos();
	resize_and_flip_Neg();
	Pos_List_AfterFlip();
	Neg_List_AfterFlip();*/
	//set_SVM("E:\\毕业设计\\代码\\标注文本\\500正负样本\\0512灰度+Sobel变.xml");

	//Detecting("E:\\毕业设计\\代码\\标注文本\\500正负样本\\灰度+Sobel.xml","E:\\毕业设计\\代码\\实验室视频20180126\\徘徊\\B(剪).mov",360,200);
	Detecting("E:\\毕业设计\\代码\\标注文本\\500正负样本\\无处理.xml","E:\\毕业设计\\代码\\测试视频\\traffic2.mov",480,270);
	//Default_Detector("E:\\毕业设计\\代码\\实验室视频20180126\\徘徊\\B.mov");

	return 0;
}