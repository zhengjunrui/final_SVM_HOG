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
	//video2image("E:\\��ҵ���\\����\\ʵ������Ƶ20180126\\ˤ��\\A(1).mov","E:\\��ҵ���\\����\\pos֡����ͼ\\") ;
	//video2image("E:\\��ҵ���\\����\\������Ƶ\\traffic3.mov","E:\\��ҵ���\\����\\pos֡����ͼ\\",2) ;
	/*Pos_List();
    Neg_List();
	resize_Pos();
	resize_and_flip_Neg();
	Pos_List_AfterFlip();
	Neg_List_AfterFlip();*/
	//set_SVM("E:\\��ҵ���\\����\\��ע�ı�\\500��������\\0512�Ҷ�+Sobel��.xml");

	//Detecting("E:\\��ҵ���\\����\\��ע�ı�\\500��������\\�Ҷ�+Sobel.xml","E:\\��ҵ���\\����\\ʵ������Ƶ20180126\\�ǻ�\\B(��).mov",360,200);
	Detecting("E:\\��ҵ���\\����\\��ע�ı�\\500��������\\�޴���.xml","E:\\��ҵ���\\����\\������Ƶ\\traffic2.mov",480,270);
	//Default_Detector("E:\\��ҵ���\\����\\ʵ������Ƶ20180126\\�ǻ�\\B.mov");

	return 0;
}