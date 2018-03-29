#include <io.h>
#include<stdlib.h>
#include <iostream>
#include <fstream>

using namespace std;

void Pos_List()
{
	cout<<"Writing..."<<endl;
	_finddata_t file;	//_finddata_t是用来存储文件各种信息的结构体
	ofstream outf; 
	//outf.open("E:\\毕业设计\\代码\\标注文本\\INRIADATAPositiveImageList.txt");
	outf.open("E:\\毕业设计\\代码\\标注文本\\PositiveImageList.txt");
	int k;
	long HANDLE;
	//k = HANDLE = _findfirst( "E:\\毕业设计\\代码\\video\\INRIADATA\\normalized_images\\train\\pos\\*.png", &file );
	k = HANDLE = _findfirst( "E:\\毕业设计\\代码\\素材pos帧\\*.png", &file );
	while( k != -1 )
	{
		outf << file.name << endl;	//输出文件的名字
		k = _findnext( HANDLE, &file );
	}
	_findclose( HANDLE );
	outf.close();
}

void Neg_List()
{
	_finddata_t file;	//_finddata_t是用来存储文件各种信息的结构体
	ofstream outf; 
	//outf.open("E:\\毕业设计\\代码\\标注文本\\INRIADATANegativeImageList.txt");
	outf.open("E:\\毕业设计\\代码\\标注文本\\NegativeImageList.txt");
	int k;
	long HANDLE;
	k = HANDLE = _findfirst( "E:\\毕业设计\\代码\\素材neg帧1\\*.png", &file );
	//k = HANDLE = _findfirst( "E:\\毕业设计\\代码\\video\\INRIADATA\\normalized_images\\train\\neg\\*.png", &file );
	while( k != -1 )
	{
		outf << file.name << endl;	//输出文件的文件名
		k = _findnext( HANDLE, &file );
	}
	_findclose( HANDLE );
	outf.close();
	cout<<"Finish!"<<endl;
	system("pause");
}