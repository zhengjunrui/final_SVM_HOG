#include <io.h>
#include<stdlib.h>
#include <iostream>
#include <fstream>

using namespace std;

void Pos_List()
{
	cout<<"Writing..."<<endl;
	_finddata_t file;	//_finddata_t�������洢�ļ�������Ϣ�Ľṹ��
	ofstream outf; 
	//outf.open("E:\\��ҵ���\\����\\��ע�ı�\\INRIADATAPositiveImageList.txt");
	outf.open("E:\\��ҵ���\\����\\��ע�ı�\\PositiveImageList.txt");
	int k;
	long HANDLE;
	//k = HANDLE = _findfirst( "E:\\��ҵ���\\����\\video\\INRIADATA\\normalized_images\\train\\pos\\*.png", &file );
	k = HANDLE = _findfirst( "E:\\��ҵ���\\����\\�ز�pos֡\\*.png", &file );
	while( k != -1 )
	{
		outf << file.name << endl;	//����ļ�������
		k = _findnext( HANDLE, &file );
	}
	_findclose( HANDLE );
	outf.close();
}

void Neg_List()
{
	_finddata_t file;	//_finddata_t�������洢�ļ�������Ϣ�Ľṹ��
	ofstream outf; 
	//outf.open("E:\\��ҵ���\\����\\��ע�ı�\\INRIADATANegativeImageList.txt");
	outf.open("E:\\��ҵ���\\����\\��ע�ı�\\NegativeImageList.txt");
	int k;
	long HANDLE;
	k = HANDLE = _findfirst( "E:\\��ҵ���\\����\\�ز�neg֡1\\*.png", &file );
	//k = HANDLE = _findfirst( "E:\\��ҵ���\\����\\video\\INRIADATA\\normalized_images\\train\\neg\\*.png", &file );
	while( k != -1 )
	{
		outf << file.name << endl;	//����ļ����ļ���
		k = _findnext( HANDLE, &file );
	}
	_findclose( HANDLE );
	outf.close();
	cout<<"Finish!"<<endl;
	system("pause");
}