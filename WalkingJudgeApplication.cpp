// WalkingJudgeApplication.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include <stdio.h>
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <opencv2\highgui/highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\core\core.hpp>
#include <windows.h>
#include <stack>

#define DYNAMIC_FEATURE_COUNT 8
#define STATIC_FEATURE_COUNT 7
#define FEATURE_COUNT DYNAMIC_FEATURE_COUNT+STATIC_FEATURE_COUNT
#define POSITION_COUNT 25
#define WALK_FREQ 11//歩行周期
#define WALK_JUDGE_COUNT 3
#define WALK_JUDGE_THRESH 0.4

#define PI 3.1415926535

using namespace std;
using namespace cv;

void import_position_data(vector<vector<Point3f>>& connected_positions, int start, int end);
void import_temp_data(vector<float>& walkingJudgeTemp);
void extract_features(vector<vector<Point3f>>& connected_positions, vector<vector<float>>& dynamic_features);
void create_walking_vec(vector<float>& walkingJudgeBase, vector<vector<float>> dynamic_features);
bool walking_judge(vector<float> x, vector<float> y);
float evaluate_angle(Point3f c, Point3f a, Point3f b);
float evaluate_seperated_angle(Point3f pA, Point3f pB, Point3f pC, Point3f pD);

enum JointType
{
	JointType_SpineBase = 0,
	JointType_SpineMid = 1,
	JointType_Neck = 2,
	JointType_Head = 3,
	JointType_ShoulderLeft = 4,
	JointType_ElbowLeft = 5,
	JointType_WristLeft = 6,
	JointType_HandLeft = 7,
	JointType_ShoulderRight = 8,
	JointType_ElbowRight = 9,
	JointType_WristRight = 10,
	JointType_HandRight = 11,
	JointType_HipLeft = 12,
	JointType_KneeLeft = 13,
	JointType_AnkleLeft = 14,
	JointType_FootLeft = 15,
	JointType_HipRight = 16,
	JointType_KneeRight = 17,
	JointType_AnkleRight = 18,
	JointType_FootRight = 19,
	JointType_SpineShoulder = 20,
	JointType_HandTipLeft = 21,
	JointType_ThumbLeft = 22,
	JointType_HandTipRight = 23,
	JointType_ThumbRight = 24,
};

const string input_position_filenames[POSITION_COUNT] = {
	"position_SpineBase.dat",
	"position_SpineMid.dat",
	"position_Neck.dat",
	"position_Head.dat",
	"position_ShoulderLeft.dat",
	"position_ElbowLeft.dat",
	"position_WristLeft.dat",
	"position_HandLeft.dat",
	"position_ShoulderRight.dat",
	"position_ElbowRight.dat",
	"position_WristRight.dat",
	"position_HandRight.dat",
	"position_HipLeft.dat",
	"position_KneeLeft.dat",
	"position_AnkleLeft.dat",
	"position_FootLeft.dat",
	"position_HipRight.dat",
	"position_KneeRight.dat",
	"position_AnkleRight.dat",
	"position_FootRight.dat",
	"position_SpineShoulder.dat",
	"position_HandTipLeft.dat",
	"position_ThumbLeft.dat",
	"position_HandTipRight.dat",
	"position_ThumbRight.dat"
};

//動的特徴量算出の際に用いる点の組み合わせ
const vector<vector<int>> dynamic_feature_use_angles = {
	{ JointType_Neck, JointType_Head, JointType_SpineShoulder },
	{ JointType_ShoulderLeft, JointType_SpineShoulder, JointType_ElbowLeft },
	{ JointType_ShoulderRight, JointType_SpineShoulder, JointType_ElbowRight },
	{ JointType_ElbowLeft, JointType_ShoulderLeft, JointType_WristLeft },
	{ JointType_ElbowRight, JointType_ShoulderRight, JointType_WristRight },
	{ JointType_KneeLeft, JointType_HipLeft, JointType_KneeRight, JointType_HipRight },
	{ JointType_KneeLeft, JointType_HipLeft, JointType_AnkleLeft },
	{ JointType_KneeRight, JointType_HipRight, JointType_AnkleRight }
};

enum DynamicFeatureType{
	Feature_Neck = 0,
	Feature_LeftShoulder = 1,
	Feature_RightShoulder = 2,
	Feature_LeftElbow = 3,
	Feature_RightElbow = 4,
	Feature_Hip = 5,
	Feature_LeftKnee = 6,
	Feature_RightKnee = 7,
};

const int walking_judge_features[] = {
	Feature_Hip,
	Feature_LeftKnee,
	Feature_RightKnee
};

int _tmain(int argc, _TCHAR* argv[])
{
	int i;
	int start = 0;
	int end = start+WALK_FREQ;

	//テンプレートデータ読み込み
	vector<float> walkingJudgeTemp;
	import_temp_data(walkingJudgeTemp);

	while (true){
		cout << "------start:" << start << "-----end:" << end << "--------" << endl;
		//位置データ取り込み
		vector<vector<Point3f>> connected_positions;
		for (i = 0; i < POSITION_COUNT; i++){
			vector<Point3f> connected_position;
			connected_positions.push_back(connected_position);
		}
		import_position_data(connected_positions, start, end);
        
		//特徴量算出
		vector<vector<float>> dynamic_features;
		for (i = 0; i < DYNAMIC_FEATURE_COUNT; i++){
			vector<float> feature;
			dynamic_features.push_back(feature);
		}
		extract_features(connected_positions, dynamic_features);

		//歩行判定に使用するベクトル作成
		vector<float> walkingJudgeBase;
		create_walking_vec(walkingJudgeBase, dynamic_features);

		//歩行判定
		bool result = walking_judge(walkingJudgeTemp, walkingJudgeBase);

		start += 1;
		end += 1;
	}
	return 0;
}

ofstream R_logs("R.dat");
bool walking_judge(vector<float> x, vector<float> y){
	int i;
	//平均値算出
	int N = x.size();
	double xmean = 0;
	double ymean = 0;
	for (i = 0; i < N; i++){
		xmean += x[i];
		ymean += y[i];
	}
	xmean /= N;
	ymean /= N;

	double sxx = 0.0;
	double syy = 0.0;
	double sxy = 0.0;
	for (i = 0; i < N; i++){
		sxx += (x[i] - xmean)*(x[i] - xmean);
		syy += (y[i] - ymean)*(y[i] - ymean);
		sxy += (x[i] - xmean)*(y[i] - ymean);
	}

	double R = sxy / sqrt(sxx*syy);
	R_logs << R << endl;
	cout << R << endl;
	if (R > WALK_JUDGE_THRESH){
		return true;
	}
	else{
		return false;
	}
}

void create_walking_vec(vector<float>& walkingJudgeBase, vector<vector<float>> dynamic_features){
	for (int i = 0; i < WALK_JUDGE_COUNT; i++){
		int fIndex = walking_judge_features[i];
		vector<float> f = dynamic_features[fIndex];
		for (auto itr = f.begin(); itr != f.end(); ++itr){
			float val = *itr;
			walkingJudgeBase.push_back(val);
		}
	}
}

void import_temp_data(vector<float>& walkingJudgeTemp){
	ifstream input_datafile;
	input_datafile.open("template.dat");
	if (input_datafile.fail()){
		cout << "ファイルが見つかりません" << endl;
		cin.get();
	}
	string str;
	int d = 0;
	while (getline(input_datafile, str)){
		walkingJudgeTemp.push_back(stof(str));
		d++;
	}
}

void import_position_data(vector<vector<Point3f>>& connected_positions, int start, int end){
	int i, j;
	for (i = 0; i < POSITION_COUNT; i++){
		ifstream input_datafile;
		input_datafile.open(input_position_filenames[i]);
		if (input_datafile.fail()){
			cout << "Exception: ファイルが見つかりません" << endl;
			cin.get();
		}
		string str;
		while (getline(input_datafile, str)){
			string tmp;
			istringstream stream(str);
			int c = 0;
			Point3f p;
			//���s�ǂ�(�X�y�[�X��split)
			while (getline(stream, tmp, ' ')){
				int val = stof(tmp);
				//�͈͂̔���(start�ȏ�end�ȉ��Ȃ玟�̃u���b�N���ǂ�)
				if (c == 0){
					if (val < start || val > end){
						break;
					}
				}
				//X���W
				else if (c == 1){
					p.x = val;
				}
				//Y���W
				else if (c == 2){
					p.y = val;
				}
				//Z���W
				else{
					p.z = val;
					//�X�^�b�N�ɒ��߂�(push)
					connected_positions[i].push_back(p);
				}
				c++;
			}
		}
	}
}

void extract_features(vector<vector<Point3f>>& connected_positions, vector<vector<float>>& dynamic_features){
	int i, j;
	//特徴量算出=>出力
	for (i = 0; i < DYNAMIC_FEATURE_COUNT; i++){
		vector<int> use_points = dynamic_feature_use_angles[i];
		if (use_points.size() == 3){
			vector<Point3f> p1 = connected_positions[use_points[0]];
			vector<Point3f> p2 = connected_positions[use_points[1]];
			vector<Point3f> p3 = connected_positions[use_points[2]];
			for (j = 0; j < p1.size(); j++){
				float angle;
				if (p1[j].x == 0.0 && p1[j].y == 0.0 && p1[j].z == 0.0){
					angle = 0.0;
				}
				else{
					angle = evaluate_angle(p1[j], p2[j], p3[j]);
				}
				dynamic_features[i].push_back(angle);
			}
		}
		else if (use_points.size() == 4){
	    	vector<Point3f> p1 = connected_positions[use_points[0]];
	    	vector<Point3f> p2 = connected_positions[use_points[1]];
     		vector<Point3f> p3 = connected_positions[use_points[2]];
    		vector<Point3f> p4 = connected_positions[use_points[3]];
			for (j = 0; j < p1.size(); j++){
				float angle;
				if (p1[j].x == 0.0 && p1[j].y == 0.0 && p1[j].z == 0.0){
					angle = 0.0;
				}
				else{
					angle = evaluate_seperated_angle(p1[j], p2[j], p3[j], p4[j]);
				}
				dynamic_features[i].push_back(angle);
			}
		}
		else{
			cout << "予期せぬエラー" << endl;
		}
	}
}

//3点を与えられたときに角度を求める
//c:角度の基準点、a,b:それ以外
float evaluate_angle(Point3f c, Point3f a, Point3f b)
{
	int ax_cx = c.x - a.x;
	int ay_cy = c.y - a.y;
	int az_cz = c.z - a.z;
	int bx_cx = c.x - b.x;
	int by_cy = c.y - b.y;
	int bz_cz = c.z - b.z;
	float cos = ((ax_cx*bx_cx) + (ay_cy*by_cy) + (az_cz*bz_cz)) / ((sqrt((ax_cx*ax_cx) + (ay_cy*ay_cy) + (az_cz*az_cz))*sqrt((bx_cx*bx_cx) + (by_cy*by_cy) + (bz_cz*bz_cz))));
	float angle = acosf(cos);
	if (cos > -1.0 && cos < 0.0){
		angle = PI - angle;
	}
	return angle;
}

//二つのベクトルのスタートが離れている場合（腰の角度算出に使用）
//膝がAorC,骨盤がBorD
float evaluate_seperated_angle(Point3f pA, Point3f pB, Point3f pC, Point3f pD)
{
	int ax_bx = pA.x - pB.x;
	int ay_by = pA.y - pB.y;
	int az_bz = pA.z - pB.z;
	int cx_dx = pC.x - pD.x;
	int cy_dy = pC.y - pD.y;
	int cz_dz = pC.z - pD.z;
	float cos = ((ax_bx*cx_dx) + (ay_by*cy_dy) + (az_bz*cz_dz)) / ((sqrt((ax_bx*ax_bx) + (ay_by*ay_by) + (az_bz*az_bz))*sqrt((cx_dx*cx_dx) + (cy_dy*cy_dy) + (cz_dz*cz_dz))));
	float angle = acosf(cos);
	if (cos > -1.0 && cos < 0.0){
		angle = PI - angle;
	}
	return angle;
}