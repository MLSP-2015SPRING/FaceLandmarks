#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "asmlib-opencv/asmmodel.h"
#include <string>
#include <vector>

using std::vector;
using std::string;
using cv::Mat;
using cv::VideoCapture;

int main(int argc, char** argv )
{
    if(argc != 3)
    {
        return 1;
    }

    string arg = argv[1];
    cv::namedWindow("Image", CV_WINDOW_NORMAL);
    vector<cv::Mat> faces;

    Mat test_img;
    for (int i = 1; i < 2; i++)
    {
        string path = arg + cv::format("subject%02d", i) + ".sad";
        VideoCapture cap(path);
        Mat image;
        cap >> image;

        test_img = image;
        if(!image.empty()) {
            faces.push_back(image);
        }
    }

    StatModel::ASMModel model(argv[2]);
    vector<vector<Point_<int>>> landmarks;
    vector<Mat>::iterator iter;
    for (iter = faces.begin(); iter != faces.end(); iter++) {
        StatModel::ASMFitResult res = model.fit(*iter);
        vector<Point_<int>> face_landmarks;
        res.toPointList(face_landmarks);
        landmarks.push_back(face_landmarks);
        std::cout << face_landmarks.size() << std::endl;

        vector<Point>::iterator piter;
        for (piter = face_landmarks.begin(); piter != face_landmarks.end(); piter++) {
            Point center = Point(piter->x, piter->y);
            circle(test_img, center, 1, CV_RGB(255,255,0),3);
        }
    }

    cv::imshow("Image", test_img);
    getchar();
    getchar();

    return 0;
}