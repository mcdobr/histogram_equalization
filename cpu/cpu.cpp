// cpu.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <chrono> 

using namespace std::chrono;
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    string image_path(argv[1]);

    Mat src = imread(image_path, IMREAD_COLOR);
    if (src.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }

    cvtColor(src, src, COLOR_BGR2GRAY);
    Mat dst;
    //Time before the histogram is computed
    auto start = high_resolution_clock::now();
    equalizeHist(src, dst);
    auto stop = high_resolution_clock::now();
    //The time after the histogram has been computed

    // Time difference in microseconds
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Execution time:"<< duration.count() << endl;

    imshow("Source image", src);
    imshow("Equalized Image", dst);
    waitKey();
    return 0;
}