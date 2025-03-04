
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char ** argv) {
	cv::Mat src = cv::imread("C:/Users/滕唯先/OneDrive/桌面/01.png", IMREAD_GRAYSCALE);
	
		if (src.empty()) {
			printf("could not load image...");
			return -1;
		}
	cv::imshow("my_window", src);
		cv::waitKey(0);
		destroyAllWindows();
	system("pause");
	return 0;
}
