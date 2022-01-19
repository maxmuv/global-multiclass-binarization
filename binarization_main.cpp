#include <opencv2/opencv.hpp>

int main() {
  cv::Mat img = cv::imread("/home/mikhail/binarization/data/Lena.jpg",
                           cv::IMREAD_GRAYSCALE);
  cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
  if (!img.empty()) cv::imshow("Gray picture", img);
  cv::waitKey();
}
