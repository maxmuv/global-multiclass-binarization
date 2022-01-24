#include <opencv2/opencv.hpp>
#include "image_processing/image_processing_unit.h"

int main() {
  cv::Mat img = cv::imread("/home/mikhail/binarization/data/Lena.jpg",
                           cv::IMREAD_GRAYSCALE);
  // cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
  MultiClassOtsuUnit unit1(img, 3);
  // TODO: exit working
  unit1.CreateNormHistogram();
  std::vector<uchar> thrs;
  unit1.SearchThresholds(thrs);
  std::cout << "Best thr: ";
  for (auto thr : thrs) {
    std::cout << (int)thr << " ";
  }
}
