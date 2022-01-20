#include <opencv2/opencv.hpp>
#include "image_processing/image_processing_unit.h"

int main() {
  cv::Mat img = cv::imread("/home/mikhail/binarization/data/Lena.jpg",
                           cv::IMREAD_GRAYSCALE);
  // cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
  MultiClassOtsuUnit unit1(img);
  // TODO: exit working
  unit1.CreateHistogram();
}
