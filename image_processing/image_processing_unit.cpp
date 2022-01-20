#include "image_processing_unit.h"

void MultiClassOtsuUnit::CreateHistogram() {
  // ToDo: Check that image is gray
  int hist_size = 256;
  float range[] = {0, 256};
  const float* hist_range[] = {range};
  bool uniform = true, accumulate = false;
  cv::Mat hist;
  cv::calcHist(&m_img, 1, 0, cv::Mat(), hist, 1, &hist_size, hist_range,
               uniform, accumulate);
  // ToDo: Make test for histogram
  std::cout << "hist = " << std::endl << " " << hist << std::endl << std::endl;
}
