#ifndef IMAGE_PROCESSING_UNIT_H
#define IMAGE_PROCESSING_UNIT_H

#include <opencv2/opencv.hpp>

class ImageProcessingUnit {
 protected:
  ImageProcessingUnit(const cv::Mat& img) { m_img = img.clone(); }
  void UpdateImage(const cv::Mat& img) { m_img = img.clone(); }

  cv::Mat m_img;

 private:
  ImageProcessingUnit() = delete;
  ImageProcessingUnit(const ImageProcessingUnit&) = delete;
};

class MultiClassOtsuUnit : public ImageProcessingUnit {
 public:
  MultiClassOtsuUnit(const cv::Mat& img) : ImageProcessingUnit(img) {}

  void CreateHistogram();
};

#endif
