#ifndef IMAGE_PROCESSING_UNIT_H
#define IMAGE_PROCESSING_UNIT_H

#include <opencv2/opencv.hpp>

class ImageProcessingUnit {
 protected:
  ImageProcessingUnit(const cv::Mat& img, const int levels) : m_levels(levels) {
    img.copyTo(m_img);
  }
  void UpdateImage(const cv::Mat& img) { m_img = img.clone(); }
  virtual void Process(cv::Mat& gray_img, cv::Mat& bin_img) = 0;
  cv::Mat m_img;
  const int m_levels;

 private:
  ImageProcessingUnit() = delete;
  ImageProcessingUnit(const ImageProcessingUnit&) = delete;
};

class MultiClassOtsuUnit : public ImageProcessingUnit {
 public:
  MultiClassOtsuUnit(const cv::Mat& img, const int levels)
      : ImageProcessingUnit(img, levels) {}

  void Process(cv::Mat& gray_img, cv::Mat& bin_img) override;
  void CreateNormHistogram();
  void SearchThresholds(std::vector<uchar>& thresholds);
  void BinarizeImage(std::vector<uchar>& thresholds, cv::Mat& gray_img,
                     cv::Mat& bin_img);

 private:
  void GenerateAllPossibleThresholds(
      std::vector<std::vector<uchar>>& all_possible_thresholds,
      std::vector<uchar> thresholds, int level);
  void CalculateIntroClassVariance(float& var,
                                   const std::vector<uchar>& cand_thresholds);

  cv::Mat m_hist;
};

#endif
