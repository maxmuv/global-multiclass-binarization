#include "image_processing_unit.h"

void MultiClassOtsuUnit::CreateNormHistogram() {
  // ToDo: Check that image is gray
  int hist_size = 256;
  float range[] = {0, 256};
  const float* hist_range[] = {range};
  bool uniform = true, accumulate = false;
  cv::calcHist(&m_img, 1, 0, cv::Mat(), m_hist, 1, &hist_size, hist_range,
               uniform, accumulate);
  float sum = 0.0f;
  for (int i = 0; i < m_hist.rows; i++) {
    sum += m_hist.at<float>(i);
  }
  for (int i = 0; i < m_hist.rows; i++) {
    m_hist.at<float>(i) /= sum;
  }
}

void MultiClassOtsuUnit::GenerateAllPossibleThresholds(
    std::vector<std::vector<uchar>>& all_possible_thresholds,
    std::vector<uchar> thresholds, int level) const {
  if (level >= m_levels - 1) return;
  const int start = (0 == level) ? 0 : thresholds[level - 1];

  for (int i = start; i < 256; i++) {
    thresholds[level] = i;
    GenerateAllPossibleThresholds(all_possible_thresholds, thresholds,
                                  level + 1);
    if (m_levels - 2 == level) all_possible_thresholds.push_back(thresholds);
  }
}

void MultiClassOtsuUnit::CalculateIntroClassVariance(
    float& var, const std::vector<uchar>& cand_thresholds) const {
  float i_var = 0.0;

  std::vector<float> prob_distr(m_levels);
  int start = 0;
  int end = 0;
  for (int cl_id = 0; cl_id < m_levels; cl_id++) {
    if (cand_thresholds.size() == cl_id)
      end = 256;
    else
      end = cand_thresholds[cl_id];
    prob_distr[cl_id] = 0.0;
    for (int j = start; j < end; j++) {
      prob_distr[cl_id] += m_hist.at<float>(j);
    }
    start = end;
  }
  float sum = 0.0;
  for (const auto prob : prob_distr) {
    sum += prob;
  }
  assert(fabs(1.0f - sum) < 0.000001f);

  start = 0;
  end = 0;
  std::vector<float> means(m_levels);
  for (int cl_id = 0; cl_id < m_levels; cl_id++) {
    if (cand_thresholds.size() == cl_id)
      end = 256;
    else
      end = cand_thresholds[cl_id];
    means[cl_id] = 0.0;
    for (int j = start; j < end; j++) {
      means[cl_id] += (0 == prob_distr[cl_id])
                          ? 0
                          : j * m_hist.at<float>(j) / prob_distr[cl_id];
    }
    start = end;
  }

  start = 0;
  end = 0;
  std::vector<float> vars(m_levels);
  for (int cl_id = 0; cl_id < m_levels; cl_id++) {
    if (cand_thresholds.size() == cl_id)
      end = 256;
    else
      end = cand_thresholds[cl_id];
    vars[cl_id] = 0.0;
    for (int j = start; j < end; j++) {
      vars[cl_id] += (0 == prob_distr[cl_id])
                         ? 0
                         : (j - means[cl_id]) * (j - means[cl_id]) *
                               m_hist.at<float>(j) / prob_distr[cl_id];
    }
    start = end;
  }

  for (int cl_id = 0; cl_id < m_levels; cl_id++) {
    i_var += prob_distr[cl_id] * vars[cl_id];
  }

  var = i_var;
}

// ToDo: Check that hist exist
void MultiClassOtsuUnit::SearchThresholds(std::vector<uchar>& thresholds,
                                          double& res_var) const {
  std::vector<std::vector<uchar>> all_possible_thresholds;
  std::vector<uchar> thresholds_for_func(m_levels - 1);
  GenerateAllPossibleThresholds(all_possible_thresholds, thresholds_for_func,
                                0);
  float min = 100000000.0;
  size_t best_id = 0;
  for (size_t t_id = 0; t_id < all_possible_thresholds.size(); t_id++) {
    float var;
    CalculateIntroClassVariance(var, all_possible_thresholds[t_id]);
    if (var < min) {
      min = var;
      best_id = t_id;
    }
  }
  thresholds = all_possible_thresholds[best_id];
  res_var = min;
}

/// According to tresholds and gray level this method choose a result color
cv::Vec3b ChooseColor(const uchar bin, std::vector<uchar> thrs) {
  thrs.emplace_back(255);
  switch (thrs.size()) {
    case 2:
      if (bin <= thrs[0])
        return cv::Vec3b(255, 0, 0);
      else
        return cv::Vec3b(0, 255, 0);
      break;
    case 3:
      if (bin <= thrs[0])
        return cv::Vec3b(255, 0, 0);
      else if (bin <= thrs[1])
        return cv::Vec3b(0, 255, 0);
      else
        return cv::Vec3b(0, 0, 255);
      break;
    case 4:
      if (bin <= thrs[0])
        return cv::Vec3b(255, 0, 0);
      else if (bin <= thrs[1])
        return cv::Vec3b(0, 255, 0);
      else if (bin <= thrs[2])
        return cv::Vec3b(0, 0, 255);
      else
        return cv::Vec3b(255, 255, 255);
      break;
    default:
      throw std::runtime_error("too many thresholds\n");
  }
}

void MultiClassOtsuUnit::BinarizeImage(std::vector<uchar>& thresholds,
                                       cv::Mat& gray_img,
                                       cv::Mat& bin_img) const {
  cv::cvtColor(gray_img, bin_img, cv::COLOR_GRAY2BGR);
  for (int y = 0; y < gray_img.rows; y++) {
    for (int x = 0; x < gray_img.cols; x++) {
      uchar val = gray_img.at<uchar>(y, x);
      cv::Scalar color = ChooseColor(gray_img.at<uchar>(y, x), thresholds);
      bin_img.at<cv::Vec3b>(y, x) =
          ChooseColor(gray_img.at<uchar>(y, x), thresholds);
    }
  }
}

void MultiClassOtsuUnit::DrawHist(std::unique_ptr<cv::Mat>& histImage) const {
  int bin_w = static_cast<int>(1.0 * 256 / m_hist.rows);
  int hist_h = 400;
  histImage.reset(new cv::Mat(hist_h, 256, CV_8UC3, cv::Scalar(0, 0, 0)));
  cv::Mat hist;
  normalize(m_hist, hist, 0, histImage->rows, cv::NORM_MINMAX, -1, cv::Mat());
  for (int i = 0; i < m_hist.rows; ++i) {
    line(*histImage.get(),
         cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
         cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
         cv::Scalar(ChooseColor(i, m_thrs)), 2, 8, 0);
  }
}

void MultiClassOtsuUnit::Process(cv::Mat& gray_img, cv::Mat& bin_img) {
  CreateNormHistogram();
  std::vector<uchar> thrs;
  double var;
  SearchThresholds(thrs, var);
  m_thrs = thrs;
  BinarizeImage(thrs, gray_img, bin_img);
}
