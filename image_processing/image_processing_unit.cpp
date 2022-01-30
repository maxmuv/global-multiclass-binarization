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
    std::vector<uchar> thresholds, int level) {
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
    float& var, const std::vector<uchar>& cand_thresholds) {
  float i_var = 0.0;

  std::vector<float> prob_distr(m_levels);
  int start = 0;
  int end = 0;
  for (size_t cl_id = 0; cl_id < m_levels; cl_id++) {
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
  for (size_t cl_id = 0; cl_id < m_levels; cl_id++) {
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
  for (size_t cl_id = 0; cl_id < m_levels; cl_id++) {
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

  for (size_t cl_id = 0; cl_id < m_levels; cl_id++) {
    i_var += prob_distr[cl_id] * vars[cl_id];
  }

  var = i_var;
}

// ToDo: Check that hist exist
void MultiClassOtsuUnit::SearchThresholds(std::vector<uchar>& thresholds,
                                          double& res_var) {
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

void MultiClassOtsuUnit::BinarizeImage(std::vector<uchar>& thresholds,
                                       cv::Mat& gray_img, cv::Mat& bin_img) {
  gray_img.copyTo(bin_img);

  for (int y = 0; y < gray_img.rows; y++) {
    for (int x = 0; x < gray_img.cols; x++) {
      int start = 0;
      int end = 0;
      for (int cl_id = 0; cl_id <= m_levels; cl_id++) {
        if (thresholds.size() == cl_id)
          end = 255;
        else
          end = thresholds[cl_id];
        if ((start <= gray_img.at<uchar>(y, x)) &&
            (gray_img.at<uchar>(y, x) <= end)) {
          bin_img.at<uchar>(y, x) = cl_id * static_cast<int>(255 / m_levels);
        }
        start = end;
      }
    }
  }
}

void MultiClassOtsuUnit::Process(cv::Mat& gray_img, cv::Mat& bin_img) {
  CreateNormHistogram();
  std::vector<uchar> thrs;
  double var;
  SearchThresholds(thrs, var);
  BinarizeImage(thrs, gray_img, bin_img);
}
