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
    float value = m_hist.at<float>(i);
    if (value < m_min) m_min = value;
    if (value > m_max) m_max = value;
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
    float& var, const std::vector<uchar>& cand_thresholds, bool print) const {
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
  if (print) {
    std::cout << "Info for image:" << std::endl;
    for (int cl_id = 0; cl_id < m_levels; ++cl_id) {
      std::cout << "Class " << std::to_string(cl_id)
                << " - expected value: " << std::to_string(means[cl_id])
                << std::endl;
    }
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
    float val = prob_distr[cl_id] * vars[cl_id];
    if (print)
      std::cout << "Class " << std::to_string(cl_id)
                << " - variance: " << std::to_string(vars[cl_id]) << std::endl;
    i_var += val;
  }

  var = i_var;
  if (print) {
    std::cout << "Introclass variance: " << std::to_string(var) << std::endl;
  }
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
  int hist_x_begin = 50;
  int hist_y_begin = 50;
  histImage.reset(new cv::Mat(hist_y_begin + hist_h + 60, 256 + hist_x_begin,
                              CV_8UC3, cv::Scalar(0, 0, 0)));
  cv::Mat hist;
  normalize(m_hist, hist, 0, histImage->rows - 60 - hist_y_begin,
            cv::NORM_MINMAX, -1, cv::Mat());
  line(*histImage.get(),
       cv::Point(hist_x_begin - 2, hist_y_begin + hist_h + 25),
       cv::Point(256 + hist_x_begin, hist_y_begin + hist_h + 25),
       cv::Scalar(255, 255, 255), 2, 8, 0);
  putText(*histImage.get(), "x, gray level",
          cv::Point(140, hist_y_begin + hist_h + 55), cv::FONT_HERSHEY_SIMPLEX,
          0.25, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  line(*histImage.get(), cv::Point(hist_x_begin - 2, 0),
       cv::Point(hist_x_begin - 2, hist_y_begin + hist_h + 25),
       cv::Scalar(255, 255, 255), 2, 8, 0);
  putText(*histImage.get(), "y, amount", cv::Point(0, 10),
          cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(255, 255, 255), 1,
          cv::LINE_AA);
  putText(*histImage.get(), "of pixels", cv::Point(0, 20),
          cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(255, 255, 255), 1,
          cv::LINE_AA);
  int y_step = (m_max - m_min) / 10;
  for (int i = m_min; i < m_max; i += y_step) {
    line(*histImage.get(),
         cv::Point(hist_x_begin - 5, hist_y_begin + hist_h -
                                         cvRound(i * hist_h / (m_max - m_min))),
         cv::Point(hist_x_begin, hist_y_begin + hist_h -
                                     cvRound(i * hist_h / (m_max - m_min))),
         cv::Scalar(255, 255, 255), 2, 8, 0);
    putText(*histImage.get(), std::to_string(i),
            cv::Point(0, hist_y_begin + hist_h -
                             cvRound(i * hist_h / (m_max - m_min))),
            cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1,
            cv::LINE_AA);
  }
  for (int i = 0; i < m_hist.rows; ++i) {
    line(*histImage.get(),
         cv::Point(hist_x_begin + bin_w * (i - 1),
                   hist_y_begin + hist_h - cvRound(hist.at<float>(i - 1))),
         cv::Point(hist_x_begin + bin_w * (i),
                   hist_y_begin + hist_h - cvRound(hist.at<float>(i))),
         cv::Scalar(ChooseColor(i, m_thrs)), 2, 8, 0);
    if (i % 30 == 0) {
      std::string text = std::to_string(i);
      switch (text.size()) {
        case 1:
          line(*histImage.get(),
               cv::Point(hist_x_begin + i * bin_w, hist_y_begin + hist_h + 30),
               cv::Point(hist_x_begin + i * bin_w, hist_y_begin + hist_h + 20),
               cv::Scalar(255, 255, 255), 2, 8, 0);
          putText(
              *histImage.get(), text,
              cv::Point(hist_x_begin + i * bin_w, hist_y_begin + hist_h + 45),
              cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1,
              cv::LINE_AA);
          break;
        case 2:
          line(*histImage.get(),
               cv::Point(hist_x_begin + i * bin_w, hist_y_begin + hist_h + 30),
               cv::Point(hist_x_begin + i * bin_w, hist_y_begin + hist_h + 20),
               cv::Scalar(255, 255, 255), 2, 8, 0);
          putText(*histImage.get(), text,
                  cv::Point(hist_x_begin + i * bin_w - 8,
                            hist_y_begin + hist_h + 45),
                  cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1,
                  cv::LINE_AA);
          break;
        case 3:
          line(*histImage.get(),
               cv::Point(hist_x_begin + i * bin_w, hist_y_begin + hist_h + 30),
               cv::Point(hist_x_begin + i * bin_w, hist_y_begin + hist_h + 20),
               cv::Scalar(255, 255, 255), 2, 8, 0);
          putText(*histImage.get(), text,
                  cv::Point(hist_x_begin + i * bin_w - 10,
                            hist_y_begin + hist_h + 45),
                  cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1,
                  cv::LINE_AA);
          break;
      }
    }
  }
  float tmp = 0;
  CalculateIntroClassVariance(tmp, m_thrs, true);
}

void MultiClassOtsuUnit::DrawPlots(
    std::vector<std::unique_ptr<cv::Mat>>& plotsImage) const {
  for (int th_id = 0; th_id < m_thrs.size(); th_id++) {
    int begin = 0;
    int end = 255;
    if (th_id != 0) begin = m_thrs[th_id - 1];
    if (th_id < m_thrs.size() - 1) end = m_thrs[th_id + 1];
    if (end == begin) throw std::runtime_error("Something wrong");
    int hist_w = 400;
    int hist_h = 400;
    int plot_x_begin = 50;
    int plot_y_begin = 50;
    plotsImage.emplace_back(new cv::Mat(hist_h + plot_y_begin + 60,
                                        hist_w + plot_x_begin, CV_8UC3,
                                        cv::Scalar(0, 0, 0)));
    int min = 10000000;
    int max = 0;
    std::vector<float> vars;
    for (int i = begin; i < end; ++i) {
      float val;
      std::vector<uchar> thrs = m_thrs;
      thrs[th_id] = i;
      CalculateIntroClassVariance(val, thrs);
      if (val < min) min = val;
      if (val > max) max = val;
      vars.push_back(val);
    }
    line(*plotsImage.back().get(),
         cv::Point(plot_x_begin - 2, plot_y_begin + hist_h + 25),
         cv::Point(hist_w + plot_x_begin, plot_y_begin + hist_h + 25),
         cv::Scalar(255, 255, 255), 2, 8, 0);
    putText(*plotsImage.back().get(),
            "x," + std::to_string(th_id + 1) + " threshold",
            cv::Point(hist_w / 2 + plot_x_begin, plot_y_begin + hist_h + 55),
            cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(255, 255, 255), 1,
            cv::LINE_AA);
    line(*plotsImage.back().get(), cv::Point(plot_x_begin - 2, 0),
         cv::Point(plot_x_begin - 2, plot_y_begin + hist_h + 25),
         cv::Scalar(255, 255, 255), 2, 8, 0);
    putText(*plotsImage.back().get(), "intro-class", cv::Point(0, 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(255, 255, 255), 1,
            cv::LINE_AA);
    putText(*plotsImage.back().get(), "variance", cv::Point(0, 20),
            cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(255, 255, 255), 1,
            cv::LINE_AA);
    int y_step = (max - min) / 10;
    for (int i = min; i < max; i += y_step) {
      line(*plotsImage.back().get(),
           cv::Point(plot_x_begin - 5,
                     plot_y_begin + hist_h -
                         cvRound((i - min) * hist_h / (max - min))),
           cv::Point(plot_x_begin,
                     plot_y_begin + hist_h -
                         cvRound((i - min) * hist_h / (max - min))),
           cv::Scalar(255, 255, 255), 2, 8, 0);
      putText(*plotsImage.back().get(), std::to_string(i),
              cv::Point(0, plot_y_begin + hist_h -
                               cvRound((i - min) * hist_h / (max - min))),
              cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1,
              cv::LINE_AA);
    }
    float bin_w = 1.0 * hist_w / (end - begin);
    for (int i = begin, j = 0; i < end - 1; ++i, ++j) {
      line(*plotsImage.back().get(),
           cv::Point(plot_x_begin +
                         cvRound(1.0 * (i - begin) / (end - begin) * hist_w),
                     plot_y_begin + hist_h -
                         cvRound(1.0 * (vars[j] - min) / (max - min) * hist_h)),
           cv::Point(
               plot_x_begin +
                   cvRound(1.0 * (i + 1 - begin) / (end - begin) * hist_w),
               plot_y_begin + hist_h -
                   cvRound(1.0 * (vars[j + 1] - min) / (max - min) * hist_h)),
           cv::Scalar(255, 255, 255), 2, 8, 0);
      if (i % 30 == 0) {
        std::string text = std::to_string(i);
        switch (text.size()) {
          case 1:
            line(*plotsImage.back().get(),
                 cv::Point(plot_x_begin + cvRound(j * bin_w),
                           plot_y_begin + hist_h + 30),
                 cv::Point(plot_x_begin + cvRound(j * bin_w),
                           plot_y_begin + hist_h + 20),
                 cv::Scalar(255, 255, 255), 2, 8, 0);
            putText(*plotsImage.back().get(), text,
                    cv::Point(plot_x_begin + cvRound(j * bin_w),
                              plot_y_begin + hist_h + 45),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1,
                    cv::LINE_AA);
            break;
          case 2:
            line(*plotsImage.back().get(),
                 cv::Point(plot_x_begin + cvRound(j * bin_w),
                           plot_y_begin + hist_h + 30),
                 cv::Point(plot_x_begin + cvRound(j * bin_w),
                           plot_y_begin + hist_h + 20),
                 cv::Scalar(255, 255, 255), 2, 8, 0);
            putText(*plotsImage.back().get(), text,
                    cv::Point(plot_x_begin + cvRound(j * bin_w) - 8,
                              plot_y_begin + hist_h + 45),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1,
                    cv::LINE_AA);
            break;
          case 3:
            line(*plotsImage.back().get(),
                 cv::Point(plot_x_begin + cvRound(j * bin_w),
                           plot_y_begin + hist_h + 30),
                 cv::Point(plot_x_begin + cvRound(j * bin_w),
                           plot_y_begin + hist_h + 20),
                 cv::Scalar(255, 255, 255), 2, 8, 0);
            putText(*plotsImage.back().get(), text,
                    cv::Point(plot_x_begin + cvRound(j * bin_w) - 10,
                              plot_y_begin + hist_h + 45),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1,
                    cv::LINE_AA);
            break;
        }
      }
    }
    float val;
    CalculateIntroClassVariance(val, m_thrs);
    for (int i = 0; i < 5; ++i) {
      cv::circle(
          *plotsImage.back().get(),
          cv::Point(plot_x_begin + cvRound(1.0 * (m_thrs[th_id] - begin) /
                                           (end - begin) * hist_w),
                    plot_y_begin + hist_h -
                        cvRound(1.0 * (val - min) / (max - min) * hist_h)),
          i, cv::Scalar(255, 0, 0));
    }
  }
}

void MultiClassOtsuUnit::Process(cv::Mat& gray_img, cv::Mat& bin_img) {
  CreateNormHistogram();
  std::vector<uchar> thrs;
  double var;
  SearchThresholds(thrs, var);
  m_thrs = thrs;
  for (int i = 0; i < thrs.size(); ++i)
    std::cout << "Threshold " << i + 1 << " : " << static_cast<int>(thrs[i])
              << std::endl;
  BinarizeImage(thrs, gray_img, bin_img);
}
