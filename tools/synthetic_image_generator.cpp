#include <opencv2/opencv.hpp>
#include <random>

void CreateImage(int n, std::unique_ptr<cv::Mat> &synthetic_image) {
  std::vector<std::mt19937> mersenne_tw_engines;
  std::vector<std::normal_distribution<>> uni_distrs;
  std::vector<int> exp_value = {85, 100, 115, 130};
  for (int i = 0; i < n; ++i) {
    mersenne_tw_engines.emplace_back(i);
    uni_distrs.emplace_back(0, 4);
  }
  for (int y = 0; y < synthetic_image.get()->rows; ++y) {
    int i = y / (synthetic_image.get()->rows / n + 1);
    std::vector<uchar> vals;
    for (int x = 0; x < synthetic_image.get()->cols; ++x) {
      vals.push_back(exp_value[i] + (uni_distrs[i])(mersenne_tw_engines[i]));
    }
    // std::stable_sort(vals.begin(), vals.end());
    for (int x = 0; x < synthetic_image.get()->cols; ++x) {
      synthetic_image.get()->at<uchar>(x, y) = vals[x];
    }
  }
}

int main() {
  int width = 400;
  int height = 400;
  for (int cl_num = 2; cl_num < 5; ++cl_num) {
    std::unique_ptr<cv::Mat> synthetic_image(
        new cv::Mat(height, width, CV_8UC1, cv::Scalar(0)));
    CreateImage(cl_num, synthetic_image);
    cv::cvtColor(*synthetic_image.get(), *synthetic_image.get(),
                 cv::COLOR_GRAY2BGR);
    cv::imwrite(
        "demo/src/synthetic_" + std::to_string(cl_num) + "_class_image.jpg",
        *synthetic_image.get());
  }
}
