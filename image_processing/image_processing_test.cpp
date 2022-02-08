#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <opencv2/opencv.hpp>
#include "image_processing_unit.h"

/// Test: a case with zero varience.
TEST_CASE("n_levels_gray_image") {
  srand(100);
  int width = 400;
  int height = 250;
  std::vector<int> v1{0, 1, 2, 3};
  std::vector<int> v2{10, 50, 65, 113};

  for (int i = 2; i < 5; ++i) {
    cv::Mat gray_image_1(height, width, CV_8UC1, cv::Scalar(0));
    cv::Mat gray_image_2(height, width, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < gray_image_1.rows; ++y) {
      for (int x = 0; x < gray_image_1.cols; ++x) {
        gray_image_1.at<uchar>(y, x) = v1[rand() % i];
        gray_image_2.at<uchar>(y, x) = v2[rand() % i];
      }
    }
    MultiClassOtsuUnit unit1(gray_image_1, i);
    MultiClassOtsuUnit unit2(gray_image_2, i);
    unit1.CreateNormHistogram();
    unit2.CreateNormHistogram();
    std::vector<uchar> thrs;
    double var = 3;
    unit1.SearchThresholds(thrs, var);
    CHECK_EQ(var, doctest::Approx(0.0));
    unit2.SearchThresholds(thrs, var);
    CHECK_EQ(var, doctest::Approx(0.0));
  }
}

/// Tests: with non-zero variance.
TEST_CASE("3_2_levels_gray_image") {
  int width = 120;
  int height = 120;
  cv::Mat gray_img(height, width, CV_8UC1, cv::Scalar(0));
  for (int y = 0; y < gray_img.rows; ++y) {
    for (int x = 0; x < gray_img.cols; x += 6) {
      gray_img.at<uchar>(y, x) = 0u;
      gray_img.at<uchar>(y, x + 1) = 1u;
      gray_img.at<uchar>(y, x + 2) = 2u;
      gray_img.at<uchar>(y, x + 3) = 253u;
      gray_img.at<uchar>(y, x + 4) = 254u;
      gray_img.at<uchar>(y, x + 5) = 255u;
    }
  }
  MultiClassOtsuUnit unit(gray_img, 2);
  unit.CreateNormHistogram();
  std::vector<uchar> thr;
  double var;
  unit.SearchThresholds(thr, var);
  CHECK_EQ(var, doctest::Approx(2.0 / 3));
}

TEST_CASE("3_3_levels_gray_image") {
  int width = 360;
  int height = 120;
  cv::Mat gray_img(height, width, CV_8UC1, cv::Scalar(0));
  for (int y = 0; y < gray_img.rows; ++y) {
    for (int x = 0; x < gray_img.cols; x += 9) {
      gray_img.at<uchar>(y, x) = 0u;
      gray_img.at<uchar>(y, x + 1) = 1u;
      gray_img.at<uchar>(y, x + 2) = 2u;
      gray_img.at<uchar>(y, x + 3) = 253u;
      gray_img.at<uchar>(y, x + 4) = 254u;
      gray_img.at<uchar>(y, x + 5) = 255u;
      gray_img.at<uchar>(y, x + 6) = 120u;
      gray_img.at<uchar>(y, x + 7) = 121u;
      gray_img.at<uchar>(y, x + 8) = 122u;
    }
  }
  MultiClassOtsuUnit unit(gray_img, 3);
  unit.CreateNormHistogram();
  std::vector<uchar> thr;
  double var;
  unit.SearchThresholds(thr, var);
  CHECK_EQ(var, doctest::Approx(2.0 / 3));
}

TEST_CASE("3_4_levels_gray_image") {
  int width = 360;
  int height = 50;
  cv::Mat gray_img(height, width, CV_8UC1, cv::Scalar(0));
  for (int y = 0; y < gray_img.rows; ++y) {
    for (int x = 0; x < gray_img.cols; x += 12) {
      gray_img.at<uchar>(y, x) = 0u;
      gray_img.at<uchar>(y, x + 1) = 1u;
      gray_img.at<uchar>(y, x + 2) = 2u;
      gray_img.at<uchar>(y, x + 3) = 253u;
      gray_img.at<uchar>(y, x + 4) = 254u;
      gray_img.at<uchar>(y, x + 5) = 255u;
      gray_img.at<uchar>(y, x + 6) = 80u;
      gray_img.at<uchar>(y, x + 7) = 81u;
      gray_img.at<uchar>(y, x + 8) = 82u;
      gray_img.at<uchar>(y, x + 9) = 130u;
      gray_img.at<uchar>(y, x + 10) = 131u;
      gray_img.at<uchar>(y, x + 11) = 132u;
    }
  }
  MultiClassOtsuUnit unit(gray_img, 4);
  unit.CreateNormHistogram();
  std::vector<uchar> thr;
  double var;
  unit.SearchThresholds(thr, var);
  CHECK_EQ(var, doctest::Approx(2.0 / 3));
}

/// Test: histogram creation.
TEST_CASE("create_norm_hist") {
  srand(15);
  int width = 100;
  int height = 100;
  std::map<uchar, float> hand_hist;
  for (int i = 0; i < 256; ++i) hand_hist[i] = 0.0;
  cv::Mat gray_img(width, height, CV_8UC1, cv::Scalar(0));
  for (int y = 0; y < gray_img.rows; ++y) {
    for (int x = 0; x < gray_img.cols; ++x) {
      uchar val = rand() % 256;
      hand_hist[val] += 1;
      gray_img.at<char>(y, x) = val;
    }
  }
  MultiClassOtsuUnit unit(gray_img, 2);
  unit.CreateNormHistogram();
  int square = width * height;
  for (int i = 0; i < 256; ++i) {
    CHECK_EQ(1.0f * hand_hist[i] / square, unit.m_hist.at<float>(i));
  }
}
