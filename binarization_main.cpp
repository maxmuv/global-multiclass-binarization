#include <fstream>
#include <opencv2/opencv.hpp>
#include "image_processing/image_processing_unit.h"

struct AdditionalInfo {
  std::string input_path;
  std::string output_path;
  std::string file;
  int levels;
};

void ParseArguments(std::string& str, AdditionalInfo& info) {
  try {
    auto it = str.find(" ");
    info.input_path = str.substr(0, it);
    str = str.substr(++it);
    it = str.find(" ");
    info.output_path = str.substr(0, it);
    str = str.substr(++it);
    info.levels = std::stoi(str);
    str.clear();
  } catch (...) {
    throw std::runtime_error("Cannot parse file");
  }
}

void ParseArguments(int argc, char** argv, AdditionalInfo& info) {
  for (int i = 1; i < argc; i += 2) {
    if ("-i" == std::string(argv[i])) {
      info.input_path = std::string(argv[i + 1]);
    } else if ("-o" == std::string(argv[i])) {
      info.output_path = std::string(argv[i + 1]);
    } else if ("-l" == std::string(argv[i])) {
      info.levels = std::stoi(argv[i + 1]);
    } else if ("-f" == std::string(argv[i])) {
      info.file = std::string(argv[i + 1]);
    }
  }
  if (((info.input_path.empty()) || (info.output_path.empty()) ||
       ((info.levels < 2) && (0 != info.levels)) || (info.levels > 4)) &&
      (info.file.empty()))
    throw std::runtime_error(
        "Use program:\n path/to/binarization_main -i path/to/gray/image -o "
        "path/to/output/image -l "
        "/number/of/classes\nor\n path/to/binarization_main -f path/to/file");
}

/** @brief App.
* Usage:
* binarization -i path\to\input\gray\image -o path\to\output\binarized\image -l levels
* or
* binarization -f path\to\file 
*/
int main(int argc, char** argv) {
  try {
    AdditionalInfo add_info;
    ParseArguments(argc, argv, add_info);
    if (add_info.file.empty()) {
      cv::Mat img = cv::imread(add_info.input_path, cv::IMREAD_GRAYSCALE);
      if (img.empty()) throw std::runtime_error("Cannot read input image");
      MultiClassOtsuUnit bin_unit(img, add_info.levels);
      cv::Mat bin;
      if (add_info.levels != 0) {
        bin_unit.Process(img, bin);
      } else {
        img.copyTo(bin);
      }
      cv::cvtColor(bin, bin, cv::COLOR_GRAY2BGR);
      if (!cv::imwrite(add_info.output_path, bin))
        throw std::runtime_error("Cannot save image");
    } else {
      std::string s;
      std::ifstream in(add_info.file);
      std::getline(in, s);
      while (!s.empty()) {
        AdditionalInfo info;
        ParseArguments(s, info);
        cv::Mat img = cv::imread(info.input_path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) throw std::runtime_error("Cannot read input image");
        MultiClassOtsuUnit bin_unit(img, info.levels);
        cv::Mat bin;
        if (info.levels != 0) {
          bin_unit.Process(img, bin);
        } else {
          img.copyTo(bin);
        }
        cv::cvtColor(bin, bin, cv::COLOR_GRAY2BGR);
        if (!cv::imwrite(info.output_path, bin))
          throw std::runtime_error("Cannot save image");
        std::getline(in, s);
      }
    }
  } catch (std::exception& w) {
    std::cout << w.what();
  } catch (...) {
  }
}
