#ifndef IMAGE_PROCESSING_UNIT_H
#define IMAGE_PROCESSING_UNIT_H

#include <opencv2/opencv.hpp>

/**
 * @brief The ImageProcessingUnit class.
 * Это базовый класс для классов, которые бинаризуют серое изображение. Он имеет
 * два метода: UpdateImage и Process. Process переопределяется
 * классом-наследником для конкретной реализации алгоритма бинаризации. Данный
 * класс содержит исходное изображение и количество классов.
 */

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

/**
 * @brief The MultiClassOtsuUnit class.
 * Это класс наследник ImageProcessingUnit. Он реализует мультиклассовый метод
 * Оцу, который заключается в минимизации суммы мультиклассовой дисперсии. Класс
 * хранит гистограмму исходного изображения, необходимую для вычислений.
 */
class MultiClassOtsuUnit : public ImageProcessingUnit {
 public:
  MultiClassOtsuUnit(const cv::Mat& img, const int levels)
      : ImageProcessingUnit(img, levels) {}
  /// Метод, который бинаризует изображение, в нем имплементирован весь
  /// алгоритм.
  void Process(cv::Mat& gray_img, cv::Mat& bin_img) override;
  /// Метод, создающий нормализованную гистограмму.
  void CreateNormHistogram();
  /// Метод, который ищет наилучшую границу, с минимальной суммой
  /// внутриклассовой дисперсии.
  void SearchThresholds(std::vector<uchar>& thresholds, double& res_var) const;
  /// Метод, который непосредственно бинаризует, то есть создает изображение с
  /// несколькими цветами по исходному с данными границами.
  void BinarizeImage(std::vector<uchar>& thresholds, cv::Mat& gray_img,
                     cv::Mat& bin_img) const;
  /// Метод, который визуализирует гистограмму
  void DrawHist(std::unique_ptr<cv::Mat>& histImage) const;
  /// Метод для рисования графиков суммы дисперсии от значения одного из порогов
  /// при фиксированных других
  void DrawPlots(std::vector<std::unique_ptr<cv::Mat>>& plotsImage) const;

 private:
  /// Рекурсивный алгоритм генерации всевозможных порогов.
  void GenerateAllPossibleThresholds(
      std::vector<std::vector<uchar>>& all_possible_thresholds,
      std::vector<uchar> thresholds, int level) const;
  /// Метод, подсчитывающий внутриклассувую дисперсию с данными порогами по
  /// гистограмме.
  void CalculateIntroClassVariance(float& var,
                                   const std::vector<uchar>& cand_thresholds,
                                   bool print = false) const;

 public:
  cv::Mat m_hist;
  int m_min = 10000000;
  int m_max = 0;
  std::vector<uchar> m_thrs;
};

#endif
