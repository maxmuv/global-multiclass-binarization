# global-multiclass-binarization

Программа для глобальной многоклассовой бинаризации. 

Зависимости программы: opencv и doxygen.

Установка и тестирование из консоли:

cmake path\to\global-multiclass-binarization

cmake --build . --target install --config Release

IMAGE_PROCESSING_TEST

binarization -f demo\src\target.txt

Использование:

binarization -i path\to\input\gray\image -o path\to\output\binarized\image -l levels

или

binarization -f path\to\file (файл должен содержать два пути и цифру через пробелы в каждой строчке)