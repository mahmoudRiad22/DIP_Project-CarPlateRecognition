# DIP_Project-CarPlateRecognition
## 0. Install and Import Dependencies
    
```
!pip install easyocr
!pip install imutils
```
```
Requirement already satisfied: easyocr in /usr/local/lib/python3.10/dist-packages (1.7.1)
Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (from easyocr) (2.1.0+cu121)
Requirement already satisfied: torchvision>=0.5 in /usr/local/lib/python3.10/dist-packages (from easyocr) (0.16.0+cu121)
Requirement already satisfied: opencv-python-headless in /usr/local/lib/python3.10/dist-packages (from easyocr) (4.8.1.78)
Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from easyocr) (1.11.4)
Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from easyocr) (1.23.5)
Requirement already satisfied: Pillow in /usr/local/lib/python3.10/dist-packages (from easyocr) (9.4.0)
Requirement already satisfied: scikit-image in /usr/local/lib/python3.10/dist-packages (from easyocr) (0.19.3)
Requirement already satisfied: python-bidi in /usr/local/lib/python3.10/dist-packages (from easyocr) (0.4.2)
Requirement already satisfied: PyYAML in /usr/local/lib/python3.10/dist-packages (from easyocr) (6.0.1)
Requirement already satisfied: Shapely in /usr/local/lib/python3.10/dist-packages (from easyocr) (2.0.2)
Requirement already satisfied: pyclipper in /usr/local/lib/python3.10/dist-packages (from easyocr) (1.3.0.post5)
Requirement already satisfied: ninja in /usr/local/lib/python3.10/dist-packages (from easyocr) (1.11.1.1)
Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from torchvision>=0.5->easyocr) (2.31.0)
Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch->easyocr) (3.13.1)
Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch->easyocr) (4.5.0)
Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch->easyocr) (1.12)
Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch->easyocr) (3.2.1)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch->easyocr) (3.1.2)
Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch->easyocr) (2023.6.0)
Requirement already satisfied: triton==2.1.0 in /usr/local/lib/python3.10/dist-packages (from torch->easyocr) (2.1.0)
Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from python-bidi->easyocr) (1.16.0)
Requirement already satisfied: imageio>=2.4.1 in /usr/local/lib/python3.10/dist-packages (from scikit-image->easyocr) (2.31.6)
Requirement already satisfied: tifffile>=2019.7.26 in /usr/local/lib/python3.10/dist-packages (from scikit-image->easyocr) (2023.12.9)
Requirement already satisfied: PyWavelets>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from scikit-image->easyocr) (1.5.0)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from scikit-image->easyocr) (23.2)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch->easyocr) (2.1.3)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision>=0.5->easyocr) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision>=0.5->easyocr) (3.6)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision>=0.5->easyocr) (2.0.7)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision>=0.5->easyocr) (2023.11.17)
Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch->easyocr) (1.3.0)
```
```
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
from google.colab import files
import io
```

## 1.Read the Dataset provided:
```
uploaded_files = files.upload()
image_array = []
image_names = []
for filename, content in uploaded_files.items():
    # Read the image from bytes
    image = cv2.imdecode(np.frombuffer(content, np.uint8), -1)
    image_array.append(image)
    image_names.append(filename)
```
```
N1.jpeg(image/jpeg) - 612946 bytes, last modified: 6/6/2022 - 100% done
N5.jpeg(image/jpeg) - 626596 bytes, last modified: 6/6/2022 - 100% done
N6.jpeg(image/jpeg) - 463596 bytes, last modified: 6/6/2022 - 100% done
N7.jpeg(image/jpeg) - 375067 bytes, last modified: 6/6/2022 - 100% done
N14.jpeg(image/jpeg) - 591965 bytes, last modified: 6/6/2022 - 100% done
N18.jpeg(image/jpeg) - 355928 bytes, last modified: 6/6/2022 - 100% done
N21.jpeg(image/jpeg) - 102904 bytes, last modified: 6/6/2022 - 100% done
N135.jpeg(image/jpeg) - 54150 bytes, last modified: 6/6/2022 - 100% done
N145.jpeg(image/jpeg) - 35194 bytes, last modified: 6/6/2022 - 100% done
N147.jpeg(image/jpeg) - 245799 bytes, last modified: 6/6/2022 - 100% done
N149.jpeg(image/jpeg) - 51700 bytes, last modified: 6/6/2022 - 100% done
N150.jpeg(image/jpeg) - 49212 bytes, last modified: 6/6/2022 - 100% done
N153.jpeg(image/jpeg) - 65363 bytes, last modified: 6/6/2022 - 100% done
N154.jpeg(image/jpeg) - 57566 bytes, last modified: 6/6/2022 - 100% done
N155.jpeg(image/jpeg) - 33900 bytes, last modified: 6/6/2022 - 100% done
N220.jpeg(image/jpeg) - 53699 bytes, last modified: 6/6/2022 - 100% done
N221.jpeg(image/jpeg) - 37642 bytes, last modified: 6/6/2022 - 100% done
N225.jpeg(image/jpeg) - 1015613 bytes, last modified: 6/6/2022 - 100% done
N238.jpeg(image/jpeg) - 81545 bytes, last modified: 6/6/2022 - 100% done
N241.jpeg(image/jpeg) - 156569 bytes, last modified: 6/6/2022 - 100% done
```
## 2.Choose a test image:
```
num_images_to_select = 1
test_sample_indexs = np.random.choice(len(image_array), num_images_to_select, replace=False)
test_sample = [image_array[i] for i in test_sample_indexs]
test_sample_names = [image_names[i] for i in test_sample_indexs]
```
![image.png](https://github.com/mahmoudRiad22/DIP_Project-CarPlateRecognition/blob/main/Dataset/MySelection/N241.jpeg)





