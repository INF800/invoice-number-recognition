# invoice-number-recognition
OCR

# How To Use

### 1. Create a `python 3.7.4`  virtual environment and activate it
### 2. Install all packages from `requirements.txt`
```
fuzzywuzzy==0.17.0
glob3==0.0.1
imutils==0.5.3
numpy==1.17.2
opencv-contrib-python==4.1.1.26
Pillow==6.2.0
pytesseract==0.3.0
XlsxWriter==1.2.1
```
### 3. Setup Tesseract. [Follow this thread](https://stackoverflow.com/questions/50951955/pytesseract-tesseractnotfound-error-tesseract-is-not-installed-or-its-not-i/#answer-53672281)
### 4. In line 13 of `invoice_number_recognition.py` add correct location of your executable `tesseract.exe` (Windows)
```
# add path to tesseract here
pytesseract.pytesseract.tesseract_cmd = r'YOUR_CORRECT_PATH'
```
*Please [Follow this thread](https://stackoverflow.com/questions/50951955/pytesseract-tesseractnotfound-error-tesseract-is-not-installed-or-its-not-i/#answer-53672281)*
### 5. Now as everything is ready, run the script
```
python invoice_number_recognition.py
```

*Note: You may add additional arguments as well (Not Recommended)*
