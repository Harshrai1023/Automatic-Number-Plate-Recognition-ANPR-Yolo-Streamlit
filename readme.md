# Automatic Number Plate Recognition (ANPR) using YOLO and Streamlit

This project aims to develop an Automatic Number Plate Recognition (ANPR) system for Indian Number plates using YOLO (You Only Look Once) object detection algorithm and Streamlit web application framework.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/harshrai1023/Automatic-Number-Plate-Recognition-ANPR-Yolo-Streamlit.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. The pre-trained YOLO weights file can be found in the `models` directory.

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Upload an image containing number plates.

4. Click the "Detect" button to start the ANPR process.

## Dataset

The dataset used for the training the YOLOv8 model can be downloaded from the Roboflow website [here](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4).
