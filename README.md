# mental_health_in_tech

## Mental Health Clustering Case Study

This project analyzes survey data related to workplace mental health using unsupervised learning methods. It applies feature selection, PCA, and K-Means clustering to uncover patterns that may help HR departments improve support programs.

## ⚙️ Setup Instructions

To run this project, make sure you have **Python 3.9** installed.  
You can download it from the official website:

👉 [Download Python 3.9](https://www.python.org/downloads/release/python-390/)

---

### 🐍 Install `pip` (Python Package Installer)

If `pip` is not already installed on your system, follow the instructions below:

#### On Windows:
1. Download [`get-pip.py`](https://bootstrap.pypa.io/get-pip.py)
2. Open Command Prompt where the file is downloaded.
3. Run:
   ```bash
   python get-pip.py

#### On macOS/Linux:
#### Ubuntu/Debian
sudo apt install python3-pip

#### macOS with Homebrew
brew install python

### Once Python, pip are installed, install all required Python packages using:
1. Run:
   ```bash
   pip install -r requirements.txt

### Project Files

- `fixing_missing_values.py` – Handles loading, cleaning, and transforming the data.
- `data_preparation_encoding.py` – Performs data transformation and encoding.
- `Kmeans.py` – feature selection, KMeans clustering and PCA.
- `summary_interpret.py` – Generates the summary plot.
- `main.py` – Coordinates the overall pipeline.

### Usage
1. Run main.py file