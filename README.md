# AdaCycle
AdaCycle: Global Statistical Adaptive Multi-Cycle Detection and Fusion with Heterogeneous Wavelet Decomposition for time series forecasting

## Updates
The detailed training logs:
- Long-term forecasting: [logs/LongForecasting/AdaCycle](logs/LongForecasting/AdaCycle)
- Short-term forecasting: [logs/ShortForecasting/AdaCycle](logs/ShortForecasting/AdaCycle)

## Usage
1. Install the dependencies
```bash
    pip install -r requirements.txt
```
2. Obtain the dataset from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) and extract it to the root directory of the project. Make sure the extracted folder is named `dataset` and has the following structure:
```
    dataset
    ├── electricity
    │   └── electricity.csv
    ├── ETT-small
    │   ├── ETTh1.csv
    │   ├── ETTh2.csv
    │   ├── ETTm1.csv
    │   └── ETTm2.csv
    ├── PEMS
    │   ├── PEMS03.npz
    │   ├── PEMS04.npz
    │   ├── PEMS07.npz
    │   └── PEMS08.npz
    ├── Solar
    │   └── solar_AL.txt
    ├── traffic
    │   └── traffic.csv
    └── weather
        └── weather.csv
```
3. Train and evaluate the model. All the training scripts are located in the `scripts` directory.
   
   **Long-term forecasting:**
   
   For Linux/macOS:
```bash
    sh ./scripts/LongForecasting/PCFNet.sh
```
    For Windows:
```bash
    ./scripts/LongForecasting/PCFNet.bat
```
   
   **Short-term forecasting:**
   
   For Linux/macOS:
```bash
    sh ./scripts/ShortForecasting/PCFNet.sh
```
    For Windows:
```bash
    ./scripts/ShortForecasting/PCFNet.bat
```
