import torch
import torch.nn as nn
import pandas as pd
import joblib
import numpy as np

SEQUENCE_LENGTH = 30

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return self.sigmoid(out).squeeze()
    
class ModelPredict:
    def __init__(self, model_path, scaler_path, data_path):
        features =  ['date', 'ticker', 'open', 'volume', 'adjclose', 'RSIadjclose15', 'RSIvolume15', 'RSIadjclose25','RSIvolume25', 
                'MACDadjclose15', 'MACDvolume15', 'MACDadjclose25','MACDvolume25','emaadjclose5', 'emavolume5',
                'emaadjclose10', 'emavolume10', 'emaadjclose15', 'emavolume15', 'emaadjclose50', 'emavolume50']
        self.model = LSTMClassifier(input_size=len(features)-2)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.scaler = joblib.load(scaler_path)
        self.data = pd.read_csv(data_path)
        self.data = self.data[features]
        self.data = self.data.dropna()
        self.data = self.data.reset_index(drop=True)
    
    def predict(self, start_date, ticker):
        index = self.data[(self.data['date'] == start_date) & (self.data['ticker'] == ticker)].index[0]
        self.pred_data = self.data.drop(columns=['date', 'ticker'])
        self.pred_data = self.scaler.transform(self.pred_data)
        x = self.pred_data[index:index+SEQUENCE_LENGTH]

        x = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            predictions = self.model(x.unsqueeze(0))
            predictions_binary = (predictions > 0.5).detach().int().numpy()
        
        return predictions_binary.flatten().tolist(), predictions.flatten().tolist()