import joblib
import os
from xgboost import XGBRegressor
import pandas as pd

class HousePricePredictor:
    def __init__(self, model_dir='model'):
        self.model = XGBRegressor()
        self.model.load_model(os.path.join(model_dir, 'xgb_model.json'))
        self.preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.joblib'))
        self.city_avg_prices = joblib.load(os.path.join(model_dir, 'city_avg_prices.joblib'))
    
    def predict(self, input_data):
      # ======================
# TAHMİN FONKSİYONU (Başka projelerde kullanım için)
# ======================
def predict_house_price(input_data, model, preprocessor, city_avg_prices):
    """
    Ev fiyatı tahmini yapar
    
    Args:
        input_data (dict): Ev özellikleri
        model: Eğitilmiş XGBoost modeli
        preprocessor: Eğitilmiş ön işleyici
        city_avg_prices: Şehir bazlı ortalama fiyatlar
        
    Returns:
        dict: Tahmin sonuçları
    """
    # DataFrame oluştur
    input_df = pd.DataFrame([input_data])
    
    # Ön işleme
    input_trans = preprocessor.transform(input_df)
    
    # Tahmin
    prediction = model.predict(input_trans)[0]
    
    # Şehir ortalaması
    city_key = f"{input_data['Country']}_{input_data['City']}"
    real_avg = city_avg_prices.get(city_key)
    real_price = real_avg * input_data['Size_m2'] if real_avg else None
    
    # Sonuçları hazırla
    result = {
        'predicted_price': prediction,
        'real_avg_per_m2': real_avg,
        'real_avg_total': real_price
    }
    
    if real_price:
        diff = prediction - real_price
        diff_percent = (diff / real_price) * 100
        result['difference_percent'] = diff_percent
        result['comparison'] = "Gerçek ortalamaya çok yakın" if abs(diff_percent) < 10 else \
                              f"Gerçek ortalamanın %{abs(diff_percent):.1f} {'üzerinde' if diff_percent > 0 else 'altında'}"
    
    return result
