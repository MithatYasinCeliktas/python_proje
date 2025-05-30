# ======================
# GEREKLİ KÜTÜPHANELER
# ======================
!pip install pandas scikit-learn xgboost pycountry matplotlib joblib > /dev/null
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pycountry
import matplotlib.pyplot as plt
import time
import joblib

# ======================
# HIPERPARAMETRE AYARLARI (Yakınsama Optimize)
# ======================
EPOCHS = 500
LEARNING_RATE = 0.05
MAX_DEPTH = 7
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
NUM_SAMPLES = 50000
TEST_SIZE = 0.2
RANDOM_STATE = 42
REAL_MAE_TOLERANCE = 0.07  # ±7% tolerans
CONVERGENCE_THRESHOLD = 0.05  # %5 yakınsama eşiği
STABILITY_WINDOW = 15         # Stabilite kontrolü için epoch penceresi

# ======================
# GERÇEK FİYAT VERİ TABANI
# ======================
city_avg_prices = {
    'TUR_Istanbul': 2000, 'TUR_Ankara': 1200, 'TUR_Izmir': 1500, 'TUR_Antalya': 1800,
    'USA_New York': 10000, 'USA_Los Angeles': 8000, 'USA_Chicago': 5000,
    'DEU_Berlin': 6500, 'DEU_Munich': 9500, 'DEU_Hamburg': 7000,
    'FRA_Paris': 11000, 'FRA_Lyon': 5500,
    'GBR_London': 12000, 'GBR_Manchester': 4000,
    'CAN_Toronto': 8500, 'AUS_Sydney': 9500, 'ESP_Madrid': 5500, 'ITA_Rome': 6000
}

# ======================
# SENTETİK VERİ OLUŞTURMA (Daha Gerçekçi)
# ======================
def generate_realistic_data(num_samples=NUM_SAMPLES):
    np.random.seed(RANDOM_STATE)
    
    # Ülkeleri ve şehirleri gerçek fiyat veritabanından al
    countries = list(set(k.split('_')[0] for k in city_avg_prices.keys()))
    cities = {}
    
    # Her ülke için şehir listesi oluştur
    for key in city_avg_prices:
        country, city = key.split('_')
        if country not in cities:
            cities[country] = []
        cities[country].append(city)
    
    # Veri oluşturma
    data = []
    for _ in range(num_samples):
        country = np.random.choice(countries)
        city = np.random.choice(cities[country])
        city_key = f"{country}_{city}"
        base_price = city_avg_prices[city_key]
        
        # Ev özellikleri (gerçekçi dağılımlar)
        m2 = max(40, np.random.normal(100, 40))  # Ortalama 100m²
        rooms = min(6, max(1, int(np.random.normal(2.5, 1))))
        floor = min(20, max(0, int(np.random.normal(3, 4))))
        has_garden = np.random.choice([0, 1], p=[0.6, 0.4])
        to_center = np.random.choice(['yakın', 'orta', 'uzak'], p=[0.3, 0.5, 0.2])
        to_transport = np.random.choice(['yakın', 'orta', 'uzak'], p=[0.5, 0.4, 0.1])
        year = np.random.randint(1990, 2023)
        
        # Fiyat hesaplama (daha gerçekçi)
        price = base_price * m2
        
        # Özelliklere göre fiyat ayarlamaları
        price *= (1 + (rooms - 1) * 0.1)  # Oda sayısı etkisi
        
        # Kat etkisi (zemin ve üst katlar farklı)
        if floor == 0:  # Zemin kat
            price *= 0.95
        elif floor == 1:  # 1. kat
            price *= 1.05
        else:  # Diğer katlar
            price *= (1 + min(floor, 10) * 0.01)
        
        # Bahçe etkisi
        price *= 1.15 if has_garden else 1.0
        
        # Konum etkileri
        if to_center == 'orta': price *= 0.95
        elif to_center == 'uzak': price *= 0.85
            
        if to_transport == 'orta': price *= 0.97
        elif to_transport == 'uzak': price *= 0.92
        
        # Yapım yılı etkisi
        age = 2023 - year
        if age < 5:  # Yeni bina
            price *= 1.2
        elif age < 15:  # Orta yaşlı
            price *= 1.05
        elif age < 30:  # Eski
            price *= 0.95
        else:  # Çok eski
            price *= 0.85
        
        # Rastgele varyasyon
        price *= np.random.uniform(0.95, 1.05)
        
        data.append([
            country, city, m2, rooms, floor, int(has_garden),
            to_center, to_transport, year, price
        ])
    
    return pd.DataFrame(data, columns=[
        'Country', 'City', 'Size_m2', 'Rooms', 'Floor', 'Garden',
        'Center_Distance', 'Transport_Distance', 'Year', 'Price'
    ])

print("Sentetik veri oluşturuluyor...")
start_time = time.time()
synthetic_data = generate_realistic_data()
print(f"Oluşturulan veri seti: {synthetic_data.shape} - Süre: {time.time()-start_time:.2f}s")

# ======================
# YAKINSAMA İZLEYİCİ (Düzeltilmiş)
# ======================
class CorrectedConvergenceMonitor:
    def __init__(self, real_prices, X_test, y_test, preprocessor):
        self.real_prices = real_prices
        self.X_test = X_test
        self.y_test = y_test
        self.preprocessor = preprocessor
        self.X_test_trans = preprocessor.transform(X_test)
        
        # Şehir bazlı verileri önceden hesapla
        self.city_data = {}
        self.expected_real_mae = 0
        for city_key, real_price_per_m2 in real_prices.items():
            country, city = city_key.split('_')
            mask = (X_test['Country'] == country) & (X_test['City'] == city)
            if mask.sum() > 0:
                sizes = X_test.loc[mask, 'Size_m2'].values
                real_prices = real_price_per_m2 * sizes
                self.city_data[city_key] = {
                    'mask': mask.values,
                    'real_total_prices': real_prices,
                    'sizes': sizes
                }
                # Beklenen Real MAE hesapla (gerçek fiyatlarla)
                self.expected_real_mae += mean_absolute_error(y_test[mask], real_prices)
        
        # Ortalama beklenen Real MAE
        if self.city_data:
            self.expected_real_mae /= len(self.city_data)
        
        # İzleme metrikleri
        self.epoch_mae = []
        self.epoch_real_mae = []
        self.mae_deltas = []  # Test MAE değişim miktarları
        self.start_time = time.time()
        
    def calculate_real_mae(self, y_pred):
        """Gerçek fiyatlara göre MAE hesaplar"""
        real_maes = []
        for city_key, data in self.city_data.items():
            if np.any(data['mask']):
                city_pred = y_pred[data['mask']]
                city_mae = mean_absolute_error(data['real_total_prices'], city_pred)
                real_maes.append(city_mae)
        return np.mean(real_maes) if real_maes else 0
    
    def calculate_convergence_ratio(self, test_mae, real_mae):
        """Test MAE ile Real MAE arasındaki yakınsama oranını hesaplar"""
        if real_mae == 0:
            return 1.0
        return min(test_mae / real_mae, real_mae / test_mae)
    
    def is_real_mae_in_tolerance(self, real_mae):
        """Real MAE'nin beklenen değerle ±%7 aralığında olup olmadığını kontrol eder"""
        if self.expected_real_mae == 0:
            return False
        lower_bound = self.expected_real_mae * (1 - REAL_MAE_TOLERANCE)
        upper_bound = self.expected_real_mae * (1 + REAL_MAE_TOLERANCE)
        return lower_bound <= real_mae <= upper_bound
    
    def is_stable(self, values, window_size):
        """Değerlerin son 'window_size' epoch'ta stabil olup olmadığını kontrol eder"""
        if len(values) < window_size:
            return False
        recent = values[-window_size:]
        return np.std(recent) < np.mean(recent) * 0.05  # %5'ten az standart sapma
    
    def callback(self, epoch, model, y_pred):
        # Test MAE
        test_mae = mean_absolute_error(self.y_test, y_pred)
        self.epoch_mae.append(test_mae)
        
        # Gerçek fiyat MAE
        real_mae = self.calculate_real_mae(y_pred)
        self.epoch_real_mae.append(real_mae)
        
        # Yakınsama oranı
        convergence_ratio = self.calculate_convergence_ratio(test_mae, real_mae)
        
        # Real MAE tolerans kontrolü
        in_tolerance = self.is_real_mae_in_tolerance(real_mae)
        tolerance_status = "✓" if in_tolerance else "✗"
        
        # MAE delta hesaplama (önceki epoch ile fark)
        if epoch > 0:
            delta = test_mae - self.epoch_mae[-2]
            self.mae_deltas.append(delta)
            
            # Test MAE davranışını düzelt
            # Test MAE > Real MAE ise azalt, Test MAE < Real MAE ise artır
            if test_mae > real_mae:
                # Azaltma eğilimi
                desired_delta = -abs(delta) if delta > 0 else delta
            else:
                # Artırma eğilimi
                desired_delta = abs(delta) if delta < 0 else delta
            
            # Delta'yi istenen yönde ayarla
            adjusted_delta = desired_delta * 0.8  # Daha yumuşak geçiş
            self.mae_deltas[-1] = adjusted_delta
        
        # Stabilite kontrolü
        stable_real = self.is_stable(self.epoch_real_mae, STABILITY_WINDOW)
        stable_test = self.is_stable(self.epoch_mae, STABILITY_WINDOW)
        converged = convergence_ratio > (1 - CONVERGENCE_THRESHOLD)
        
        # İlerleme raporu
        elapsed = time.time() - self.start_time
        status = "✓" if converged else "✗"
        stability = "S" if stable_real and stable_test else "U"
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Test MAE: ${test_mae:,.2f} - "
              f"Real MAE: ${real_mae:,.2f} - Yakınsama: {convergence_ratio:.3f} {status} - "
              f"Tolerans: {tolerance_status} - Stabilite: {stability} - Süre: {elapsed:.2f}s")
        
        # Durdurma koşulları
        stop_reasons = []
        if converged and stable_real and in_tolerance:
            stop_reasons.append("Yakınsama, stabilite ve tolerans sağlandı")
        
        if in_tolerance and stable_real and stable_test:
            stop_reasons.append("Hedef aralıkta ve stabil")
        
        if stop_reasons:
            print(f"✓ Eğitim durduruldu: {', '.join(stop_reasons)}")
            print(f"✓ Beklenen Real MAE: ${self.expected_real_mae:,.2f} ±{REAL_MAE_TOLERANCE*100:.0f}%")
            print(f"✓ Son Real MAE: ${real_mae:,.2f}")
            return True
        
        return False


# ======================
# VERİ ÖN İŞLEME
# ======================
if __name__ == "__main__":
    print("Sentetik veri oluşturuluyor...")
    synthetic_data = generate_realistic_data()
    
cat_cols = ['Country', 'City', 'Center_Distance', 'Transport_Distance']
num_cols = ['Size_m2', 'Rooms', 'Floor', 'Year']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
# ======================
# MODEL EĞİTİMİ (Gerçek Iteratif Eğitim)
# ======================
model = XGBRegressor(
    n_estimators=1,  # Her epoch'ta sadece 1 ağaç ekleyeceğiz
    learning_rate=LEARNING_RATE,
    max_depth=MAX_DEPTH,
    subsample=SUBSAMPLE,
    colsample_bytree=COLSAMPLE_BYTREE,
    random_state=RANDOM_STATE
)

monitor = CorrectedConvergenceMonitor(city_avg_prices, X_test, y_test, preprocessor)

print("\nModel eğitimi başlıyor...")
start_time = time.time()

# İlk epoch
model.fit(X_train_trans, y_train)
y_pred = model.predict(X_test_trans)
stop_training = monitor.callback(0, model, y_pred)

# Sonraki epoch'lar - GERÇEK iteratif eğitim
for epoch in range(1, EPOCHS):
    if stop_training:
        break
        
    # Modeli güncelle (yeni ağaç ekle)
    model.fit(X_train_trans, y_train, xgb_model=model)
    
    # Yeni tahminler al
    y_pred = model.predict(X_test_trans)
    
    # İlerlemeyi kontrol et
    stop_training = monitor.callback(epoch, model, y_pred)

print(f"\nToplam eğitim süresi: {time.time()-start_time:.2f}s")
