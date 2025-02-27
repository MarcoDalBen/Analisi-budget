import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def load_data(file_path):
    df = pd.read_excel(file_path)
    df[['Mese', 'Anno']] = df['Mese Anno'].str.split(' ', expand=True)
    df['Anno'] = pd.to_numeric(df['Anno'])
    mesi_mapping = {"Gennaio": 1, "Febbraio": 2, "Marzo": 3, "Aprile": 4, "Maggio": 5, "Giugno": 6,
                    "Luglio": 7, "Agosto": 8, "Settembre": 9, "Ottobre": 10, "Novembre": 11, "Dicembre": 12}
    df['Mese'] = df['Mese'].map(mesi_mapping)
    df['Data'] = pd.to_datetime({'year': df['Anno'], 'month': df['Mese'], 'day': 1})
    df = df.sort_values(by='Data')
    df.set_index('Data', inplace=True)
    df = df.asfreq('MS')
    df.dropna(inplace=True)
    return df[['Valore venduto']]

def holt_winters_forecast(df, periods=12):
    model = ExponentialSmoothing(
        df['Valore venduto'], 
        trend='add', 
        seasonal='add', 
        seasonal_periods=12,
        damped_trend=True
    )
    fit = model.fit(optimized=True)
    forecast = fit.forecast(periods)
    
    return fit, forecast

def main():
    file_path = "dati_complessivi_1M.xlsx"
    df = load_data(file_path)
    fit, forecast = holt_winters_forecast(df, 12)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Valore venduto'], label="Storico")
    plt.plot(forecast.index, forecast, label="Previsione", linestyle="dashed", color="red")
    plt.xlabel("Data")
    plt.ylabel("Valore venduto")
    plt.title("Previsione con Holt-Winters (Trend Smorzato e Lisciamento)")
    plt.legend()
    plt.grid()
    plt.show()

    print(forecast)
    print(fit.params)
    print(f"Totale previsto: €{round(sum(forecast), 2)}")

if __name__ == "__main__":
    main()
