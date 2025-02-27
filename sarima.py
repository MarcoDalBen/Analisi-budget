import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX


def load_data(file_path):
    df = pd.read_excel(file_path)
    df[['Mese', 'Anno']] = df['Mese Anno'].str.split(' ', expand=True)
    df['Anno'] = pd.to_numeric(df['Anno'])
    mesi_mapping = {"Gennaio": 1, "Febbraio": 2, "Marzo": 3, "Aprile": 4, "Maggio": 5, "Giugno": 6,
                    "Luglio": 7, "Agosto": 8, "Settembre": 9, "Ottobre": 10, "Novembre": 11, "Dicembre": 12}
    df['Mese'] = df['Mese'].map(mesi_mapping)
    df['Data'] = pd.to_datetime(
        {'year': df['Anno'], 'month': df['Mese'], 'day': 1})
    df = df.sort_values(by='Data')
    df.set_index('Data', inplace=True)
    df = df.asfreq('MS')
    df.dropna(inplace=True)
    return df[['Valore venduto']]


def test_stationarity(timeseries):
    result = adfuller(timeseries)
    print(f"ADF Statistic: {result[0]:.3f}")
    print(f"p-value: {result[1]:.3f}")
    if result[1] < 0.05:
        print("La serie è stazionaria")
    else:
        print("La serie NON è stazionaria, potrebbe essere necessario differenziarla")


def find_best_sarima_order(df):
    best_aic = float("inf")
    best_order = None
    best_seasonal_order = None

    d = 1 if adfuller(df['Valore venduto'])[1] > 0.05 else 0
    D = 1

    for p in range(3):
        for q in range(3):
            for P in range(2):
                for Q in range(2):
                    try:
                        model = SARIMAX(df, order=(p, d, q), seasonal_order=(P, D, Q, 12),
                                        enforce_stationarity=False, enforce_invertibility=False)
                        model_fit = model.fit(disp=False)
                        if model_fit.aic < best_aic:
                            best_aic = model_fit.aic
                            best_order = (p, d, q)
                            best_seasonal_order = (P, D, Q, 12)
                    except Exception as e:
                        continue
    return best_order, best_seasonal_order, best_aic


def train_sarima(df, start_date, forecast_steps=12, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    df_filtered = df[df.index <= start_date]
    if df_filtered.empty:
        print("Errore: Nessun dato disponibile per addestrare il modello SARIMA.")
        return None, None, None, None

    model = SARIMAX(df_filtered, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()
    forecast_res = model_fit.get_forecast(steps=forecast_steps)
    forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()
    forecast_index = pd.date_range(
        start=start_date + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')

    return forecast, forecast_index, conf_int, model_fit.aic, model_fit.bic


def main():
    file_path = "dati_complessivi_1M.xlsx"
    df = load_data(file_path)
    test_stationarity(df['Valore venduto'])

    print("Calcolando il miglior ordine SARIMA...")
    best_order, best_seasonal_order, best_aic = find_best_sarima_order(df)
    print(
        f"Miglior ordine trovato: {best_order}, {best_seasonal_order}")

    while True:
        start_date_str = input(
            "Inserisci la data di inizio previsione (YYYY-MM): ")
        try:
            start_date = pd.to_datetime(start_date_str, format="%Y-%m")
            break
        except ValueError:
            print("Formato non valido! Usa YYYY-MM.")

    while True:
        try:
            forecast_steps = int(
                input("Inserisci il numero di previsioni da effettuare: "))
            if forecast_steps > 0:
                break
        except ValueError:
            print("Inserisci un numero valido!")

    forecast, forecast_index, conf_int, aic, bic = train_sarima(
        df, start_date, forecast_steps, best_order, best_seasonal_order)

    if forecast is None:
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df, label="Dati storici", marker='o', linestyle='-')
    plt.plot(forecast_index, forecast, label="Previsione SARIMA",
             marker='o', linestyle='dashed', color='red')
    plt.fill_between(
        forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='red', alpha=0.2)

    df_real = df[df.index.isin(forecast_index)]
    if not df_real.empty:
        plt.plot(df_real.index, df_real['Valore venduto'],
                 label="Valori Reali", linestyle='-', color='green')

    plt.title("Previsione del Valore Venduto con SARIMA")
    plt.xlabel("Anno")
    plt.ylabel("Valore Venduto (€)")
    plt.legend()
    plt.grid()
    plt.show()

    forecast_df = pd.DataFrame(
        {'Data': forecast_index, 'Previsione Valore Venduto': forecast.values})
    print(forecast_df)

    print(f"\nMetriche del modello:")
    print(f"AIC: {aic:.2f}")
    print(f"BIC: {bic:.2f}")

    sum_forecast = sum(forecast)
    print(f"\nTotale previsto: €{sum_forecast:.2f}")

    #print(forecast_df['Previsione Valore Venduto'].to_csv())

if __name__ == "__main__":
    main()