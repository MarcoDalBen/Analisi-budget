import pandas as pd
import matplotlib.pyplot as plt


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


def moving_average_forecast(df, window=6, forecast_horizon=12):
    df['Media mobile'] = df['Valore venduto'].rolling(window=window).mean()
    last_moving_avg = df['Media mobile'].iloc[-1]
    
    future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=forecast_horizon, freq='MS')
    forecast_values = [last_moving_avg] * forecast_horizon
    
    forecast_df = pd.DataFrame({'Valore venduto previsto': forecast_values}, index=future_dates)
    return forecast_df


def plot_forecast(df, forecast_df):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Valore venduto'], label='Storico', marker='o')
    plt.plot(df.index, df['Media mobile'], label='Media mobile', linestyle='dotted')
    plt.plot(forecast_df.index, forecast_df['Valore venduto previsto'], label='Previsione', linestyle='dashed', marker='o')
    plt.xlabel('Data')
    plt.ylabel('Valore venduto')
    plt.title('Previsione con Media Mobile')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    file_path = "dati_complessivi_1M.xlsx"
    df = load_data(file_path)
    forecast_df = moving_average_forecast(df)
    plot_forecast(df, forecast_df)
    
    print(forecast_df)
    print(f"Totale previsto: €{round(sum(forecast_df['Valore venduto previsto']), 2)}")


if __name__ == "__main__":
    main()