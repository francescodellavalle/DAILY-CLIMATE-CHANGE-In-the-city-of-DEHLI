import streamlit as st
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pmdarima as pm
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Page configuration
st.set_page_config(
    page_title='Weather change',
    page_icon='☀️'
)

# Carica il dataset
dataset_path = "DailyDelhiClimateTrain.csv" 
df = pd.read_csv(dataset_path)
st.title("Daily climate change in the city of Delhi 🌦️")
# Display the dataset
st.write("Dataset completo:")
st.write(df)

# Analisi in serie
st.header("Analisi in serie:")
#  grafici che si desidera visualizzare in base alle colonne del dataset
st.write("### Temperatura Media")
st.line_chart(df.set_index('date')['meantemp'])
st.write("### Umidità")
st.line_chart(df.set_index('date')['humidity'])
st.write("### Velocità del Vento")
st.line_chart(df.set_index('date')['wind_speed'])
st.write("### Pressione Media")
st.line_chart(df.set_index('date')['meanpressure'])

# Titolo della barra laterale
st.sidebar.title("⚙️ Settings dashboard")

# features
numeric_features = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']

# Sezione "Relazioni tra feature"
st.header("Relazione tra meantemp e altre feature:")
selected_features = st.sidebar.multiselect("Seleziona le feature da confrontare con la temperatura:", numeric_features, default=['humidity'])

if selected_features:
    fig = px.scatter(df, x='meantemp', y=selected_features, title=f"meantemp con {', '.join(selected_features)}")
    st.plotly_chart(fig)
else:
    st.warning("Seleziona almeno una feature da confrontare con meantemp.")


# Analisi dei dati con Plotly Express
st.header("Analisi dei dati: ")

# Mappa tra le opzioni nel selectbox e i nomi originali delle colonne
column_labels_mapping = {"Meantemp": "meantemp", "Humidity": "humidity", "Wind Speed": "wind_speed", "Mean Pressure": "meanpressure"}

# Seleziona colonna per l'analisi interattiva
selected_column_interactive_label = st.sidebar.selectbox("Seleziona una colonna per l'analisi:",
                                                 [""] + list(column_labels_mapping.keys()))  

# Verifica se è stata selezionata una colonna
if selected_column_interactive_label:
    # Mappa il nome originale della colonna dalla label selezionata
    selected_column_interactive = column_labels_mapping[selected_column_interactive_label]

    # Creazione del DataFrame aggregato
    df_aggregated_interactive = df.groupby('date')[[selected_column_interactive]].mean().reset_index()

    # Creazione del grafico interattivo
    fig_interactive = px.line(df_aggregated_interactive, x='date', y=selected_column_interactive,
                              title=f'Analisi interattiva - {selected_column_interactive}', line_shape='linear',
                              labels={'value': 'Valore', 'date': 'data'})

    # Mostra il grafico interattivo
    st.plotly_chart(fig_interactive)

else:
    st.warning("Seleziona almeno una colonna per l'analisi interattiva.")


# Modello SARIMA
st.header("Modello SARIMA:")
time_column_sarima = 'date'
temp_column_sarima = 'meantemp'

df['date'] = pd.to_datetime(df['date'])
df_aggregated_sarima = df.groupby(pd.Grouper(key='date', freq='M')).mean().reset_index()

if len(df_aggregated_sarima) < 2:
    st.error("Non ci sono abbastanza dati per generare le previsioni.")
else:
    order_sarima = (2, 0, 0)
    seasonal_order_sarima = (1, 1, 1, 4)

    model_sarima = sm.tsa.SARIMAX(df_aggregated_sarima[temp_column_sarima], order=order_sarima, seasonal_order=seasonal_order_sarima)
    result_sarima = model_sarima.fit()

    #st.subheader("Risultati del modello SARIMA:")
    #st.write(result_sarima.summary())

    forecast_period_years_sarima = st.sidebar.slider("Seleziona il periodo di previsione SARIMA (anni)", 1, 7, 2)

    st.info("Dopo aver selezionato il numero di anni per il forecast, clicca su 'Esegui previsioni SARIMA'")

    if st.button("Esegui previsioni SARIMA"):
        last_date_sarima = pd.to_datetime(df_aggregated_sarima[time_column_sarima].max())
        forecast_period_start_sarima = last_date_sarima + pd.DateOffset(days=1)
        forecast_period_end_sarima = forecast_period_start_sarima + pd.DateOffset(years=forecast_period_years_sarima)
        forecast_index_sarima = pd.date_range(start=forecast_period_start_sarima, end=forecast_period_end_sarima, freq='M')

        forecast_df_sarima = pd.DataFrame({'date': forecast_index_sarima[:-1],
                                           'Forecast': result_sarima.get_forecast(steps=len(forecast_index_sarima)-1).predicted_mean})

        fig_sarima = px.line(df_aggregated_sarima, x=time_column_sarima, y=temp_column_sarima,
                             title=f'{temp_column_sarima} con Previsioni SARIMA')
        fig_sarima.add_scatter(x=forecast_df_sarima['date'], y=forecast_df_sarima['Forecast'],
                               mode='lines', name='Previsioni Future', line=dict(color='red'))
        st.plotly_chart(fig_sarima)


# Modello ARIMA
#st.write("Modello ARIMA:")
#time_column_arima = 'date'
#temp_column_arima = 'meantemp'

#df['date'] = pd.to_datetime(df['date'])
#df_aggregated_arima = df.groupby(pd.Grouper(key='date', freq='M')).mean().reset_index()

#if len(df_aggregated_arima) < 2:
 #   st.error("Non ci sono abbastanza dati per generare le previsioni.")
#else:
 #   order_arima = (5, 1, 0)  # Ordine del modello ARIMA (p, d, q)
    
  #  model_arima = ARIMA(df_aggregated_arima[temp_column_arima], order=order_arima)
  #  result_arima = model_arima.fit()

  #  st.subheader("Risultati del modello ARIMA:")
  #  st.write(result_arima.summary())

   # forecast_period_years_arima = st.slider("Seleziona il periodo di previsione ARIMA (anni)", 1, 30, 7)

   # if st.button("Esegui previsioni ARIMA"):
    #    last_date_arima = pd.to_datetime(df_aggregated_arima[time_column_arima].max())
     #   forecast_period_start_arima = last_date_arima + pd.DateOffset(days=1)
     #   forecast_period_end_arima = forecast_period_start_arima + pd.DateOffset(years=forecast_period_years_arima)
     #   forecast_index_arima = pd.date_range(start=forecast_period_start_arima, end=forecast_period_end_arima, freq='M')

      #  forecast_arima = result_arima.predict(start=len(df_aggregated_arima), end=len(df_aggregated_arima) + len(forecast_index_arima) - 1)
        
      #  forecast_df_arima = pd.DataFrame({'date': forecast_index_arima[:-1],
        #                          'Forecast': np.concatenate([df_aggregated_arima[temp_column_arima].values, forecast_arima])})

        #fig_arima = px.line(df_aggregated_arima, x=time_column_arima, y=temp_column_arima,
         #                   title=f'{temp_column_arima} con Previsioni ARIMA')
        #fig_arima.add_scatter(x=forecast_df_arima['date'], y=forecast_df_arima['Forecast'],
         #                     mode='lines', name='Previsioni Future', line=dict(color='red'))
        #st.plotly_chart(fig_arima)