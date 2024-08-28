# Databricks notebook source
# MAGIC %pip install databricks-sdk==0.20.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install prophet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from delta.tables import DeltaTable 
from pyspark.sql.functions import col, when, sum, current_date, lit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.model_selection import ParameterGrid

# COMMAND ----------

# DBTITLE 1,Group by main SKU category
data_to_predict = spark.sql("""
  select u.workspace_id, 
  u.usage_date as ds, 
  u.sku_name as sku, 
  cast(u.usage_quantity as double) as dbus, 
  cast(lp.pricing.default*usage_quantity as double) as custo 
  
  from system.billing.usage u 
      inner join system.billing.list_prices lp on u.cloud = lp.cloud and
        u.sku_name = lp.sku_name and
        u.usage_start_time >= lp.price_start_time and
        (u.usage_end_time <= lp.price_end_time or lp.price_end_time is null)
  where u.usage_unit = 'DBU'
""")

# Criar uma nova coluna 'original_sku' para armazenar os valores originais de 'sku'
data_to_predict = data_to_predict.withColumn("original_sku", col("sku"))

# Aplicar a transformação na coluna 'sku' original
data_to_predict = data_to_predict.withColumn("sku",
                     when(col("sku").contains("ALL_PURPOSE"), "ALL_PURPOSE")
                    .when(col("sku").contains("JOBS"), "JOBS")
                    .when(col("sku").contains("DLT"), "DLT")
                    .when(col("sku").contains("SQL"), "SQL")
                    .when(col("sku").contains("INFERENCE"), "MODEL_INFERENCE")
                    .otherwise("OTHER"))
data_to_predict.display()

# Somar o consumo em nível diário (1 valor por dia e por sku+espaço de trabalho)
data_to_predict_daily = data_to_predict.groupBy(col("ds"),\
                                                col('sku'),\
                                                col('workspace_id')).\
                                                  agg(sum('dbus').alias("dbus"),\
                                                      sum('custo').alias("custo"))



# COMMAND ----------

data_to_predict.write.format('delta').mode('overwrite').saveAsTable(f"gold_prod.custodb_diario")

# COMMAND ----------

# DBTITLE 1,Funções para escolha de melhores hyper parametros e treinamento modelo
#Predict days, for the next month
forecast_frequency='d'
forecast_periods=30

interval_width=0.80
include_history=True

# Função para ajustar hiperparâmetros e encontrar o melhor modelo
def tune_hyperparameters(history_pd, holidays):
    params_grid = {
        'growth':['logistic'],
        'seasonality_mode': ['additive'],
        'yearly_seasonality': [True],
        'weekly_seasonality': [True],
        'daily_seasonality': [False],
        'changepoint_prior_scale': [0.01, 0.05, 0.08, 0.1, 0.3],
        'holidays_prior_scale': [0.1, 0.2, 0.3, 0.4, 0.5],
        'n_changepoints': [100, 150, 200]
    }
    grid = ParameterGrid(params_grid)
    
    # Dividir os dados em treinamento e teste
    end_date = pd.to_datetime('today').normalize()  # Data atual
    start_date = end_date - pd.DateOffset(days=31)  # Últimos 30 dias antes da data atual

    mask1 = (history_pd['ds'] < start_date)
    X_tr = history_pd.loc[mask1]
    
    mask2 = (history_pd['ds'] >= start_date) & (history_pd['ds'] < end_date)
    X_tst = history_pd.loc[mask2]

    model_parameters = pd.DataFrame(columns=['MAPE', 'Parameters'])
    
    for p in grid:
        train_model = Prophet(
            changepoint_prior_scale=p['changepoint_prior_scale'],
            holidays_prior_scale=p['holidays_prior_scale'],
            n_changepoints=p['n_changepoints'],
            seasonality_mode=p['seasonality_mode'],
            weekly_seasonality=p['weekly_seasonality'],
            yearly_seasonality=p['yearly_seasonality'],
            holidays=holidays,
            interval_width=0.80
        )
        X_tr['floor'] = 0

        train_model.fit(X_tr)
        train_forecast = train_model.make_future_dataframe(periods=30, freq='D', include_history=False)

        train_forecast['floor'] = 0
    

        train_forecast = train_model.predict(train_forecast)
        test = train_forecast[['ds', 'yhat']]

        Actual = X_tst[['ds', 'y']]

        Actual = Actual.set_index('ds')
        MAPE = np.mean(np.abs((Actual['y'] - abs(test.set_index('ds')['yhat'])) / Actual['y']) * 100)
        model_parameters = model_parameters.append({'MAPE': MAPE, 'Parameters': p}, ignore_index=True)
    
    # Encontrar os melhores parâmetros
    best_params = model_parameters.sort_values(by=['MAPE']).reset_index(drop=True).iloc[0]
    best_model_params = best_params['Parameters']

    # print("Melhores Parâmetros Encontrados:")
    # print(best_model_params)

    # Treinar o modelo com os melhores parâmetros
    best_model = Prophet(
        changepoint_prior_scale=best_model_params['changepoint_prior_scale'],
        holidays_prior_scale=best_model_params['holidays_prior_scale'],
        n_changepoints=best_model_params['n_changepoints'],
        seasonality_mode=best_model_params['seasonality_mode'],
        weekly_seasonality=best_model_params['weekly_seasonality'],
        daily_seasonality=best_model_params['daily_seasonality'],
        yearly_seasonality=best_model_params['yearly_seasonality'],
        holidays=holidays,
        interval_width=0.80
    )

    history_pd['floor'] = 0

    best_model.fit(history_pd)
    return best_model, best_model_params

# Função para gerar previsões
def generate_forecast(history_pd, display_graph=True):
    # Exemplo de dados de feriados (substitua por sua tabela real)
    feriados_regionais = pd.DataFrame({
        'ds': pd.to_datetime([
            '2023-08-15', '2023-08-31', '2024-08-15', '2024-08-31', '2025-08-15', '2025-08-31'
        ]),
        'holiday': [
            'Nossa Senhora de Abadia', 'Aniversário de Uberlândia', 
            'Nossa Senhora de Abadia', 'Aniversário de Uberlândia',
            'Nossa Senhora de Abadia', 'Aniversário de Uberlândia'
        ]
    })

    # Feriados personalizados adicionais
    feriados_personalizados = spark.sql("select DATA as ds, holiday from default.feriados_previsao_varejo").toPandas()

    # Combine os feriados regionais e personalizados
    holidays = pd.concat([feriados_regionais, feriados_personalizados], ignore_index=True)

    # Remover valores ausentes
    history_pd = history_pd.dropna()

    # Excluir o dia atual se estiver incompleto
    today = pd.to_datetime('today').normalize()
    history_pd = history_pd[history_pd['ds'] < today]
    
    # Definir o modelo com hiperparâmetros otimizados
    best_model, best_params = tune_hyperparameters(history_pd, holidays)
    
    # Fazer previsões com o melhor modelo
    future_pd = best_model.make_future_dataframe(periods=forecast_periods, freq=forecast_frequency, include_history=include_history)

    # print(future_pd.head())
    future_pd['floor'] = 0

    forecast_pd = best_model.predict(future_pd)

    if display_graph:
        best_model.plot(forecast_pd)
        plt.show()

    # Adicionar as previsões ao histórico
    f_pd = forecast_pd[['ds', 'yhat', 'yhat_upper', 'yhat_lower']].set_index('ds')
    results_pd = f_pd.join(history_pd[['ds','y','dbus']].set_index('ds'), how='left')
    results_pd.reset_index(level=0, inplace=True)
    results_pd['ds'] = results_pd['ds'].dt.date
    results_pd['sku'] = history_pd['sku'].iloc[0]
    results_pd['workspace_id'] = history_pd['workspace_id'].iloc[0]

    return results_pd


# COMMAND ----------

#Sum all the SKUs & Workspace for global consumption trend (by default we want a view for all our billing usage across all workspace, so we need to train a specific model on that too)
global_forecast = data_to_predict_daily.groupBy(col("ds")).agg(sum('custo').alias("y"), sum('dbus').alias("dbus")) \
                                       .withColumn('sku', lit('ALL')) \
                                       .withColumn('workspace_id', lit('ALL')).toPandas()


global_forecast = generate_forecast(global_forecast)

# spark.createDataFrame(global_forecast).withColumn('training_date', current_date()) \
                                    #   .write.mode('overwrite').option("mergeSchema", "true").saveAsTable("billing_forecast")


custodb_forecast = spark.createDataFrame(global_forecast).select(\
                                    col("ds").alias("data"),\
                                    col("yhat").alias("previsao"),\
                                    col("y").alias("real"),\
                                    col("yhat_upper").alias("previsao_superior"),\
                                    when(col("yhat_lower") < 0, 0).otherwise(col("yhat_lower")).alias("previsao_inferior")
                                    )

# COMMAND ----------

# custodb_forecast.write.format('delta').mode('overwrite').saveAsTable(f"gold_prod.custodb_forecast")

# Filtrar os dados para update (onde a data é maior ou igual ao dia atual)
custodb_forecast_filtered = custodb_forecast.filter(col("data") >= current_date())

# Especificar a tabela de destino
delta_table = DeltaTable.forName(spark, "gold_prod.custodb_forecast")

# Realizar o merge: atualizar as linhas onde a data é maior ou igual ao dia atual, ou inserir novas
delta_table.alias("target").merge(
    custodb_forecast_filtered.alias("source"),
    "target.data = source.data"
).whenMatchedUpdateAll(
).whenNotMatchedInsertAll(
).execute()
