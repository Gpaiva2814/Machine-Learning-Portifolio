# Databricks notebook source
# MAGIC %run "./Funcoes"

# COMMAND ----------

df = abt_previsao().toPandas()

first_zero_index = df[df['VALOR_TOTAL'] == 0].index.min()
df_train = df.loc[:first_zero_index -1 ]
df_pred = df.loc[first_zero_index:]

# COMMAND ----------

parameters = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'objective': ['reg:squarederror','reg:tweedie']
}
parameters_sma = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'objective': ['reg:squarederror']
}
#Metricas Escolha
scoring = {
    'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
    'MSE': make_scorer(mean_squared_error, greater_is_better=False),
    'RMSE': make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred))),
    'MAPE': make_scorer(mean_absolute_percentage_error, greater_is_better=False),
    'R2': make_scorer(r2_score)
}

# COMMAND ----------

Previsao_varejo  = previsao(df_train,df_pred,parameters,scoring,label='VALOR_TOTAL',Metric= 'MAPE')
Previsao_smart = previsao(df_train,df_pred,parameters_sma,scoring,label='VALOR_SMART_TOTAL',Metric= 'MAPE')

# COMMAND ----------

df_previsao = pd.DataFrame({'DATA': df_pred['DATA'],'PREVISAO': Previsao_varejo, 'PREVISAO_SMART': Previsao_smart})
df_previsao['DATA'] = pd.to_datetime(df_previsao['DATA']).dt.date
df['DATA'] = pd.to_datetime(df['DATA']).dt.date
df_previsao = pd.merge(df_previsao, df[['DATA','VALOR_TOTAL','VALOR_SMART_TOTAL']], on='DATA', how='left')
df_previsao_spark = spark.createDataFrame(df_previsao)

# Obter a data atual
current_day = datetime.now()

# Adicionar a nova coluna apenas se a data atual for dia 1
is_first_day = current_day.day == 1

if is_first_day:
    # Adicionar a nova coluna se for dia 1
    df_previsao_spark = df_previsao_spark.withColumn("PRIMEIRA_PREVISAO", F.lit(F.col("PREVISAO")))
    df_previsao_spark = df_previsao_spark.withColumn("PRIMEIRA_PREVISAO_SMART", F.lit(F.col("PREVISAO_SMART")))

# COMMAND ----------

columns = df_previsao_spark.columns
has_primeira_previsao = "PRIMEIRA_PREVISAO" in columns
has_primeira_previsao_smart = "PRIMEIRA_PREVISAO_SMART" in columns

df_original = spark.sql("SELECT * FROM gold_prod.previsao_mercado")
# Mesclar os DataFrames usando o método join
resultado = df_original.join(df_previsao_spark, "DATA", "outer") \
                .select(
                    "DATA",
                    F.coalesce(df_previsao_spark["PREVISAO"], df_original["PREVISAO"]).alias("PREVISAO"),
                    F.coalesce(df_previsao_spark["PREVISAO_SMART"], df_original["PREVISAO_SMART"]).alias("PREVISAO_SMART"),
                    F.coalesce(df_previsao_spark["VALOR_TOTAL"], df_original["VALOR_TOTAL"]).alias("VALOR_TOTAL"),
                    F.coalesce(df_previsao_spark["VALOR_SMART_TOTAL"], df_original["VALOR_SMART_TOTAL"]).alias("VALOR_SMART_TOTAL"),
                    F.coalesce(
                        F.col("PRIMEIRA_PREVISAO") if has_primeira_previsao else df_original["PRIMEIRA_PREVISAO"],
                        df_original["PRIMEIRA_PREVISAO"]
                    ).alias("PRIMEIRA_PREVISAO"),
                    F.coalesce(
                        F.col("PRIMEIRA_PREVISAO_SMART") if has_primeira_previsao_smart else df_original["PRIMEIRA_PREVISAO_SMART"],
                        df_original["PRIMEIRA_PREVISAO_SMART"]
                    ).alias("PRIMEIRA_PREVISAO_SMART")
                )

# COMMAND ----------

resultado.write.format('delta').mode('overwrite').saveAsTable(f"gold_prod.previsao_mercado")