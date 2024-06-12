# Databricks notebook source
import pyspark.sql.functions as F
import pandas as pd
import numpy as np
from datetime import datetime

import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error
import xgboost as xgb

# COMMAND ----------

def abt_previsao():
    df = spark.sql("""
    WITH 
    -- TABELA SELECAO MESMOS CLI
    tb_clival_periodo as
    (
    select CODLOJASITEF, 
    COUNT(DISTINCT NUMANOMES) AS QTD_MESES
    from bronze.resumofat_tef
    WHERE 
    date_format(TO_DATE(CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(DATA_TRN, 'yyyyMMdd')) AS TIMESTAMP)), "yyyyMM") < date_format(current_date(), "yyyyMM")
    group by CODLOJASITEF
    )
    ,tb_meses_total as (
    SELECT COUNT(DISTINCT NUMANOMES) AS QTD_MESES_TOTAL
    from bronze.resumofat_tef
    WHERE 
    date_format(TO_DATE(CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(DATA_TRN, 'yyyyMMdd')) AS TIMESTAMP)), "yyyyMM") < date_format(current_date(), "yyyyMM")
    )
    ,tb_mesmos_cli as(
    select DISTINCT
    t2.CODLOJASITEF as codcli,
    t1.DESUNDNGCCLI
    from bronze.dimcliend t1 
    left join tb_clival_periodo t2 on (t2.CODLOJASITEF = t1.codcliend)
    INNER JOIN tb_meses_total t3 ON (t2.QTD_MESES = t3.QTD_MESES_TOTAL OR t1.DESUNDNGCCLI = 'SMART')
    WHERE t1.DESUNDNGCCLI in ('VAREJO ALIMENTAR','SMART')
    )

    -- BASE DE PERIODOS ABT
    ,tb_semana_pascoa as 
    (SELECT tpod.numero_do_ano_mes,(tpod.Semanas_do_Ano - 1) as Semanas_do_Ano, 1 as Flag
                FROM gold_prod.dim_pod tpod
                INNER JOIN default.feriados_previsao_varejo tpas ON (tpod.Data = tpas.DATA) 
                WHERE tpas.holiday = 'Pascoa'
    )
    ,tb_pos_vesp_sexta as 
    (SELECT (tpod.DATA - 1) AS VESPERA_SEXTA,
            (tpod.DATA + 1) AS POS_SEXTA
    FROM gold_prod.dim_pod tpod
    INNER JOIN default.feriados_previsao_varejo tpas ON (tpod.Data = tpas.DATA) 
    WHERE tpod.numero_do_dia_da_semana = 6 
    AND tpas.holiday is not null
    ) 
    ,tb_bridge as 
    (SELECT 
        (CASE WHEN (tpod.numero_do_dia_da_semana =  5 AND tpas.holiday is not null) THEN (tpod.DATA + 1) END) AS BRIDGE_SEXTA,
        (CASE WHEN (tpod.numero_do_dia_da_semana =  3 AND tpas.holiday is not null) THEN (tpod.DATA - 1) END) AS BRIDGE_SEGUNDA
    FROM gold_prod.dim_pod tpod
    INNER JOIN default.feriados_previsao_varejo tpas ON (tpod.Data = tpas.DATA) 
    WHERE (tpod.numero_do_dia_da_semana =  5 AND tpas.holiday is not null)
    OR    (tpod.numero_do_dia_da_semana =  3 AND tpas.holiday is not null)
    )
    ,tb_vespera as 
    (SELECT 
    (tpod.DATA - 1) AS VESPERA
    FROM gold_prod.dim_pod tpod
    INNER JOIN default.feriados_previsao_varejo tpas ON (tpod.Data = tpas.DATA) 
    )
    ,tb_geral_pod as 
    (
    select 
        t1.Data AS DATA,
        t1.Dia as DIA_MES,
        (CASE WHEN t2.holiday IS NULL THEN 0 ELSE 1 END) AS FERIADO,
        (CASE WHEN t2.holiday = 'Pascoa' OR t3.Flag is not null THEN 1 ELSE 0 END) AS SEMANA_PASCOA,
        (CASE WHEN t4A.VESPERA_SEXTA IS NOT NULL THEN 1 ELSE 0 END) AS VESPERA_SEXTA,
        (CASE WHEN t4B.POS_SEXTA IS NOT NULL THEN 1 ELSE 0 END) AS POS_SEXTA,
        (CASE WHEN t5A.BRIDGE_SEXTA IS NOT NULL OR t5B.BRIDGE_SEGUNDA IS NOT NULL THEN 1 ELSE 0 END) AS BRIDGE,
        t1.flag_primeiro_dia_do_mes AS PRIMEIRO_DIA,
        t1.flag_ultimo_dia_do_mes AS ULTIMO_DIA,
        (CASE WHEN t6.VESPERA IS NOT NULL THEN 1 ELSE 0 END) AS VESPERA,
        (CASE WHEN t1.numero_do_dia_da_semana = 1 THEN 1 ELSE 0 END) AS DOM,
        (CASE WHEN t1.numero_do_dia_da_semana = 2 THEN 1 ELSE 0 END) AS SEG,
        (CASE WHEN t1.numero_do_dia_da_semana = 3 THEN 1 ELSE 0 END) AS TER,
        (CASE WHEN t1.numero_do_dia_da_semana = 4 THEN 1 ELSE 0 END) AS QUA,
        (CASE WHEN t1.numero_do_dia_da_semana = 5 THEN 1 ELSE 0 END) AS QUI,
        (CASE WHEN t1.numero_do_dia_da_semana = 6 THEN 1 ELSE 0 END) AS SEX,
        (CASE WHEN t1.numero_do_dia_da_semana = 7 THEN 1 ELSE 0 END) AS SAB,
        t1.numero_da_semana_no_mes AS SEMANA_MES,
        (CASE WHEN t1.numero_da_semana_no_mes = 1 THEN 1 ELSE 0 END) AS SEMANA_1,
        (CASE WHEN t1.numero_da_semana_no_mes = 2 THEN 1 ELSE 0 END) AS SEMANA_2,
        (CASE WHEN t1.numero_da_semana_no_mes = 3 THEN 1 ELSE 0 END) AS SEMANA_3,    
        (CASE WHEN t1.numero_da_semana_no_mes = 4 THEN 1 ELSE 0 END) AS SEMANA_4,
        (CASE WHEN t1.numero_da_semana_no_mes = 5 THEN 1 ELSE 0 END) AS SEMANA_5,
        (CASE WHEN t2.holiday = 'Vespera Confraternizacao Universal' THEN 1 ELSE 0 END) AS VESPERA_ANO_NOVO,
        (CASE WHEN t2.holiday = 'Black Thursday' THEN 1 ELSE 0 END) AS BLACK_THURSDAY,
        (CASE WHEN t2.holiday = 'Dia do Trabalho' THEN 1 ELSE 0 END) AS DIA_DO_TRABALHO,
        (CASE WHEN t2.holiday = 'Carnaval1' THEN 1 ELSE 0 END) AS CARNAVAL1,
        (CASE WHEN t2.holiday = 'Nossa Sr.a Aparecida - Padroeira do Brasil' THEN 1 ELSE 0 END) AS APARECIDA,
        (CASE WHEN t2.holiday = 'Pascoa' THEN 1 ELSE 0 END) AS PASCOA,
        (CASE WHEN t2.holiday = 'Paixao de Cristo' THEN 1 ELSE 0 END) AS PAIXAO_CRISTO,
        (CASE WHEN t2.holiday = 'Black Friday' THEN 1 ELSE 0 END) AS BLACK_FRIDAY,
        (CASE WHEN t2.holiday = 'Tiradentes' THEN 1 ELSE 0 END) AS TIRADENTES,
        (CASE WHEN t2.holiday = 'Sab. Pascoa' THEN 1 ELSE 0 END) AS Sab_Pascoa,
        (CASE WHEN t2.holiday = 'Qui. Paixao' THEN 1 ELSE 0 END) AS Qui_Paixao,
        (CASE WHEN t2.holiday = 'Vespera Dia das Maes' THEN 1 ELSE 0 END) AS VESPERA_MAES,
        (CASE WHEN t2.holiday = 'Carnaval2' THEN 1 ELSE 0 END) AS CARNAVAL2,
        (CASE WHEN t2.holiday = 'Confraternizacao Universal' THEN 1 ELSE 0 END) AS ANO_NOVO,
        (CASE WHEN t2.holiday = 'Carnaval3' THEN 1 ELSE 0 END) AS CARNAVAL3,
        (CASE WHEN t2.holiday = 'Vespera de Natal' THEN 1 ELSE 0 END) AS VESPERA_NATAL,
        (CASE WHEN t2.holiday = 'Proclamacao da Republica' THEN 1 ELSE 0 END) AS PROC_REPUBLICA,
        (CASE WHEN t2.holiday = 'Dia dos Pais' THEN 1 ELSE 0 END) AS DIA_DOS_PAIS,
        (CASE WHEN t2.holiday = 'Natal' THEN 1 ELSE 0 END) AS NATAL,
        (CASE WHEN t2.holiday = 'Corpus Christi' THEN 1 ELSE 0 END) AS CORPUS_CHRISTI,
        (CASE WHEN t2.holiday = 'Independencia do Brasil' THEN 1 ELSE 0 END) AS INDEPENDENCIA,
        (CASE WHEN t2.holiday = 'Dia das Maes' THEN 1 ELSE 0 END) AS DIA_DAS_MAES,
        (CASE WHEN t2.holiday = 'Finados' THEN 1 ELSE 0 END) AS FINADOS,
        t1.numero_do_ano_mes AS ANOMES
    from gold_prod.dim_pod t1
    left join default.feriados_previsao_varejo t2 ON (t1.Data = t2.DATA)
    left join tb_semana_pascoa t3 ON (t1.numero_do_ano_mes = t3.numero_do_ano_mes AND t1.Semanas_do_Ano = t3.Semanas_do_Ano)
    left join tb_pos_vesp_sexta t4A ON (t1.DATA = t4A.VESPERA_SEXTA)
    left join tb_pos_vesp_sexta t4B ON (t1.DATA = t4B.POS_SEXTA)
    left join tb_bridge t5A ON (t1.DATA = t5A.BRIDGE_SEXTA)
    left join tb_bridge t5B ON (t1.DATA = t5B.BRIDGE_SEGUNDA)
    left join tb_vespera t6 ON (t1.DATA = t6.VESPERA)
    WHERE t1.Data BETWEEN DATE("2021-07-01") and last_day(current_date()) 
    )
    -- VARIAVEIS RELACIONADAS A VALOR
    ,tb_valor_dia AS (
    select t1.DATA_TRN,
    ROUND(SUM(CASE WHEN t2.DESUNDNGCCLI = 'SMART' AND t1.TIPO_TRANSACAO = 'Compra Pix' THEN t1.VALOR ELSE 0 END),2) AS VLR_SMART_PIX,
    ROUND(SUM(CASE WHEN t2.DESUNDNGCCLI = 'VAREJO ALIMENTAR' AND t1.TIPO_TRANSACAO = 'Compra Pix' THEN t1.VALOR ELSE 0 END),2) AS VLR_TOTAL_PIX,
    ROUND(SUM(CASE WHEN t2.DESUNDNGCCLI = 'VAREJO ALIMENTAR' THEN t1.VALOR ELSE 0 END),2) AS VALOR_TOTAL,
    ROUND(SUM(CASE WHEN t2.DESUNDNGCCLI = 'SMART' THEN t1.VALOR ELSE 0 END),2) AS VALOR_SMART_TOTAL
    from bronze.resumofat_tef t1 INNER JOIN tb_mesmos_cli T2 ON (t1.CODLOJASITEF = t2.codcli)
    GROUP BY ALL
    )

    -- ABT
    SELECT  t1.*,
            (CASE WHEN t2.VLR_TOTAL_PIX IS NOT NULL THEN 1 ELSE 0 END) AS PIX_CATEG,
            COALESCE(t2.VLR_SMART_PIX,0) AS VLR_SMART_PIX,
            COALESCE(t2.VLR_TOTAL_PIX,0) AS VLR_TOTAL_PIX,   
            COALESCE(t2.VALOR_TOTAL,0) AS VALOR_TOTAL,
            COALESCE(t2.VALOR_SMART_TOTAL,0) AS VALOR_SMART_TOTAL
    FROM tb_geral_pod t1
    LEFT JOIN tb_valor_dia t2 ON (t1.DATA = to_date(t2.DATA_TRN, 'yyyyMMdd'))
    ORDER BY t1.DATA ASC
    """)
    return(df)

# COMMAND ----------

def limites_outlier(X_train, label = None):
    if label == 'VALOR_SMART_TOTAL':
        LI = (X_train[label]).quantile(0.05)
        LS = (X_train[label]).quantile(0.95)
        # df["OUTLIER"] = np.where(df[label]>LS,1,np.where(df[label]<LI,-1,0))

    elif label == 'VALOR_TOTAL':
        Q1 = (X_train[label]).quantile(0.25)
        Q3 = (X_train[label]).quantile(0.75)
        IQR = Q3 - Q1
        LS = Q3 + 1.5*(IQR)
        LI = Q1 - 1.5*(IQR)
    return(LI,LS)

def create_features(X_train, LI, LS, label=None):
    X_train = X_train.copy()
    X_train['DATA'] = pd.to_datetime(X_train['DATA'])
    y_train = X_train[label]
    X_train['DIA_SEMANA'] = X_train['DATA'].dt.dayofweek
    X_train.loc[(X_train['DATA'].dt.month == 12) & (X_train['DATA'].dt.day.isin([24, 31])), 'DIA_SEMANA'] = -10
    X_train.loc[(((X_train['DATA'].dt.month == 1) & (X_train['DATA'].dt.day.isin([1]))) | ((X_train['DATA'].dt.month == 12) & (X_train['DATA'].dt.day.isin([25])))), 'DIA_SEMANA'] = 10
    X_train['TRIMESTRE'] = X_train['DATA'].dt.quarter
    X_train['MES'] = X_train['DATA'].dt.month
    X_train['ANO'] = X_train['DATA'].dt.year
    X_train['DIA_ANO'] = X_train['DATA'].dt.dayofyear
    X_train['DIA_MES'] = X_train['DATA'].dt.day
    X_train.loc[((X_train['DATA'].dt.month == 1) & (X_train['DATA'].dt.day.isin([1]))), 'DIA_MES'] = 10
    X_train['SEMANA_ANO'] = X_train['DATA'].dt.isocalendar().week.astype(int)
    X_train['TER'] = (X_train['DATA'].dt.dayofweek == 1).astype(int)
    X_train['QUA'] = (X_train['DATA'].dt.dayofweek == 2).astype(int)
    X_train['QUI'] = (X_train['DATA'].dt.dayofweek == 3).astype(int)
    X_train['SEX'] = (X_train['DATA'].dt.dayofweek == 4).astype(int)
    X_train['SAB'] = (X_train['DATA'].dt.dayofweek == 5).astype(int)
    X_train['DOM'] = (X_train['DATA'].dt.dayofweek == 6).astype(int)
    X_train.loc[(X_train['DATA'].dt.month == 12) & (X_train['DATA'].dt.day.isin([24, 31])), 'DOM'] = 0
    X_train['SEG'] = (X_train['DATA'].dt.dayofweek == 0).astype(int)
    X_train["OUTLIER"] = np.where(y_train>LS,1,np.where(y_train<LI,-1,0))
    X_train.loc[(X_train['DATA'].dt.month == 12) & (X_train['DATA'].dt.day.isin([24, 31])), 'OUTLIER'] = 1
    X_train.loc[(((X_train['DATA'].dt.month == 1) & (X_train['DATA'].dt.day.isin([1]))) | ((X_train['DATA'].dt.month == 12) & (X_train['DATA'].dt.day.isin([25])))), 'OUTLIER'] = -10
    X_train.loc[((X_train['DATA'].dt.month == 1) & (X_train['DATA'].dt.day.isin([1]))), 'PRIMEIRO_DIA'] = -10
    X_train['DECORRIDO'] = (pd.to_datetime(max(X_train['DATA'])) - pd.to_datetime(X_train['DATA'])).dt.days

    if label == 'VALOR_SMART_TOTAL':
        X_train = X_train[['DIA_SEMANA','TRIMESTRE','MES','ANO','DIA_ANO','DIA_MES','SEMANA_ANO','FERIADO','ULTIMO_DIA',
                'PRIMEIRO_DIA','SEMANA_MES','VESPERA','BRIDGE','ANO_NOVO','CARNAVAL1','CARNAVAL2','CARNAVAL3',
                'PAIXAO_CRISTO','PASCOA','TIRADENTES','DIA_DO_TRABALHO','VESPERA_MAES','DIA_DAS_MAES',
                'CORPUS_CHRISTI','DIA_DOS_PAIS','INDEPENDENCIA','APARECIDA','FINADOS','PROC_REPUBLICA',
                'BLACK_THURSDAY','BLACK_FRIDAY','VESPERA_NATAL','NATAL','VESPERA_ANO_NOVO','OUTLIER',
                'POS_SEXTA','VESPERA_SEXTA','SEX','SAB','DOM','PIX_CATEG']]
    elif label == 'VALOR_TOTAL':
        X_train = X_train[['DIA_SEMANA','TRIMESTRE','MES','ANO','DIA_ANO','DIA_MES','SEMANA_ANO','FERIADO','ULTIMO_DIA',
                'PRIMEIRO_DIA','SEMANA_MES','BRIDGE','VESPERA','ANO_NOVO','CARNAVAL1','CARNAVAL2','CARNAVAL3',
                'PAIXAO_CRISTO','PASCOA','TIRADENTES','DIA_DO_TRABALHO','VESPERA_MAES','DIA_DAS_MAES',
                'CORPUS_CHRISTI','DIA_DOS_PAIS','INDEPENDENCIA','APARECIDA','FINADOS','PROC_REPUBLICA',
                'BLACK_THURSDAY','BLACK_FRIDAY','VESPERA_NATAL','NATAL','VESPERA_ANO_NOVO','OUTLIER',
                'SEX','SAB','DOM','SEMANA_PASCOA','POS_SEXTA','VESPERA_SEXTA','PIX_CATEG']]
    else:
        raise ValueError('Invalid label')


    return X_train,y_train

def previsao(X_train,X_test,parameters,scoring,label=None,Metric = 'MAPE'):
    LI,LS = limites_outlier(X_train,label)
    X_train,y_train = create_features(X_train,LI,LS,label)
    X_test,y_test = create_features(X_test,LI,LS,label)
    
    if label == 'VALOR_SMART_TOTAL':
        X_test['OUTLIER'] = 0
        with mlflow.start_run():

            mlflow.sklearn.autolog()
            grid_search = GridSearchCV(xgb.XGBRegressor(), parameters, scoring=scoring, refit=Metric, cv=5)
            grid_search.fit(X_train, y_train)
            best_index = grid_search.best_index_
            metrics_model = {"MAPE":grid_search.cv_results_['mean_test_MAPE'][best_index],
                        "RMSE":grid_search.cv_results_['mean_test_RMSE'][best_index],
                        "R2":grid_search.cv_results_['mean_test_R2'][best_index],
                        "MAE":grid_search.cv_results_['mean_test_MAE'][best_index],   
                        "MSE":grid_search.cv_results_['mean_test_MSE'][best_index]
                        }
        mlflow.log_metrics(metrics_model)

        mlflow.end_run()
        # importance = xgb.plot_importance(grid_search.best_estimator_, height=0.5)
        Previsto = (grid_search.best_estimator_).predict(X_test)

    elif label == 'VALOR_TOTAL':
        with mlflow.start_run():

            mlflow.sklearn.autolog()
            grid_search = GridSearchCV(xgb.XGBRegressor(), parameters, scoring=scoring, refit=Metric, cv=5)
            grid_search.fit(X_train, y_train)
            best_index = grid_search.best_index_
            metrics_model = {"MAPE":grid_search.cv_results_['mean_test_MAPE'][best_index] * -1,
                        "RMSE":grid_search.cv_results_['mean_test_RMSE'][best_index] * -1,
                        "R2":grid_search.cv_results_['mean_test_R2'][best_index],
                        "MAE":grid_search.cv_results_['mean_test_MAE'][best_index] * -1,   
                        "MSE":grid_search.cv_results_['mean_test_MSE'][best_index] * -1
                        }
        mlflow.log_metrics(metrics_model)

        mlflow.end_run()
        # importance = xgb.plot_importance(grid_search.best_estimator_, height=0.5)
        Previsto = (grid_search.best_estimator_).predict(X_test)

    return Previsto