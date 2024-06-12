# Portifolio

This portfolio has copyrighted data science and machine learning codes for real examples

## Content

- Classifier (SVM, PassiveAgressiveClassifier) 
- Regression (XGBRegressor)

### [Food retail forecast and network REDE](https://github.com/Gpaiva2814/Machine-Learning-Portifolio/tree/29667d563b7aa2b68f6d4bc639928fd6bce2f4cc/PrevisaoVarejoSmart)

Sales forecasting in the food retail industry is crucial for strategic planning, enabling better inventory management, resource optimization, and profit maximization. Accurate analysis and data allow for anticipating seasonal demands, adjusting marketing strategies, and providing a more satisfying shopping experience for customers. Sales forecasting is a powerful ally in the quest for efficiency and competitiveness in the food retail sector. In this analysis, we used various powerful tools and packages to ensure accurate forecasts.

We utilized XGBoost for its high performance and accuracy in regression tasks. Model tuning and evaluation were conducted using scikit-learn, specifically employing GridSearchCV for hyperparameter optimization and various metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and the coefficient of determination (R2) for model evaluation.

To ensure transparency and reproducibility of machine learning experiments, we used MLflow for tracking experiments, logging metrics, and saving models. The integration of these tools allowed us to build robust models for both general food retail data and a specific supermarket chain (SMART), each treated with customized models to accommodate their unique characteristics.

In production, the model will provide three key pieces of information: the initial forecast generated on the 1st of each month, an adjusted forecast with the model retrained using data up to the day before (d-1), and the actual realized value to monitor performance.

Here's a summary of the tools and libraries used:

- MLflow: For experiment tracking, model management, and reproducibility.
- Scikit-learn: For machine learning, including model evaluation and hyperparameter tuning.
- XGBoost: For building and training the regression model due to its performance and efficiency in handling large datasets.

This comprehensive approach, combining advanced data processing and machine learning techniques, enabled us to create accurate and reliable sales forecasts for the food retail industry, enhancing strategic decision-making and operational efficiency.

###### Model results in production:
![Previsao_maio](https://github.com/Gpaiva2814/Machine-Learning-Portifolio/assets/123079404/017d2fd8-a820-4495-830f-40c4a0dae564)



### [GTIN Classifier](https://github.com/Gpaiva2814/Machine-Learning-Portifolio/blob/main/GTINClassifier.ipynb)

In Brazil there is no obligation to present the GTIN on invoices, a classification model based on machine learning that determines the possible GTIN (Global Trade Item Number) of a product from its description becomes useful for the
 correct identification of products, even without this mandatory information. Using the template, companies can keep accurate and reliable records, avoiding potential problems arising from missing information.

 ### [Market Classification (Section, Basket, Sub-basket)](https://github.com/Gpaiva2814/Machine-Learning-Portifolio/blob/main/Classificador_mercadologico.ipynb)

The division of market classifications into section, basket and sub-basket is fundamental for retail. This structured organization of products brings benefits such as a better shopping experience, performance analysis, strategic decision-making, customization of the product mix and more targeted communication. This division enables efficient management, allowing retailers to meet consumers' needs more effectively and increase their competitiveness in the market.
