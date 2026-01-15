# Storm Damage Prediction

![](assets/test3.png)

![](assets/test1.png)

A machine learning model that predicts property damage from natural disasters. Using NOAA storm data, we use various pre-processing and curation decisions to achieve ~70% R^2 accuracy in testing.

https://stormprediction.pages.dev

## Features
- Predicts property damage in USD based on storm details
- Utilizes XGBoost to achieve 70% R^2 accuracy
- Processes user descriptions using sentence embeddings
- Handles various storm types: Thunderstorms, Tornadoes, Floods, Hurricanes and 26 more!

## Details
- XGBoost Regressor
- R^2 Score: ~70%
- Data from NOAA Storm Events Database
- Range: $0-$50,000

![](assets/test2.png)