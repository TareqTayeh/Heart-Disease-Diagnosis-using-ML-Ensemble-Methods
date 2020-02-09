# Heart Disease Diagnosis

Heart disease affects many people in Canada and around the world. Currently, in Canada, more than 33,600 people die each year as a result of heart disease, making it one of the leading causes of death. If a patient could be identified as someone at risk for developing heart disease, then preventative measures could be taken to lower their risk, including quitting smoking, getting regular exercise, eating healthy and quitting/limiting alcohol consumption. Additionally, if the patient is identified to be at risk for developing heart disease, they can be monitored for high blood pressure and high cholesterol, and these factors can be controlled with proper health care to help prevent the development of heart disease.  

Our goal was to accurately detect heart disease in a patient using various machine learning methods and techniques. We used a dataset that contains information on 303 patients with 14 attributes to achieve our goal. This data was provided from “V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.” (Cleveland Database) on Heart Disease in the UCI Machine Learning Repo (https://archive.ics.uci.edu/ml/datasets/Heart+Disease). The used dataset file in this project is called `heart_dataset_complete.csv`, and can be found under the `Dataset` folder.

The following table highlights the data attributes, descriptions and types:

| #          | Attribute | Description                                                                                                                                                            | Type                  |
|:----------:|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| 1          | age       | Age in Years                                                                                                                                                           | Integer               |
| 2          | sex       | Sex in Binary: `0 = Female, 1 = Male`                                                                                                                                  | Binary                |
| 3          | cp        | Chest Pain :  `1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal pain, 4 = Asymptomatic`                                                                        | Integer / Categorical |
| 4          | trestbps  | Resting Blood Pressure in mm Hg (millimeters of Mercury) on admission to the hospital                                                                                  | Integer / Continuous  |
| 5          | chol      | Serum Cholesterol in mg/dl (milligrams per deciliter)                                                                                                                  | Integer / Continuous  |
| 6          | fbs       | Fasting Blood Sugar: `0 = < 120mg/dl, 1 = > 120mg/dl`                                                                                                                 | Binary                |
| 7          | restecg   | Resting electrocardiographic results: `0 =  Normal, 1 =  Having ST-T wave abnormality,  2= Showing probable or definite left ventricular hypertrophy by Estes' criteria` | Integer / Categorical |
| 8          | thalach   | Maximum heart rate achieved during thallium stress test                                                                                                                | Integer / Continuous  |
| 9          | exang     | Exercise induced angina: `0 = No, 1 = Yes `                                                                                                                              | Binary                |
| 10         | oldpeak   | ST depression induced by exercise relative to rest                                                                                                                     | Integer / Continuous  |
| 11         | slope     | Slope of peak exercise ST segment: `0 = Downsloping, 1 = Flat, 2 = Upsloping`                                                                                           | Integer / Categorical |
| 12         | ca        | Number of major vessels colored by fluoroscopy: `0-3 vessels`                                                                                                            | Integer / Categorical |
| 13         | thal      | Thallium stress test result: `1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect`                                                                                       | Integer / Categorical |
| 14         | target    | Heart disease present or not: `0 = No Heart Disease, 1 = Heart Disease`                                                                                                  | Binary                |

First of all, the data was pre-processed via data editing, cleansing, generation and normalization techniques. Feature selection techniques of intrinsic discrepancy, correlation coefficients, and permutation importance were then applied onto the data set to eliminate the weakly correlated and unimportant features. 4 machine learning models were then applied on the selected features and their values; Logistic Regression, Support Vector Machine, Neural Network, and Naive Bayes. The models were tuned and implemented using the  Scikit, Keras, and Tenserflow packages in Python. Afterwards, an ensemble learning method was implemented, which combined the results of the 4 models before using a voting scheme to produce the final output. That yielded better accuracy and computation time than any of the other models run solo. 5 runs of the 5-fold cross validation validation method was in place for each method, providing a more random train/test split procedure and less skewed results. Each model's results were averaged across all runs then compared with each other using evaluation metrics, which included accuracy, recall, precision, f1-score, and a confusion matrix.

You can find our written IEEE format journal paper for this project "Heart Disease Diagnosis Using Machine Learning.pdf" under the main directory. The paper is organized as follows: 
• Section I: Introduction <br />
• Section II: Background & Evaluation Metrics <br />
• Section III: Related Work <br />
• Section IV: Data Set <br />
• Section V: Data Pre-Processing <br />
• Section VI: Feature Selection Techniques <br />
• Section VII: Validation Process <br />
• Section VIII: Implementations <br />
• Section IX: Results <br />
• Section X: Comparison of Results <br />
• Section XI: Conclusions
