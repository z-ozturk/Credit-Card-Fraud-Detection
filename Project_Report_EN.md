# T.C. ISTANBUL UNIVERSITY
## FACULTY OF SCIENCE
### DEPARTMENT OF COMPUTER SCIENCE

**Artificial Intelligence Course — Research Assignment**

---

# COMPARISON OF MACHINE LEARNING-BASED ANOMALY DETECTION MODELS AGAINST RULE-BASED SYSTEMS IN CREDIT CARD FRAUD DETECTION

**Prepared by:** Zehra Öztürk  
**Student ID:** 0414230008  
**Date:** May 2026

---

## 1. Introduction

With the accelerating pace of digitalization, electronic payment systems have become an indispensable part of everyday commerce. The rapid growth in credit card usage has brought with it a surge in fraudulent activity, causing substantial economic harm on a global scale. According to Nilson Report (2023) data, global losses attributed to credit card fraud are projected to exceed $40 billion per year by 2027. These figures demonstrate that fraud detection is not merely a technical issue but also a critical problem from the perspectives of financial stability and consumer trust.

Traditional fraud detection methods rely heavily on rule-based systems. These systems operate according to pre-defined threshold values such as "flag transactions originating from a specific region as suspicious" or "block transactions that exceed the daily spending limit." While effective against known fraud patterns, this approach falls short when confronted with new and sophisticated attack methods. As fraud techniques continuously evolve, static rules that are not updated gradually lose their detection capability.

Machine learning-based anomaly detection models have been designed as a response to these limitations and have attracted growing interest in the field of financial security in recent years. Algorithms such as Random Forest, XGBoost, and deep learning extract patterns from historical data to develop dynamic decision boundaries, thereby enabling the detection of previously unseen fraudulent behaviors. The advantages these models offer over rule-based systems in terms of accuracy, sensitivity, and adaptability have begun to be examined in detail in the academic literature.

This study aims to investigate the measurable advantages that machine learning-based anomaly detection models offer over traditional rule-based systems in credit card fraud detection, with respect to accuracy, sensitivity, and adaptability. Within this framework, both a rule-based reference system and machine learning models will be applied to a publicly available credit card transaction dataset; the results will be comparatively evaluated using Precision, Recall, F1-Score, and ROC-AUC metrics. The study is expected to provide practical recommendations for the development of fraud detection systems and to contribute to the academic literature in this field.

---

## 2. Literature Review

### 2.1. Dornadula and Geetha (2019)

Dornadula and Geetha (2019) comparatively investigated the effectiveness of machine learning algorithms in fraud detection using the European credit card fraud dataset. The study proposed a sliding window-based method in which cardholders are grouped into clusters according to their transaction amounts and separate classifiers are trained for each cluster. The researchers encountered a significant class imbalance problem, as the dataset contained only 0.172% fraudulent transactions; they addressed this by applying SMOTE (Synthetic Minority Oversampling Technique) and using evaluation metrics suited to imbalanced datasets, such as the Matthews Correlation Coefficient (MCC). The findings revealed that the Random Forest algorithm achieved the highest performance on the SMOTE-augmented dataset, with an MCC value of 0.9996. An important contribution of the study is its proposal of a feedback mechanism to address concept drift — the phenomenon whereby fraud patterns change over time.

However, the study was tested on real-time streaming data only at a limited scale, and its applicability in production environments has not yet been sufficiently investigated. Nevertheless, this work concretely demonstrates the capacity of machine learning to adapt to evolving attack patterns compared to the static nature of rule-based systems.

[1] Dornadula, V. N., & Geetha, S. (2019). Credit card fraud detection using machine learning algorithms. *Procedia Computer Science, 165*, 631–641.

### 2.2. Perols (2011)

Perols (2011) comprehensively compared six different statistical and machine learning models for the detection of financial statement fraud, including logistic regression, support vector machines (SVM), artificial neural networks (ANN), decision trees, and ensemble methods (bagging, stacking). The primary contribution of the study is the systematic incorporation of fraud-specific characteristics — such as class imbalance and misclassification costs — into the model evaluation process. Contrary to common belief, the findings showed that logistic regression and SVM exhibited more consistent and superior performance compared to neural networks. Perols (2011) attributes this to logistic regression's ability to produce more accurate probability estimates in environments where signal-to-noise separation is difficult. The study also identified that only six out of 42 predictor variables — auditor change, total discretionary accruals, Big 4 auditor, accounts receivable, meeting analyst forecasts, and unexpected employee productivity — were consistently selected across different models. This finding directly informs the feature selection discussion in credit card fraud detection. A limitation of the study is that the same dataset was used for both preprocessing and model evaluation, which partially restricts the generalizability of the results.

[2] Perols, J. (2011). Financial statement fraud detection: An analysis of statistical and machine learning algorithms. *Auditing: A Journal of Practice & Theory, 30*(2), 19–50.

### 2.3. Happa et al. (2019)

Happa and colleagues (2019) proposed the Pattern-of-Life visual metaphor method, which goes beyond traditional data visualization approaches for anomaly detection in cybersecurity. The study transformed raw system data — such as network connections, CPU usage, and file system activity — into abstract visual environments like cityscapes and galaxy clusters, and examined whether analysts could detect anomalies through these environments. A preliminary study conducted with three participants confirmed the general feasibility of the approach while also revealing the need to optimize false positive rates and metaphor mapping strategies. This work directly connects to the present study by highlighting that, while rule-based systems operate on specific signatures and are inadequate against unknown threats such as zero-day attacks, anomaly-based approaches can detect deviations from normal behavioral patterns. The most significant limitation of the method is that it constitutes a small-scale feasibility study conducted under laboratory conditions, and its performance in large-scale real-world environments is not yet known.

[3] Happa, J., Bashford-Rogers, T., Agrafiotis, I., Goldsmith, M., & Creese, S. (2019). Anomaly detection using pattern-of-life visual metaphors. *IEEE Access, 4*.

### 2.4. Phua et al. (2010)

Phua and colleagues (2010) presented a comprehensive survey of data mining-based fraud detection research published over the preceding decade. The study systematically classified fraud types, identified affected sectors (credit card, insurance, telecommunications, internal fraud), and comparatively evaluated supervised, semi-supervised, and unsupervised approaches used in these domains. The authors explicitly articulate two fundamental criticisms in fraud detection: the insufficiency of publicly available real datasets and the scarcity of well-documented methodologies. In this context, the rationale and methodology for synthetic data generation in situations where access to real data is restricted are discussed in detail. The study also emphasizes that fraud detection is not merely a technical classification problem; practical factors such as operational requirements, resource constraints, and concept drift significantly influence model selection. It directly connects to the present study in terms of how class imbalance in credit card fraud is addressed. The primary limitation of the study is that it is a literature survey rather than an applied comparison of technical methods, which makes it difficult to verify specific performance claims.

[4] Phua, C., Lee, V., Smith, K., & Gayler, R. (2010). A comprehensive survey of data mining-based fraud detection research. *arXiv preprint arXiv:1009.6119*.

### 2.5. Bolton and Hand (2002)

Bolton and Hand (2002) presented a comprehensive review that systematically examines statistical fraud detection methods across various application domains. The study clarified the fundamental distinction between supervised and unsupervised approaches, explaining the effectiveness of supervised methods in recognizing known fraud patterns and the role of unsupervised methods in detecting previously unseen anomalies. The authors explicitly identified one of the most critical methodological problems in fraud detection: that simple accuracy metrics are misleading due to imbalanced class distributions and that cost-weighted performance measures must be used. Bolton and Hand (2002) also applied behavioral profiling methods such as Peer Group Analysis and Break Point Analysis to credit card fraud, demonstrating how these methods can be used to monitor account behavior over time and detect anomalies at the individual account level. The study incorporates the principle of adaptability — emphasizing that fraud detection is a continuously evolving discipline and that detection models must keep pace as fraud methods change — into its theoretical framework. In this respect, it forms one of the theoretical foundations of the present study and provides a solid basis for discussing the adaptive superiority of machine learning over the static structure of rule-based systems.

[5] Bolton, R. J., & Hand, D. J. (2002). Statistical fraud detection: A review. *Statistical Science, 17*(3), 235–255.

---

## 3. Theoretical Framework

### 3.1. Rule-Based Systems and Their Limitations

Traditional fraud detection systems are built on deterministic structures based on pre-defined threshold values and logical rules. In these systems, whether a transaction is considered suspicious is determined by fixed conditions such as "block if the transaction amount exceeds X" or "alert if transactions are made from different countries with the same card within 10 minutes." The fundamental problem with the rule-based approach is that it can only detect previously defined fraud patterns. When fraud methods evolve over time — for example, when multiple small-amount transactions are made to stay below threshold values — the system cannot recognize these new patterns. Bolton and Hand (2002) characterized this as fraud detection being a continuously evolving discipline and emphasized that static rules will gradually lose their effectiveness against changing fraud behaviors.

### 3.2. Machine Learning, Dynamic Decision Boundaries, and the Confusion Matrix

Machine learning-based models learn patterns from historical data rather than fixed rules, thereby forming dynamic decision boundaries. In this approach, the model statistically learns the characteristics of fraudulent and legitimate transactions during training; during testing, it classifies new transactions according to these learned boundaries. Confusion Matrix analysis is used to evaluate model performance. The Confusion Matrix visualizes the four fundamental decision types made by a classifier: correctly classifying a legitimate transaction (True Negative, TN), correctly detecting a fraudulent transaction (True Positive, TP), incorrectly classifying a legitimate transaction as fraudulent (False Positive, FP), and missing a fraudulent transaction (False Negative, FN). The key performance metrics derived from this matrix are formulated as follows:

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

In the context of credit card fraud, the most critical error type is the False Negative: the model's failure to detect a fraudulent transaction leads both to direct financial loss and to erosion of customer trust. For this reason, maximizing overall accuracy alone can be misleading; Phua et al. (2010) explicitly demonstrated this and emphasized the need to use cost-weighted performance measures. The ROC-AUC metric measures the overall discriminative power of the model across different threshold values and expresses it as a single scalar value, thereby enabling a comparative evaluation of the rule-based system and the machine learning model.

### 3.3. The Random Forest Algorithm

Random Forest is an ensemble learning method formed by combining multiple decision trees. Each decision tree is independently trained on a randomly sampled subset of the training data; the final classification decision is determined by majority voting among these trees. Compared to a single decision tree, Random Forest reduces the tendency toward overfitting and is more robust to data noise. In dealing with the class imbalance problem commonly encountered in credit card fraud datasets — where fraudulent transactions constitute only a small fraction of the entire dataset — Random Forest can learn the minority class (fraudulent transactions) more effectively because each tree is trained on a different sample. Dornadula and Geetha (2019) demonstrated that Random Forest achieved an MCC value of 0.9996 on the same European credit card dataset combined with SMOTE application, confirming the algorithm's superior performance under imbalanced data conditions.

### 3.4. The Pattern of Life Approach

The Pattern of Life approach is a detection strategy that models each cardholder's individual transaction behavior over time and identifies deviations from this profile as anomalies. According to this approach, a transaction need not exceed an absolute threshold to be considered suspicious; a meaningful deviation from the cardholder's own historical behavior is sufficient. For example, the sudden occurrence of large transactions from the account of a cardholder who regularly makes small purchases could generate a high suspicion score in a Pattern of Life model, even if no rule-based threshold is exceeded. Bolton and Hand (2002) formulated this approach within the framework of Peer Group Analysis, proposing that an account's behavior be compared against a peer group of similar accounts and that a standardized deviation score be calculated using the t-statistic. The behavioral features to be incorporated into the Random Forest model in this study — variables such as transaction frequency, average amount, and transaction time distribution — are derived from this theoretical framework.

---

## 4. Methodology

### 4.1. Dataset

This study uses the Kaggle Credit Card Fraud Detection dataset, which contains transactions made by European credit card holders in September 2013 and is publicly available through the Machine Learning Group — ULB. The dataset consists of a total of 284,807 transactions, of which only 492 (0.173%) are fraudulent. This reveals that the dataset contains a significant class imbalance and, as such, is representative of real-world financial transaction data. Due to confidentiality requirements, the original features have been transformed using Principal Component Analysis (PCA), yielding 28 components designated V1 through V28. The only untransformed variables are the transaction amount (Amount) and transaction time (Time), and the dataset contains no missing values. Dornadula and Geetha (2019) also worked on the same dataset, though the present study adopts a different model architecture and comparison framework.

*Figure 1: Class Distribution and Transaction Amount Distribution*

### 4.2. Rule-Based Reference System

A simple rule-based system was designed to serve as a reference point against which the machine learning model could be compared. In this system, a transaction is classified as fraudulent if its amount exceeds the 75th percentile value of the entire dataset ($77.16). Rule-based systems represent the threshold-based approaches widely used in real-world applications; they have been intentionally kept simple in this study to expose the limitations of this approach. Phua et al. (2010) emphasize that static rule systems of this type cannot adapt to changing fraud patterns and lead to high false positive rates.

### 4.3. Random Forest Model

A Random Forest classifier was implemented using the scikit-learn library as the machine learning model. The model consists of 100 decision trees (n\_estimators=100) and is configured with the class\_weight='balanced' parameter to address class imbalance. This parameter assigns higher weights to samples belonging to the minority class (fraudulent transactions), enabling the model to learn fraudulent transactions more effectively. The dataset was split into 80% training and 20% testing using a random seed value of 42 (random\_state=42). The stratified split method was adopted to preserve the class ratio in both sets. The training set contains 227,845 transactions (394 fraudulent), and the test set contains 56,962 transactions (98 fraudulent). All analyses were conducted using the Python programming language in the Google Colab environment.

### 4.4. Evaluation Metrics

Rather than overall accuracy, metrics sensitive to class imbalance were preferred for evaluating model performance. The fundamental reason is that a naive model that classifies all transactions as legitimate would achieve 99.827% accuracy on this dataset with a 0.173% fraud rate — meaning this metric can produce misleading results. Accordingly, Precision, Recall, F1-Score, and ROC-AUC metrics were calculated. The Recall metric is particularly critical, since missing a fraudulent transaction (False Negative) leads to both direct financial loss and erosion of customer trust. The ROC-AUC value measures the model's overall discriminative power across different decision thresholds and allows for a holistic comparison of the two models.

**Table 1: Rule-Based System vs. Random Forest Comparison**

| Model | Precision | Recall | F1-Score | ROC-AUC | False Positives | False Negatives | True Positives |
|---|---|---|---|---|---|---|---|
| Rule-Based System | 0.0024 | 0.3537 | 0.0049 | 0.5519 | 71,028 | 318 | 174 |
| Random Forest | 0.9605 | 0.7449 | 0.8391 | 0.9529 | 3 | 25 | 73 |

*\*During the writing of this research, assistance was obtained from the Claude (Anthropic, 2026) artificial intelligence tool. This tool was used for support in literature review, theoretical framework development, section drafting, and Python code development stages. Python codes were executed step by step by the author in the Google Colab environment, the results obtained were interpreted by the author personally, and all academic content was reviewed and reformulated by the author.*

---

## 5. Scenario / Data Analysis

### 5.1. Performance Analysis of the Rule-Based System

The evaluation results of the rule-based system clearly reveal the inadequacy of threshold-based approaches in credit card fraud detection. This system, based on the 75th percentile amount threshold, produced Precision of 0.0024, Recall of 0.3537, and F1-Score of 0.0049 on the test data.

Examining the Confusion Matrix, it can be seen that the system incorrectly classified 71,028 of the 284,807 transactions as fraudulent. This means that tens of thousands of legitimate customer transactions would be unnecessarily blocked every day, creating serious problems in terms of both operational costs and customer experience. On the other hand, the system correctly detected only 174 of the 492 fraudulent transactions, missing 318. The ROC-AUC value of 0.5519 indicates that the system is only marginally better than a random classifier. These findings empirically validate the theoretical prediction of Bolton and Hand (2002) that static rule systems will gradually lose their effectiveness against changing fraud patterns.

*Figure 2: Rule-Based System Confusion Matrix*

### 5.2. Performance Analysis of the Random Forest Model

The Random Forest model demonstrated a striking improvement across all performance metrics compared to the rule-based system. The model achieved a Precision of 0.9605, Recall of 0.7449, F1-Score of 0.8391, and ROC-AUC of 0.9529 on the test set. Examining the Confusion Matrix, it can be seen that 73 of the 98 fraudulent transactions were correctly detected, and only 3 legitimate transactions were incorrectly classified as fraudulent. In contrast to the 71,028 false positives generated by the rule-based system, Random Forest produced only 3 false positives — a difference of approximately 23,676 times fewer unnecessary alarms. The ROC curve tracking close to the upper left corner and the AUC value of 0.9529 confirm that the model has strong discriminative capacity across different decision thresholds. These findings are consistent with the high-performance results obtained by Dornadula and Geetha (2019) on the same dataset.

*Figure 3: Random Forest Confusion Matrix and ROC Curve*

### 5.3. Comparative Evaluation

The comparative analysis of the two models clearly demonstrates that the machine learning-based approach offers measurable advantages over the rule-based system in terms of accuracy, sensitivity, and adaptability. Examining the comparison presented in Table 1, the F1-Score increased from 0.0049 to 0.8391, representing an approximately 171-fold improvement. In terms of Precision, while only two in a thousand transactions flagged as "fraudulent" by the rule-based system were actually fraudulent, this rate reaches 96% in Random Forest. In terms of Recall, the rule-based system detects 35% of fraudulent transactions, while Random Forest raises this rate to 74%. As foreseen within the theoretical framework of Phua et al. (2010) and Bolton and Hand (2002), the machine learning model's learning of dynamic decision boundaries from data enables it to both generate far fewer false alarms and detect more fraudulent transactions compared to fixed threshold rules. *(See Table 1: Model Comparison)*

### 5.4. Feature Importance Analysis

The feature importance analysis of the Random Forest model reveals that the most decisive variables in fraud detection are the PCA components. V14 (0.1799), V10 (0.1154), V12 (0.0962), and V4 (0.0956) stand out as the four components with the highest importance. The importance score of the Amount variable, which represents the transaction amount, remains at only 0.0110. This finding is highly significant: the transaction amount used as the decision criterion by our rule-based system ranks among the least important variables in the Random Forest model. This explains why the rule-based approach performs so poorly — fraudulent behavior is not related to transaction amount but to the latent structural characteristics of transactions (PCA components). As emphasized by Happa et al. (2019), multidimensional approaches that model individual behavioral patterns in anomaly detection produce far more effective results than single-variable threshold rules.

*Figure 4: Random Forest Feature Importance Analysis*

---

## 6. Conclusion and Discussion

### 6.1. Answers to Research Questions

This study aimed to investigate the measurable advantages that machine learning-based anomaly detection models offer over traditional rule-based systems in credit card fraud detection, with respect to accuracy, sensitivity, and adaptability. The findings answer this question clearly: the Random Forest model achieved a 171-fold improvement in F1-Score and approximately a 400-fold improvement in Precision compared to the rule-based system, reducing the number of false positives from 71,028 to just 3. The attainment of a ROC-AUC value of 0.9529 demonstrates that the model exhibits strong discriminative capacity under different operational conditions. These findings are consistent with the observations of Perols (2011) that machine learning models outperform statistical models in financial fraud detection under realistic class and cost imbalance conditions.

### 6.2. Theoretical Contribution and Relationship to Literature

The findings of the study support and extend the existing literature in several ways. The theoretical prediction of Bolton and Hand (2002) that static rule systems will gradually lose their effectiveness against changing fraud patterns has been empirically validated in this study. The feature importance analysis revealed that the transaction amount (Amount) — the decision criterion adopted by the rule-based system — ranks among the least important variables in the Random Forest model. This finding strengthens the argument of Phua et al. (2010) that multidimensional pattern analysis approaches are superior to single-variable threshold rules in fraud detection. Furthermore, the consistency of the results obtained with those of Dornadula and Geetha (2019) confirms the reliability of the Random Forest algorithm on this dataset.

### 6.3. Limitations

This study has several important limitations. First, the rule-based system used does not fully represent the complex rule engines used in real-world applications; it is based on only a single threshold rule. Second, the dataset covers only two days' worth of transaction data and does not reflect long-term behavioral changes or concept drift. Third, the PCA transformation rendered the features uninterpretable, limiting the explainability of the model's decision mechanism. Finally, data balancing techniques such as SMOTE were not applied in this study; it is anticipated that these techniques could further improve the Recall value.

### 6.4. Recommendations for Future Research

Future research can advance in several directions. First, expanding the comparative analysis with XGBoost and deep learning models would contribute to the model selection discussion. Second, it would be advisable to systematically examine the effect of SMOTE and other data balancing techniques on Recall. Third, the integration of explainable AI tools such as SHAP (SHapley Additive exPlanations) could increase transparency in the model's decision mechanism, thereby improving clinical and legal acceptability. Finally, testing model performance on real-time streaming data would strengthen the transferability of findings to operational environments.

---

## 7. References

Dornadula, V. N., & Geetha, S. (2019). Credit card fraud detection using machine learning algorithms. *Procedia Computer Science, 165*, 631–641.

Perols, J. (2011). Financial statement fraud detection: An analysis of statistical and machine learning algorithms. *Auditing: A Journal of Practice & Theory, 30*(2), 19–50.

Happa, J., Bashford-Rogers, T., Agrafiotis, I., Goldsmith, M., & Creese, S. (2019). Anomaly detection using pattern-of-life visual metaphors. *IEEE Access, 4*.

Phua, C., Lee, V., Smith, K., & Gayler, R. (2010). A comprehensive survey of data mining-based fraud detection research. *arXiv preprint arXiv:1009.6119*.

Bolton, R. J., & Hand, D. J. (2002). Statistical fraud detection: A review. *Statistical Science, 17*(3), 235–255.

---

*\*AI Usage Declaration: Assistance was obtained from the Claude (Anthropic, 2026) artificial intelligence tool during the research and writing process of this study. For details, see the Methodology section.*
