# Phase 1: Business Understanding

## 1. Case Description

### Dataset

Customer Satisfaction (Airline), This dataset contains data about customers for an airline company. Customer satisfaction is an important metric for a company's success. It would be interesting to predict customer satisfaction and identify the underlying drivers for customer satisfaction from the company's success.

### Problem Description

You are the managers of a company. The company faces no problem, but is looking to use the available data to improve the business model. Your task is to find a way in which data mining in combination with a given dataset can be used to gain insight and improve the business model.

---

## 2. Our Interpretation of the Case, Before Looking at the Data

We work for an airline, and want to find a way to use the Customer Satisfaction dataset to gain insight and improve the business model of the airline.

An airline generates revenue primarily through ticket sales, and via ancillary services such as baggage fees, seat upgrades, on-flight services, and loyalty programs (e.g. Lufthansa's Miles & More). The airline's profitability depends on managing high fixed costs like fuel, aircraft purchase and maintenance, labor, and attracting and retaining customers, particularly business and loyal travellers, who provide stable, recurring income. To differentiate themselves in a highly competitive market, airlines usually position themselves either in the high-end sector or the budget sector, the long-distance or short-distance sector. Customer satisfaction is an important factor determining whether a customer might choose the airline again for their next flight. Understanding and optimizing passenger satisfaction is therefore core to improving the business model, as it influences loyalty, reputation and long-term profitability of the airline.

By applying data mining techniques to this dataset, we aim to extract actionable insights that can inform and improve how the airline designs and delivers its services.

---

## 3. Business Objectives

The primary objective of this project is to use data mining techniques to identify key drivers of customer satisfaction and develop predictive models that can inform strategic business decisions to improve the airline's business model. This encompasses understanding what factors most strongly influence passenger satisfaction, predicting customer satisfaction to enable proactive service improvements, and identifying which aspects of the business model can be optimized based on data-driven insights.

Secondary objectives include service quality optimization, where we aim to identify service areas requiring immediate improvement, prioritize resource allocation based on impact on satisfaction, and understand the relative importance of different service attributes. We also seek customer segmentation insights to understand differences between customer segments such as loyal versus disloyal customers, business versus personal travel, and class differences, which will help identify high-risk customer segments prone to dissatisfaction and develop targeted strategies for different customer groups. Additionally, we want to conduct operational impact analysis to quantify the impact of operational factors like delays and flight distance on satisfaction, understand how service quality interacts with operational constraints, and identify operational improvements that could enhance satisfaction. Finally, we aim to build predictive capability by developing models that can accurately predict customer satisfaction, enable proactive identification of at-risk customers, and support data-driven decision making.

---

## 4. Research Questions

The main research question guiding this project is: How can we use data mining to identify the key drivers of customer satisfaction and predict satisfaction levels to improve the airline's business model?

To address this main question, we will explore several specific research questions. First, we want to understand what factors most strongly predict passenger satisfaction, including which service attributes such as wifi, seat comfort, food, and others have the highest impact on satisfaction, whether there are interaction effects between different services, and what is the relative importance of service quality versus operational factors like delays, class, and travel type.

Second, we aim to determine whether we can accurately predict passenger satisfaction by assessing the predictive power of available features, identifying which machine learning algorithms perform best for this classification task, and determining what level of accuracy we can achieve in predicting satisfaction.

Third, we seek to identify which passenger segments are most at risk of dissatisfaction by examining demographic patterns such as age and gender associated with dissatisfaction, understanding how travel characteristics like loyal versus disloyal customers, business versus personal travel, and class affect satisfaction, and determining whether we can identify high-risk customer profiles.

Fourth, we want to understand the relationship between service quality and satisfaction by identifying which services need the most improvement based on current ratings, determining the priority order for service enhancements, and examining how different service categories correlate with overall satisfaction.

Fifth, we aim to understand how operational factors affect satisfaction by quantifying the impact of flight delays on satisfaction, determining whether flight distance influences satisfaction levels, and understanding how operational constraints interact with service quality perceptions.

Finally, we want to identify what aspects of the business model can benefit from machine learning by determining which service improvements would have the highest ROI, understanding how predictive models can support customer retention strategies, and identifying what insights can inform pricing, service offerings, or operational decisions.

---

## 5. Success Criteria

For technical success, we aim to achieve classification accuracy greater than 85% in predicting customer satisfaction, successfully identify the top 5 to 10 most important satisfaction drivers, achieve precision and recall greater than 80% for both satisfied and dissatisfied classes, and ensure the model performs well on the test set, indicating good generalization.

For business success, we aim to provide clear, prioritized recommendations for service improvement, identify at least 3 to 5 high-impact improvement opportunities, successfully identify distinct customer segments with different satisfaction drivers, ensure insights directly relate to improving revenue, retention, or operational efficiency, and ensure recommendations are feasible and can be implemented by the airline.

---

## 6. Stakeholders

The primary stakeholders for this project include airline management, who are strategic decision-makers that will use insights for business model improvements, the customer service department, which is the operational team responsible for service delivery and customer interactions, the operations team, which manages flight operations, delays, and service logistics, and the marketing department, which uses insights for customer segmentation and targeted campaigns.

Secondary stakeholders include passengers, who are indirect beneficiaries of improved service quality, investors, who are interested in business model improvements that enhance profitability, and the data science team, which is responsible for implementing and maintaining predictive models.

---

## 7. Constraints and Assumptions

The project faces several constraints. The dataset contains historical information that may not reflect current market conditions or service offerings. Our analysis is constrained to available features in the dataset, with no external data sources such as weather, competitor data, or economic indicators. Some missing values exist in Arrival Delay in Minutes, with 310 missing values in the training set that require handling. There is no time-series information to track satisfaction trends over time. The scope focuses on passenger satisfaction only and does not include cost data, revenue impact, or competitive positioning.

We make several assumptions for this project. We assume that satisfaction ratings and customer responses are reliable indicators of actual satisfaction. We assume that historical patterns and relationships will continue to be relevant. We assume that identified service improvements are technically and financially feasible. We assume that strong correlations between features and satisfaction indicate causal relationships, with appropriate caveats. Finally, we assume that the training and test datasets are representative of the airline's customer base.

---

## 8. Project Scope

The project scope includes comprehensive exploratory data analysis with visualization of the dataset, development of classification models to predict customer satisfaction, identification and ranking of key satisfaction drivers, analysis of satisfaction patterns across different customer segments, evaluation of different service attributes and their impact, understanding the role of delays, flight distance, and other operational factors, and providing actionable insights for improving the business model.

The project scope explicitly excludes development of production systems for real-time prediction, detailed financial analysis of recommended improvements, comparison with competitor airlines or industry benchmarks, analysis of weather, economic conditions, or other external variables, time-series analysis of satisfaction trends, and establishing definitive causal relationships, focusing instead on correlation and prediction.

---

## 9. Business Understanding of the Data

The dataset contains information that captures the complete customer journey from booking to arrival, allowing us to understand how different aspects of the airline experience contribute to overall satisfaction. The variables in the dataset can be understood from a business perspective as representing three key dimensions of the airline's operations: customer characteristics, service quality touchpoints, and operational performance.

Customer characteristic variables include demographic information such as Gender and Age, which help the airline understand its customer base and identify potential demographic patterns in satisfaction. The Customer Type variable distinguishes between Loyal and Disloyal customers, which is crucial for the business as loyal customers represent recurring revenue and are typically more valuable to the airline. The Type of Travel variable (Business versus Personal Travel) is important because business travelers often have different expectations, higher tolerance for certain inconveniences, and represent a more stable revenue stream. The Class variable (Business, Eco, Eco Plus) directly relates to pricing strategy and service tier offerings, where higher classes typically command premium prices and are expected to deliver superior experiences.

Service quality variables represent the various touchpoints where customers interact with the airline throughout their journey. These include pre-flight services such as Ease of Online booking and Online boarding, which relate to the digital experience and operational efficiency. Airport services include Gate location, Checkin service, and Baggage handling, which represent the ground operations that customers experience before and after the flight. In-flight services encompass Food and drink, Seat comfort, Leg room service, Inflight entertainment, Inflight service, On-board service, and Cleanliness, which directly relate to the core product experience during the flight. Connectivity services include Inflight wifi service and Departure/Arrival time convenient, which relate to modern customer expectations for connectivity and schedule flexibility.

From a business perspective, these service quality variables represent areas where the airline can invest resources to improve customer experience. Understanding which of these services have the strongest relationship with overall satisfaction helps the airline prioritize investments and allocate resources effectively. For example, if seat comfort has a stronger relationship with satisfaction than gate location, the airline might prioritize investments in aircraft seating over terminal improvements.

Operational variables include Flight Distance, Departure Delay in Minutes, and Arrival Delay in Minutes. Flight Distance relates to the route network and operational complexity, where longer flights may require different service standards and customer expectations. Delays represent operational performance and directly impact customer experience, as delays are one of the most common sources of customer complaints in the airline industry. Understanding how delays affect satisfaction, and how service quality can mitigate the negative impact of delays, is crucial for operational decision-making.

The target variable, Satisfaction, represents the overall customer sentiment and is the ultimate measure of whether the airline has successfully delivered a positive experience. This variable is critical for the business because satisfied customers are more likely to return, recommend the airline to others, and provide stable revenue through repeat business. Dissatisfied customers, on the other hand, may switch to competitors, leave negative reviews, and reduce the airline's market share and profitability.

---

## 10. Target Variable Selection

The satisfaction variable is selected as the target variable for this data mining project because it directly aligns with the business objective of improving the airline's business model through understanding and optimizing customer experience. Satisfaction serves as a comprehensive outcome measure that captures the overall customer experience across all touchpoints, from booking to arrival.

From a business perspective, satisfaction is the most relevant target variable because it directly relates to customer retention, loyalty, and long-term profitability. Satisfied customers are more likely to become repeat customers, which is crucial for the airline's revenue stability. They are also more likely to recommend the airline to others, contributing to organic growth through word-of-mouth marketing. Additionally, satisfied customers are less likely to switch to competitors, reducing customer churn and the associated costs of acquiring new customers to replace lost ones.

The satisfaction variable is also appropriate because it is actionable. Unlike other potential target variables such as revenue or profit, which may be influenced by external factors beyond the airline's control, satisfaction is directly influenced by the services and experiences that the airline provides. This means that insights derived from predicting satisfaction can be translated into concrete actions to improve service delivery, operational performance, and resource allocation.

Alternative target variables were considered but deemed less suitable for this project. Customer retention or churn prediction could be valuable, but these would require longitudinal data tracking customers over multiple flights, which is not available in this dataset. Revenue per customer could be interesting, but it is not included in the dataset and would require additional data sources. Service-specific ratings could be used as targets, but these represent intermediate outcomes rather than the ultimate business objective of overall customer satisfaction.

The satisfaction variable is structured as a binary classification problem (satisfied versus neutral or dissatisfied), which is appropriate for several reasons. First, it simplifies the prediction task while still capturing the essential distinction between positive and negative experiences. Second, it aligns with business decision-making, where the airline needs to identify customers at risk of dissatisfaction to intervene proactively. Third, binary classification models are well-established and interpretable, allowing for clear communication of results to business stakeholders.

The slight class imbalance in the satisfaction variable (56.7% neutral or dissatisfied versus 43.3% satisfied) reflects the reality that dissatisfaction is somewhat more common than satisfaction in the airline industry, which is consistent with industry trends where customer complaints often outnumber positive feedback. This imbalance is not severe enough to require special handling techniques, but it will be considered during model training to ensure that predictions are not biased toward the majority class.

---

## 11. Initial Data Understanding (Preliminary)

### 2.1 Initial Exploration

Using train_df.info() and train_df.describe(), we can see that the train dataset has 103,904 rows and 24 columns. There is one variable (Arrival Delay in Minutes) of type float, 18 variables of type integer, and 5 categorical variables. We chose to display the first 10 rows of the dataset to get a closer look at the data. Every row is one customer, uniquely identified by an id.

### 2.2 Check for Missing Values

When checking for missing values in the train dataset, we found that the Arrival Delay in Minutes variable contains 310 missing values. This should be kept in mind, as imputation may be required later before building the model.

### 2.3 Classic Statistics

The mean age of customers in the dataset is 39 years, and the mean flight distance is 1,189.45 km. The minimum and maximum values also appear reasonable: age ranges from 7 to 85 years, and flight distance ranges from 31 km to 4,983 km. While 31 km seems extremely short, it could be a short island-to-island flight operated by the airline.

Regarding the maximum values for Departure Delay in Minutes (1,592) and Arrival Delay in Minutes (1,584), these delays appear unusually large but could represent flights that were rescheduled to the next day, for example due to weather conditions.

When examining the summary statistics for the categorical variables, we can see that the training dataset has an approximately 50:50 split between genders, while it heavily overrepresents customers classified as Loyal Customer and those traveling for Business Travel. For the Class variable, there is a relatively even distribution between Business and Eco travelers, with 49,665 and 46,745 observations respectively.

### 2.4 Satisfaction (Target Variable) Analysis

The satisfaction variable, which serves as our target variable for the prediction model, shows a slightly imbalanced distribution. About 56.7% of customers are classified as neutral or dissatisfied, while 43.3% are satisfied. This indicates that dissatisfaction or neutrality is somewhat more common among customers. The imbalance is not severe, but it should still be considered during model training to ensure that predictions are not biased toward the majority class. This distribution reflects the reality of the airline industry where customer complaints and neutral experiences often outnumber highly satisfied experiences, making it an important business challenge to address.

### 2.6 Average Satisfaction Scores per Category

The average satisfaction scores across the different satisfaction score columns range from 2.73 to 3.64 on a five-point scale. The highest-rated features are Inflight service (3.64) and Baggage handling (3.63), while the lowest-rated ones are Inflight WiFi service (2.73) and Ease of Online booking (2.76). This suggests that while operational services such as in-flight and baggage handling are generally well received, digital touchpoints and connectivity services were not well rated by customers.

### 2.7 Demographics

The demographic analysis reveals that the mean age of customers is 39 years, with a range from 7 to 85 years. The gender distribution shows an approximately 50:50 split between males and females. The dataset shows a heavy overrepresentation of Loyal Customers compared to Disloyal Customers, and similarly, Business Travel is heavily overrepresented compared to Personal Travel.

### 2.8 Travel Characteristics

The analysis of travel characteristics shows that there is a relatively even distribution between Business class (49,665 observations) and Eco class (46,745 observations), with fewer observations in Eco Plus class (7,494). The customer type distribution heavily favors Loyal Customers over Disloyal Customers, and Business Travel is heavily overrepresented compared to Personal Travel. These distributions suggest that the airline's customer base consists primarily of loyal, business travelers, which aligns with the business context where these customers provide stable, recurring income.

### 2.9 Flight Characteristics

The Flight Distance Distribution graph is skewed towards shorter distance flights, indicating that there are a lot of customers traveling short distance. The mean flight distance is 1,189.45 km, with a range from 31 km to 4,983 km. The delay analysis shows that the mean departure delay is 14.82 minutes with a median of 0 minutes, and the mean arrival delay is 15.18 minutes with a median of 0 minutes. The maximum delays of 1,592 minutes for departure and 1,584 minutes for arrival are unusually large but could represent flights that were rescheduled to the next day due to weather conditions or other operational issues.

---

## 12. Expected Deliverables

The project will deliver several key outputs across the CRISP-DM phases. Phase 1, Business Understanding, will provide complete business context and objectives, research questions and success criteria, and an initial data understanding summary. Phase 2, Data Understanding, will deliver a comprehensive exploratory data analysis with detailed visualizations and statistical summaries, data quality assessment, and key patterns and initial insights.

Phase 3, Data Preparation, will document data cleaning procedures, missing value handling strategy, feature engineering approaches, and train/test split methodology. Phase 4, Modeling, will provide model selection rationale, algorithm comparison including Logistic Regression, Random Forest, XGBoost, and SVM, hyperparameter tuning results, and model training documentation.

Phase 5, Evaluation, will deliver model performance metrics including accuracy, precision, recall, and F1-score, confusion matrices and ROC curves, feature importance analysis, model validation on test set, and business impact assessment. Finally, Phase 6 will provide a complete CRISP-DM documentation with executive summary, methodology overview, key findings and insights, actionable business recommendations, and limitations and future work.

---

## 13. Timeline

The project follows a seven-week timeline aligned with the CRISP-DM methodology. Week 2 focuses on Business Understanding, which is now complete with this report. Week 3 is dedicated to Data Understanding, which is currently in progress and will deliver an EDA report with visualizations. Weeks 3-4 cover Data Preparation, delivering a cleaned dataset and feature engineering. Weeks 4-5 focus on Modeling, delivering trained models and comparison. Week 6 covers Evaluation, delivering performance metrics and validation. Week 7 concludes with the Final Report, delivering the complete CRISP-DM report.

---

## 14. Next Steps

The immediate actions for Week 3 include completing the Data Understanding phase with a deep dive into customer segments including loyal versus disloyal customers, business versus personal travel, and class differences, conducting statistical analysis of satisfaction drivers, performing gap analysis between satisfied and dissatisfied customers, and creating comprehensive visualizations.

We also need to prepare for the teacher meeting by presenting research objectives and approach, discussing modeling strategy, confirming evaluation metrics focus, and addressing any questions or concerns. Additionally, we need to plan for Data Preparation by developing a strategy for handling missing values, planning feature engineering approaches, and designing train/test validation strategy.

---

## 15. Notes for Teacher Meeting (Week 3)

The key points to present during the teacher meeting include our research focus on using data mining to identify satisfaction drivers and predict satisfaction to improve the business model, our approach using classification models to predict satisfaction with emphasis on feature importance and customer segmentation, and our expected outcomes of actionable insights for service improvement and customer retention strategies.

Questions to discuss include preference for modeling approach, specifically whether to focus on classification or include clustering and association rules, evaluation metrics emphasis including accuracy, precision/recall balance, and business impact, any specific requirements or constraints for the project, and feedback on research questions and objectives.

---

**Document Version**: 1.0  
**Last Updated**: Week 2  
