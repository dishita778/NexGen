# NexGen Predictive Delivery Optimizer

Predict delivery delays before they happen and take action early.


## Summary 

This project is a data-driven logistics tool that predicts which deliveries are at risk of being delayed.
It uses machine learning and interactive visual dashboards to help logistics teams make faster, smarter decisions.

Built with Streamlit, it combines multiple CSV datasets (orders, delivery, routes, fleet, and feedback) into one clean analytics app.



## What actually it does

* Merges and cleans logistics data automatically.
* Predicts on-time vs delayed deliveries.
* Shows interactive charts on delivery trends and customer feedback.
* Highlights key factors causing delays (route, priority, carrier, etc.).
* Lets you download predictions and cleaned data.



## Tech Stack

Python, Streamlit, Pandas, Scikit-learn, Plotly



## How does this run

1. Clone this repo:

   Clone this on git-hub. 
   Eg. How I will clone in my repo
   git clone https://github.com/<dishita778>/nexgen-delivery-optimizer.git
   cd nexgen-delivery-optimizer
   
2. Install dependencies:


   pip install -r requirements.txt
   
3. Start the app:


   streamlit run app.py
   

Then open the link shown in your terminal — usually [http://localhost:8501](http://localhost:8501).



## Example Insights

* Top delay reasons by region
* Ratings vs delay correlation
* Feature importance for delay prediction
* Model performance metrics (accuracy, recall, F1)





