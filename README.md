## KNN-ML-Motor

A KNN Machine Learning model setup to classify good and bad motor insurance business for new and existing clients based on a specific insurance market.


### The key specs of this model are the following:
1. It builds an ML KNN model and stores it in pickle format<br> 
2. It setups a RESTFul service using Flask to consume the built ML model by passing vehicle and other driver details parameters<br>


### Prerequisites are the following:
1. A CSV type report containing individual client policy details like insurance premium, driver nationaly, driver age, incurred claim cost, etc.<br>
2. Microsoft IIS server with version 7.x and above<br> 
3. Python 2.7.x or 3.7.x and above<br>