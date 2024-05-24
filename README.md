# churn_prediction

1. clone git repository (git clone https://github.com/sumanthegdegithub/churn_prediction.git)
2. create a folder data to keep data
3. create v-env with python -m venv venv
4. Activate venv with .\venv\Scripts\activate
5. Initialize dvc with dvc init
6. dvc add data/[data files]
7. dvc add  all *.dvc files
8. dvc remote add origin [https://dagshub.com/sumanthegdegithub/churn_prediction.dvc]
9. dvc remote modify origin --local auth basic   
10. dvc remote modify origin --local user sumanthegdegithub
11. dvc remote modify origin --local password [password]
12