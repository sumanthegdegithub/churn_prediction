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
12.
13.
Traceback (most recent call last):
  File "/app/app/main.py", line 12, in <module>
    from app.api import api_router
  File "/app/app/api.py", line 13, in <module>
    from churn_model import __version__ as model_version
  File "/usr/local/lib/python3.12/site-packages/churn_model/__init__.py", line 1, in <module>
    from churn_model.config.core import parent, config
ModuleNotFoundError: No module named 'churn_model.config'