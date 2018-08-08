# Dockerfile to create env: build using `datmo env build`
# [1] Base environment to start from:
# Find more at https://hub.docker.com/u/datmo/
FROM datmo/xgboost:cpu

RUN mkdir /templates
RUN mkdir /static
RUN mkdir /code

COPY ./app.py ./model.py ./xgb_model.pkl ./testinput.json /code/
COPY templates/index.html /templates/
COPY static /static/

WORKDIR /code

EXPOSE 5000

CMD python /code/app.py

