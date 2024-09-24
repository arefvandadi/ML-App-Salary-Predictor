FROM python:3.12

WORKDIR /app

COPY ./data/stack-overflow-developer-survey-2020 ./data/stack-overflow-developer-survey-2020
COPY ./app.py .
COPY ./page_options ./page_options
COPY ./data/trained-ML-model/Full_ML_Model.pkl ./data/trained-ML-model/Full_ML_Model.pkl
COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS=false"]
