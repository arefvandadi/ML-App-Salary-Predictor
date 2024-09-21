FROM python:3.12

WORKDIR /app

COPY ./data/stack-overflow-developer-survey-2020 ./data/stack-overflow-developer-survey-2020
COPY ./app.py .
COPY ./explore.py .
COPY ./predict.py .
COPY ./Full_ML_Model.pkl .

COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
