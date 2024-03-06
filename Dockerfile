FROM frolvlad/alpine-miniconda3:python3.7
LABEL authors="Iheb"

WORKDIR /usr/app

COPY requirements.txt .
RUN pip install -r  requirements.txt

EXPOSE 80
COPY src/app.py .
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "80"]