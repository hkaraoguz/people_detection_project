FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN pip install pymongo opencv-python-headless numpy pydantic[dotenv] pytz

EXPOSE 8000

COPY ./app /app