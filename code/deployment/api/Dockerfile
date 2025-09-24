FROM python:3.11-slim


WORKDIR /app


COPY deployment/api/main.py /app/

COPY models /models


RUN apt-get update \
	&& apt-get install -y --no-install-recommends build-essential gcc \
	&& rm -rf /var/lib/apt/lists/*


RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir fastapi uvicorn[standard] scikit-learn numpy pydantic


EXPOSE 8000


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]