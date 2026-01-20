FROM python:3.12-slim AS base

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY . .


RUN pip install --no-cache-dir -r requirements.txt


RUN pip install --no-cache-dir -e .


ENTRYPOINT ["python", "src/vdm_pokemon/train.py"]

