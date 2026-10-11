FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY contracts /app/contracts
COPY evaluation /app/evaluation
COPY generators /app/generators

CMD ["python", "-m", "evaluation.eval_runner", "--generator", "gaussian_copula", "--seed", "42"]
