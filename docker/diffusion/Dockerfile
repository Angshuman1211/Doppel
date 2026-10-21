FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app

# Same dependency set as the CTGAN image: the diffusion model needs torch, and ctgan/rdt come with the file.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /app/generators/requirements-neural.txt

CMD ["python", "-m", "generators.generate", "--generator", "tabular_diffusion", "--seed", "42"]
