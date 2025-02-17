FROM python:3.11-slim

# Installation des dépendances système
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Configuration de Java
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Répertoire de travail
WORKDIR /app

# Copie des fichiers requis
COPY requirements.txt .
COPY app.py .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Port d'exposition
EXPOSE 8000

# Démarrage de l'application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]