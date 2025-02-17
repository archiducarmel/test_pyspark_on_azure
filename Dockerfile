FROM python:3.11-slim

# Installation des dépendances système
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Configuration de Java de manière permanente
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/jvm/java-11-openjdk-amd64/bin"

# Répertoire de travail
WORKDIR /app

# Copie des fichiers spécifiques nécessaires
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY startup.sh startup.sh

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Rendre le script de démarrage exécutable
RUN chmod +x startup.sh

# Port d'exposition
EXPOSE 8000

# Démarrage de l'application
CMD ["./startup.sh"]