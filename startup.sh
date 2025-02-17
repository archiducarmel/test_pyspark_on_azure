#!/bin/bash

# Configuration supplémentaire de l'environnement si nécessaire
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/jvm/java-11-openjdk-amd64/bin"

# Démarrage de l'application
gunicorn --bind=0.0.0.0:8000 app:app