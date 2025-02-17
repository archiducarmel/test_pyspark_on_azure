# startup.sh
#!/bin/bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$PATH:$JAVA_HOME/bin
gunicorn --bind=0.0.0.0:8000 app:app