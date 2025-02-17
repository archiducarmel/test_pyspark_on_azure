from flask import Flask, jsonify
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, mean, median
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
import os
from azure.storage.blob import BlobServiceClient
from io import StringIO
import pandas as pd

app = Flask(__name__)

# Configuration Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
CONTAINER_NAME = os.getenv('AZURE_STORAGE_CONTAINER_NAME', 'root')
BLOB_NAME = os.getenv('AZURE_STORAGE_BLOB_NAME', 'application_train.csv')

# Convertisseur personnalisé pour les types numpy
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app.json_encoder = NumpyEncoder

# Initialisation de Spark pour Azure App Service
spark = SparkSession.builder \
    .appName("HomeCreditRiskAPI") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.hadoop.fs.azure.account.key.{storage_account}.blob.core.windows.net", os.getenv('AZURE_STORAGE_ACCESS_KEY')) \
    .master("local[*]") \
    .getOrCreate()

def get_data_from_blob():
    """Récupérer les données depuis Azure Blob Storage"""
    try:
        # Créer le client Blob service
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        
        # Obtenir le client conteneur
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Télécharger le blob
        blob_client = container_client.get_blob_client(BLOB_NAME)
        blob_data = blob_client.download_blob()
        
        # Lire le contenu
        content = blob_data.content_as_text()
        
        return content
        
    except Exception as e:
        print(f"Erreur lors de la lecture du blob: {e}")
        raise

def load_spark_data():
    """Charger et mettre en cache les données dans Spark"""
    if not hasattr(load_spark_data, "df"):
        # Lire les données depuis Azure Blob
        csv_content = get_data_from_blob()
        
        # Convertir en DataFrame Spark via Pandas
        pdf = pd.read_csv(StringIO(csv_content))
        load_spark_data.df = spark.createDataFrame(pdf)
        
        # Mettre en cache
        load_spark_data.df.cache()
        load_spark_data.df.count()
    
    return load_spark_data.df

@app.route('/')
def hello():
    return "Welcome on my API"
    
@app.route('/api/plot/target_distribution')
def plot_target_distribution():
    df_spark = load_spark_data()
    target_counts = df_spark.groupBy('TARGET') \
        .agg(count('*').alias('count')) \
        .toPandas()
    
    # Convertir les données en types Python natifs
    target_counts['TARGET'] = target_counts['TARGET'].astype(int)
    target_counts['count'] = target_counts['count'].astype(int)
    
    fig = px.bar(
        target_counts,
        x='TARGET',
        y='count',
        title='Distribution des défauts de paiement',
        labels={'TARGET': 'Défaut de paiement', 'count': 'Nombre de clients'},
        template='plotly_white'
    )
    
    fig.update_layout(
        showlegend=False,
        title_x=0.5,
        bargap=0.1
    )
    
    return jsonify(fig.to_json())

@app.route('/api/plot/income_distribution')
def plot_income_distribution():
    df_spark = load_spark_data()
    
    stats = df_spark.select(
        mean('AMT_INCOME_TOTAL').alias('mean'),
        median('AMT_INCOME_TOTAL').alias('median')
    ).collect()[0]
    
    income_data = df_spark.select('AMT_INCOME_TOTAL') \
        .toPandas()
    
    # Convertir en types Python natifs
    income_data['AMT_INCOME_TOTAL'] = income_data['AMT_INCOME_TOTAL'].astype(float)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=income_data['AMT_INCOME_TOTAL'].tolist(),
        nbinsx=50,
        name='Distribution'
    ))
    
    fig.update_layout(
        title={
            'text': 'Distribution des revenus',
            'x': 0.5
        },
        showlegend=False,
        xaxis_title='Revenu total',
        yaxis_title='Nombre de clients',
        template='plotly_white',
        bargap=0.1
    )
    
    mean_value = float(stats['mean'])
    median_value = float(stats['median'])
    
    fig.add_annotation(
        text=f"Moyenne: {mean_value:,.0f}\nMédiane: {median_value:,.0f}",
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        showarrow=False,
        align="right"
    )
    
    return jsonify(fig.to_json())

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
