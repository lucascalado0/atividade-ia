import psycopg2
from datetime import datetime
from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
from collections import Counter

# Cria o app Flask
app = Flask(__name__)

# Carrega o modelo
modelo = joblib.load('modelo_rg.pkl')

def conectar_banco():
    return psycopg2.connect(
        host="localhost",
        database="predicoes_ml",
        user="postgres",
        password="123456"
    )

# Rota padrão
@app.route('/')
def home():
    return 'API do Modelo de Câncer - Online!'

@app.route('/home', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Captura os dados do formulário
            radius_mean = float(request.form.get("radius"))
            texture_mean = float(request.form.get("texture"))
            perimeter_mean = float(request.form.get("perimeter"))
            area_mean = float(request.form.get("area"))

            # Agrupa os dados em um array para predição
            features = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean]])
            
            # Realiza a predição
            previsao = modelo.predict(features)
            diagnostico = "Maligno" if previsao[0] == 1 else "Benigno"

            # Salva no banco
            conn = conectar_banco()
            cursor = conn.cursor()
            sql = """
                INSERT INTO predicoes (radius_mean, texture_mean, perimeter_mean, area_mean, diagnostico, data_hora)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            valores = (radius_mean, texture_mean, perimeter_mean, area_mean, diagnostico, datetime.now())
            cursor.execute(sql, valores)
            conn.commit()

            # Busca as predições após inserir
            cursor.execute("SELECT id, radius_mean, texture_mean, perimeter_mean, area_mean, diagnostico, data_hora FROM predicoes ORDER BY data_hora DESC")
            dados = cursor.fetchall()

            cursor.close()
            conn.close()

            return render_template("index.html", previsao=diagnostico, dados=dados)

        except Exception as e:
            return render_template("index.html", previsao=f"Erro: {e}")

    elif request.method == "GET" and request.args.get("ver_dados") == "1":
        try:
            conn = conectar_banco()
            cursor = conn.cursor()
            cursor.execute("SELECT id, radius_mean, texture_mean, perimeter_mean, area_mean, diagnostico, data_hora FROM predicoes ORDER BY data_hora DESC")
            dados = cursor.fetchall()
            cursor.close()
            conn.close()
            return render_template("index.html", previsao=None, dados=dados)
        except Exception as e:
            return render_template("index.html", previsao=f"Erro: {e}")

    return render_template("index.html")

@app.route('/dashboard')
def dashboard():
    try:
        conn = conectar_banco()
        cursor = conn.cursor()
        cursor.execute("SELECT radius_mean, texture_mean, perimeter_mean, area_mean, diagnostico, data_hora FROM predicoes ORDER BY data_hora DESC")
        registros = cursor.fetchall()
        cursor.close()
        conn.close()

        # Cria DataFrame para facilitar
        colunas = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'diagnostico', 'data_hora']
        df = pd.DataFrame(registros, columns=colunas)

        # Contagem das classes
        contagem_classes = dict(Counter(df['diagnostico']))

        # Estatísticas básicas
        estatisticas = df.describe().to_dict()

        return render_template("dashboard.html", contagem_classes=contagem_classes, estatisticas=estatisticas, registros=registros)
    
    except Exception as e:
        return f"Erro ao carregar dashboard: {e}"

# Rota para exibir as predições salvas
@app.route('/dados')
def dados():
    try:
        conn = conectar_banco()
        cursor = conn.cursor()
        cursor.execute("SELECT radius_mean, texture_mean, perimeter_mean, area_mean, diagnostico, data_hora FROM predicoes ORDER BY data_hora DESC")
        registros = cursor.fetchall()
        cursor.close()
        conn.close()
        return render_template("dados.html", registros=registros)
    except Exception as e:
        return f"Erro ao carregar dados: {e}"

# Rota para exportar as predições como CSV
from flask import Response
import csv

@app.route('/exportar')
def exportar():
    try:
        conn = conectar_banco()
        cursor = conn.cursor()
        cursor.execute("SELECT radius_mean, texture_mean, perimeter_mean, area_mean, diagnostico, data_hora FROM predicoes ORDER BY data_hora DESC")
        registros = cursor.fetchall()
        cursor.close()
        conn.close()

        def gerar_csv():
            yield 'radius_mean,texture_mean,perimeter_mean,area_mean,diagnostico,data_hora\n'
            for linha in registros:
                yield ','.join([str(valor) for valor in linha]) + '\n'

        return Response(gerar_csv(), mimetype='text/csv', headers={"Content-Disposition": "attachment;filename=predicoes.csv"})
    except Exception as e:
        return f"Erro ao exportar dados: {e}"


if __name__ == '__main__':
    app.run(debug=True)