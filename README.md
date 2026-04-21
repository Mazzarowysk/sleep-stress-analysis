# sleep-stress-analysis
Este repositório reúne o artigo científico e o código-fonte em Python utilizados na análise da relação entre estresse, qualidade do sono e prevalência de distúrbios em profissões de alta demanda psicossocial.


# 🧠 Sleep Stress Analysis

Este projeto tem como objetivo analisar a relação entre estresse ocupacional e qualidade do sono utilizando técnicas de ciência de dados e machine learning em Python.

---

## 🎯 Objetivo

Investigar como diferentes níveis de estresse influenciam a qualidade do sono e a prevalência de distúrbios em profissionais de alta demanda, com base em dados reais.

---

## 📊 Dataset

O estudo utiliza o **Sleep Health and Lifestyle Dataset**, contendo informações sobre:

* Idade
* Profissão
* Duração do sono
* Qualidade do sono
* Nível de estresse
* Frequência cardíaca
* Pressão arterial
* IMC (Índice de Massa Corporal)
* Atividade física

---

## 🧪 Metodologia

O projeto foi dividido em etapas:

### 🔹 1. Pré-processamento

* Remoção de duplicatas
* Tratamento de outliers (IQR)
* Codificação de variáveis categóricas
* Normalização de dados

### 🔹 2. Engenharia de Features

* Criação do índice de saúde do sono
* Agrupamento por faixa etária
* Classificação de nível de atividade

### 🔹 3. Análise Exploratória

* Comparação entre profissões
* Distribuição de distúrbios do sono
* Análise de IMC

### 🔹 4. Análise Estatística

* Correlação de Pearson entre estresse e sono

### 🔹 5. Modelagem Preditiva

* Classificação da qualidade do sono
* Uso de Random Forest

---

## 📈 Principais Resultados

* Correlação negativa forte entre estresse e qualidade do sono (**r ≈ -0.89**)
* Profissões como **professores e enfermeiros** apresentam maior incidência de distúrbios
* Relação entre sobrepeso/obesidade e problemas de sono
* Estresse identificado como principal fator associado à piora do sono

---

## 🛠️ Tecnologias Utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## ▶️ Como Executar o Projeto

### 1. Clone o repositório

```bash
git clone https://github.com/Mazzarowysk/sleep-stress-analysis.git
cd sleep-stress-analysis
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Execute o script

```bash
python src/analise_sono.py
```

---

## 📁 Estrutura do Projeto

```
sleep-stress-analysis/
├── data/            # Dataset utilizado
├── src/             # Código principal
├── article/         # Artigo científico
├── outputs/         # Gráficos gerados
├── README.md
└── requirements.txt
```

---

## 📄 Artigo Científico

O artigo completo está disponível na pasta:

📁 `article/`

---

## 👨‍💻 Autor

**Marcelinho Mazzarowysk**

Projeto desenvolvido como estudo em Ciência de Dados, integrando análise estatística, visualização e modelagem preditiva.

---

