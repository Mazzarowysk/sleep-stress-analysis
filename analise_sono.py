"""
SISTEMA COMPLETO DE ANÁLISE DE SAÚDE DO SONO
Integra pré-processamento, análise exploratória e modelagem preditiva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Configurações
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ============================================================================
# PARTE 1: CLASSE DE PRÉ-PROCESSAMENTO
# ============================================================================

class SleepDataPreprocessor:
    """Pré-processamento completo dos dados de sono e estilo de vida"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.label_encoders = {}
        self.scaler = None
        self.df_balanced = None

    def check_data_quality(self):
        print("=== ANÁLISE INICIAL DA BASE ===")
        print(f"Total de registros: {len(self.df)}")
        print(f"Total de colunas: {len(self.df.columns)}")
        print("\nValores nulos por coluna:")
        print(self.df.isnull().sum())
        print(f"\nValores duplicados: {self.df.duplicated().sum()}")
        print("\nDistribuição das profissões:")
        print(self.df['Occupation'].value_counts())
        print("\nDistribuição dos distúrbios do sono:")
        print(self.df['Sleep Disorder'].value_counts())

    def remove_duplicates(self):
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        final_count = len(self.df)
        print(f"Removidos {initial_count - final_count} registros duplicados")

    def handle_blood_pressure(self):
        # Colunas de BP já estão separadas no arquivo de entrada
        pass

    def handle_categorical_variables(self):
        categorical_columns = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
        for col in categorical_columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
        print("Variáveis categóricas codificadas:")
        for col in categorical_columns:
            le = self.label_encoders[col]
            print(f"{col}: {dict(zip(le.classes_, range(len(le.classes_))))}")

    def handle_outliers(self):
        numerical_columns = ['Age', 'Sleep Duration', 'Quality of Sleep',
                             'Physical Activity Level', 'Stress Level', 'Heart Rate',
                             'Daily Steps', 'BP_Systolic', 'BP_Diastolic']
        for col in numerical_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
            self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
        print("Outliers tratados com o método IQR")

    def create_new_features(self):
        self.df['Age_Group'] = pd.cut(self.df['Age'],
                                      bins=[20, 30, 40, 50, 60],
                                      labels=['20-29', '30-39', '40-49', '50-59'])
        self.df['Sleep_Health_Index'] = (
            self.df['Sleep Duration'] * 0.4 +
            self.df['Quality of Sleep'] * 0.4 +
            (10 - self.df['Stress Level']) * 0.2
        )
        self.df['Activity_Level'] = pd.cut(self.df['Physical Activity Level'],
                                          bins=[0, 30, 60, 100],
                                          labels=['Low', 'Medium', 'High'])
        le_age = LabelEncoder()
        le_activity = LabelEncoder()
        self.df['Age_Group'] = le_age.fit_transform(self.df['Age_Group'].astype(str))
        self.df['Activity_Level'] = le_activity.fit_transform(self.df['Activity_Level'].astype(str))
        self.label_encoders['Age_Group'] = le_age
        self.label_encoders['Activity_Level'] = le_activity
        print("Novas features criadas: 'Age_Group', 'Sleep_Health_Index', 'Activity_Level'")

    def balance_dataset(self, target_column='Occupation'):
        print("\n=== BALANCEAMENTO DO DATASET ===")
        original_dist = Counter(self.df[target_column])
        print("Distribuição original por profissão:")
        for occupation, count in original_dist.items():
            occupation_name = self.label_encoders['Occupation'].inverse_transform([occupation])[0]
            print(f"{occupation_name}: {count} registros")
        occupation_counts = self.df[target_column].value_counts()
        target_size = occupation_counts.median()
        balanced_dfs = []
        for occupation in self.df[target_column].unique():
            occupation_df = self.df[self.df[target_column] == occupation]
            current_size = len(occupation_df)
            occupation_name = self.label_encoders['Occupation'].inverse_transform([occupation])[0]
            if current_size < target_size:
                oversampled_df = resample(occupation_df, replace=True, n_samples=int(target_size), random_state=42)
                balanced_dfs.append(oversampled_df)
                print(f"{occupation_name}: Oversampling de {current_size} para {len(oversampled_df)}")
            elif current_size > target_size * 1.5:
                undersampled_df = resample(occupation_df, replace=False, n_samples=int(target_size), random_state=42)
                balanced_dfs.append(undersampled_df)
                print(f"{occupation_name}: Undersampling de {current_size} para {len(undersampled_df)}")
            else:
                balanced_dfs.append(occupation_df)
                print(f"{occupation_name}: Mantido com {current_size} registros")
        self.df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        final_dist = Counter(self.df_balanced[target_column])
        print("\nDistribuição balanceada por profissão:")
        for occupation, count in final_dist.items():
            occupation_name = self.label_encoders['Occupation'].inverse_transform([occupation])[0]
            print(f"{occupation_name}: {count} registros")

    def normalize_numerical_features(self):
        numerical_columns = ['Age', 'Sleep Duration', 'Quality of Sleep',
                             'Physical Activity Level', 'Stress Level', 'Heart Rate',
                             'Daily Steps', 'BP_Systolic', 'BP_Diastolic', 'Sleep_Health_Index']
        self.scaler = StandardScaler()
        self.df_balanced[numerical_columns] = self.scaler.fit_transform(self.df_balanced[numerical_columns])
        print("\nVariáveis numéricas normalizadas com StandardScaler")

    def get_preprocessed_data(self):
        return self.df_balanced

    def get_label_mappings(self):
        return self.label_encoders

    def run_full_preprocessing(self):
        print("INICIANDO PRÉ-PROCESSAMENTO COMPLETO")
        print("=" * 50)
        self.check_data_quality()
        self.remove_duplicates()
        self.handle_blood_pressure()
        self.handle_categorical_variables()
        self.handle_outliers()
        self.create_new_features()
        self.balance_dataset()
        self.normalize_numerical_features()
        print("\n" + "=" * 50)
        print("PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
        print(f"Dataset final: {len(self.df_balanced)} registros")
        print(f"Número de features: {len(self.df_balanced.columns)}")
        return self.df_balanced


# ============================================================================
# PARTE 2: CLASSE DE ANÁLISE PRINCIPAL
# ============================================================================

class BalancedSleepAnalysis:
    """Análise exploratória e modelagem dos dados de sono"""
    
    def __init__(self, processed_df, label_mappings):
        self.df = processed_df
        self.label_maps = {col: {int(k): v for k, v in d.items()} for col, d in label_mappings.items()}
        self.results = {}

    def analyze_occupation_sleep_quality(self):
        """Análise detalhada da relação entre profissão e qualidade do sono"""
        print("\n=== ANÁLISE PROFISSÃO vs QUALIDADE DO SONO (BALANCEADA) ===")
        occupation_stats = self.df.groupby('Occupation').agg({
            'Sleep Duration': ['mean', 'std'],
            'Quality of Sleep': ['mean', 'std'],
            'Sleep_Health_Index': ['mean', 'std'],
            'Stress Level': ['mean', 'std'],
            'Sleep Disorder': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        })
        occupation_stats.columns = ['_'.join(map(str, col)).strip() for col in occupation_stats.columns.values]
        occupation_stats = occupation_stats.round(3)
        occupation_names = [self.label_maps['Occupation'].get(int(occ)) for occ in occupation_stats.index]
        results = []
        for i, occ_code in enumerate(occupation_stats.index):
            occ_name = occupation_names[i]
            sleep_quality = occupation_stats.loc[occ_code, 'Quality of Sleep_mean']
            sleep_duration = occupation_stats.loc[occ_code, 'Sleep Duration_mean']
            stress_level = occupation_stats.loc[occ_code, 'Stress Level_mean']
            sleep_disorder_code = occupation_stats.loc[occ_code, 'Sleep Disorder_<lambda>']
            if pd.isna(sleep_disorder_code) or int(sleep_disorder_code) not in self.label_maps['Sleep Disorder']:
                sleep_disorder = 'Não Definido/NaN'
            else:
                sleep_disorder = self.label_maps['Sleep Disorder'].get(int(sleep_disorder_code))
            if sleep_disorder == 'nan':
                sleep_disorder = 'Sem Distúrbio (Maioria)'
            results.append({
                'Profissão': occ_name,
                'Qualidade_Sono_Media': sleep_quality,
                'Duração_Sono_Media': sleep_duration,
                'Nivel_Estresse_Medio': stress_level,
                'Disturbio_Sono_Mais_Comum': sleep_disorder
            })
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Qualidade_Sono_Media', ascending=False).reset_index(drop=True)
        print("\nRanking de Qualidade do Sono por Profissão:")
        print(f"{'Ranking de Qualidade do Sono (Dados Normalizados)':^80}")
        print("-" * 80)
        print(results_df.to_string(index=False))
        return results_df

    def plot_occupation_analysis(self, results_df):
        """Cria visualizações para a análise de profissões"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        sns.barplot(x='Qualidade_Sono_Media', y='Profissão', data=results_df, ax=axes[0,0], palette='viridis', hue='Profissão', legend=False)
        axes[0,0].set_title('Qualidade Média do Sono por Profissão (Normalizada) 😴')
        axes[0,0].set_xlabel('Qualidade do Sono (Normalizado)')
        axes[0,0].set_ylabel('Profissão')
        sns.barplot(x='Duração_Sono_Media', y='Profissão', data=results_df, ax=axes[0,1], palette='plasma', hue='Profissão', legend=False)
        axes[0,1].set_title('Duração Média do Sono por Profissão (Normalizada) 🌙')
        axes[0,1].set_xlabel('Duração do Sono (Normalizado)')
        axes[0,1].set_ylabel('')
        sns.barplot(x='Nivel_Estresse_Medio', y='Profissão', data=results_df, ax=axes[1,0], palette='magma', hue='Profissão', legend=False)
        axes[1,0].set_title('Nível Médio de Estresse por Profissão (Normalizado) 🥵')
        axes[1,0].set_xlabel('Nível de Estresse (Normalizado)')
        axes[1,0].set_ylabel('Profissão')
        disturbios = results_df['Disturbio_Sono_Mais_Comum'].value_counts()
        axes[1,1].pie(disturbios.values, labels=disturbios.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Pastel1'))
        axes[1,1].set_title('Distribuição dos Distúrbios do Sono Mais Comuns')
        plt.tight_layout()
        plt.savefig('occupation_sleep_analysis_balanced.png', dpi=300, bbox_inches='tight')
        plt.show()

    def build_prediction_model(self):
        """Constrói modelo preditivo para qualidade do sono"""
        print("\n=== MODELO PREDITIVO: Classificando Alta/Baixa Qualidade do Sono 🧠 ===")
        features = ['Age', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 
                    'Daily Steps', 'BP_Systolic', 'BP_Diastolic', 'Occupation']
        X = self.df[features]
        y = (self.df['Quality of Sleep'] > self.df['Quality of Sleep'].median()).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred, target_names=['Baixa Qualidade (0)', 'Alta Qualidade (1)']))
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nImportância das Features:")
        print(feature_importance.to_string(index=False))
        return model, feature_importance


# ============================================================================
# PARTE 3: ANÁLISES ESPECÍFICAS (CORRELAÇÃO, IMC, PATOLOGIAS)
# ============================================================================

def analyze_stress_sleep_correlation(original_data_path='Sleep_health_and_lifestyle_dataset_optimized.csv'):
    """Calcula e plota a correlação entre estresse e qualidade do sono"""
    print("\n=== ANÁLISE DE CORRELAÇÃO: Estresse vs. Qualidade do Sono ===")
    try:
        df_original = pd.read_csv(original_data_path)
        df_clean = df_original.drop_duplicates().copy()
        X, Y = 'Stress Level', 'Quality of Sleep'
        correlation_r = df_clean[X].corr(df_clean[Y])
        print(f"\nVariáveis Analisadas: {X} vs. {Y}")
        print(f"Coeficiente de Correlação de Pearson (r): {correlation_r:.4f}")
        if correlation_r < 0:
            direction = "negativa"
            strength = "muito forte" if abs(correlation_r) > 0.7 else "forte" if abs(correlation_r) > 0.5 else "moderada"
        elif correlation_r > 0:
            direction = "positiva"
            strength = "muito forte" if abs(correlation_r) > 0.7 else "forte" if abs(correlation_r) > 0.5 else "moderada"
        else:
            direction = "nula"
            strength = "inexistente"
        print(f"Interpretação: Correlação {direction} e {strength}.")
        plt.figure(figsize=(8, 6))
        sns.regplot(x=X, y=Y, data=df_clean, scatter_kws={'alpha': 0.5, 'color': 'skyblue'}, line_kws={'color': 'red'})
        plt.title(f'Correlação: Qualidade do Sono vs. Nível de Estresse (r = {correlation_r:.2f})', fontsize=14)
        plt.xlabel('Nível de Estresse (1 - 10)', fontsize=11)
        plt.ylabel('Qualidade do Sono (1 - 10)', fontsize=11)
        plt.text(df_clean[X].min() + 0.5, df_clean[Y].max() - 0.5, 
                 f'r = {correlation_r:.2f}', fontsize=12, color='red', weight='bold')
        plt.tight_layout()
        filename = 'correlacao_estresse_sono.png'
        plt.savefig(filename, dpi=300)
        print(f"\nGráfico salvo como: {filename}")
        plt.show()
    except FileNotFoundError:
        print(f"ERRO: Arquivo original '{original_data_path}' não encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro durante a análise: {e}")


def plot_bmi_distribution_top_n_chart(original_data_path='Sleep_health_and_lifestyle_dataset_optimized.csv', n=7):
    """Plota a distribuição de IMC nas top N profissões"""
    print(f"\n=== GERANDO GRÁFICO: Distribuição de IMC nas Top {n} Profissões ===")
    try:
        df_original = pd.read_csv(original_data_path)
        df_clean = df_original.drop_duplicates().copy()
        top_n_occupations = df_clean['Occupation'].value_counts().head(n).index.tolist()
        df_top_n = df_clean[df_clean['Occupation'].isin(top_n_occupations)].copy()
        bmi_occupation_counts = pd.crosstab(df_top_n['Occupation'], df_top_n['BMI Category'])
        bmi_occupation_percent = bmi_occupation_counts.div(bmi_occupation_counts.sum(axis=1), axis=0) * 100
        if 'Obese' in bmi_occupation_percent.columns:
            plot_data = bmi_occupation_percent.sort_values(by='Obese', ascending=True)
        else:
            plot_data = bmi_occupation_percent.sort_index(ascending=True)
        fig, ax = plt.subplots(figsize=(12, 7))
        plot_data[['Normal', 'Overweight', 'Obese']].plot(
            kind='barh', stacked=True, 
            color=['green', 'orange', 'red'],
            ax=ax
        )
        plt.title(f'Distribuição Percentual da Categoria de IMC nas Top {n} Profissões', fontsize=16)
        plt.xlabel('Porcentagem (%)', fontsize=12)
        plt.ylabel('Profissão', fontsize=12)
        plt.legend(title='Categoria de IMC', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        filename = f'distribuicao_imc_top{n}_registros.png'
        plt.savefig(filename, dpi=300)
        print(f"\nGráfico salvo como: {filename}")
        plt.show()
        print("\nTabela de Distribuição do IMC (Porcentagem por Profissão):")
        print(plot_data.to_string(float_format="%.2f"))
    except FileNotFoundError:
        print(f"ERRO: Arquivo original '{original_data_path}' não encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")


def plot_patology_risk_chart():
    """Plota o gráfico de risco de patologias do sono por profissão"""
    print("\n=== GRÁFICO: Risco de Patologias do Sono por Profissão ===")
    data = {
        'Profissão': ['Doctor', 'Engineer', 'Accountant', 'Nurse', 'Teacher', 'Lawyer', 'Software Engineer'],
        'N': [69, 65, 36, 29, 15, 12, 12],
        'Patologia_Percent': [5.80, 12.31, 8.33, 68.97, 73.33, 33.33, 33.33]
    }
    plot_data = pd.DataFrame(data)
    plot_data = plot_data.sort_values(by='Patologia_Percent', ascending=False).reset_index(drop=True)
    plt.figure(figsize=(10, 7))
    sns.barplot(x='Patologia_Percent', y='Profissão', data=plot_data, palette='coolwarm', hue='Profissão')
    plt.legend().remove()
    plt.title('Risco de patologias do sono nas sete profissões com maior número de registros', fontsize=16)
    plt.xlabel('Porcentagem de Indivíduos com Patologia (%)', fontsize=12)
    plt.ylabel('Profissão', fontsize=12)
    plt.xlim(0, plot_data['Patologia_Percent'].max() * 1.1)
    for index, row in plot_data.iterrows():
        label = f"{row['Patologia_Percent']:.2f}% (N={row['N']})"
        plt.text(row['Patologia_Percent'] + 0.5, index, label, va='center', ha='left', color='black', fontsize=10)
    plt.tight_layout()
    filename = 'patologias_top7_registros_ordenado.png'
    plt.savefig(filename, dpi=300)
    print(f"\nGráfico salvo como: {filename}")
    plt.show()


# ============================================================================
# PARTE 4: FUNÇÃO PRINCIPAL - EXECUÇÃO COMPLETA
# ============================================================================

def run_complete_analysis(input_file='Sleep_health_and_lifestyle_dataset_optimized.csv'):
    """
    Executa o pipeline completo:
    1. Pré-processamento dos dados
    2. Análise de profissões vs qualidade do sono
    3. Modelagem preditiva
    4. Análises específicas (correlação, IMC, patologias)
    """
    print("=" * 80)
    print("SISTEMA COMPLETO DE ANÁLISE DE SAÚDE DO SONO")
    print("=" * 80)
    
    # ------------------------------------------------------------------------
    # ETAPA 1: Carregamento e Pré-processamento
    # ------------------------------------------------------------------------
    print("\n[ETAPA 1] CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS")
    print("-" * 50)
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"\nERRO: O arquivo '{input_file}' não foi encontrado.")
        print("Certifique-se de que o arquivo está no mesmo diretório do script.")
        return
    
    # Pré-limpeza para o arquivo otimizado
    if 'Systolic_BP' in df.columns and 'Diastolic_BP' in df.columns:
        df = df.rename(columns={'Systolic_BP': 'BP_Systolic', 'Diastolic_BP': 'BP_Diastolic'})
    if 'BMI_Category_Code' in df.columns:
        df = df.drop('BMI_Category_Code', axis=1)
    if 'BP_Category' in df.columns:
        df = df.drop('BP_Category', axis=1)
    
    # Executar pré-processamento
    preprocessor = SleepDataPreprocessor(df)
    processed_df = preprocessor.run_full_preprocessing()
    
    # Salvar dataset processado
    processed_df.to_csv('sleep_health_processed_balanced.csv', index=False)
    
    # Salvar mapeamentos das labels
    label_mappings = {}
    for col, encoder in preprocessor.get_label_mappings().items():
        label_mappings[col] = {int(i): label for i, label in enumerate(encoder.classes_)}
    with open('label_mappings.json', 'w') as f:
        json.dump(label_mappings, f, indent=2)
    
    print("\nArquivos salvos:")
    print("- sleep_health_processed_balanced.csv")
    print("- label_mappings.json")
    
    # ------------------------------------------------------------------------
    # ETAPA 2: Análise Principal (Profissões vs Qualidade do Sono)
    # ------------------------------------------------------------------------
    print("\n[ETAPA 2] ANÁLISE DE PROFISSÕES VS QUALIDADE DO SONO")
    print("-" * 50)
    
    analyzer = BalancedSleepAnalysis(processed_df, label_mappings)
    occupation_results = analyzer.analyze_occupation_sleep_quality()
    analyzer.plot_occupation_analysis(occupation_results)
    
    # ------------------------------------------------------------------------
    # ETAPA 3: Modelo Preditivo
    # ------------------------------------------------------------------------
    print("\n[ETAPA 3] MODELO PREDITIVO DE QUALIDADE DO SONO")
    print("-" * 50)
    
    model, feature_importance = analyzer.build_prediction_model()
    
    # ------------------------------------------------------------------------
    # ETAPA 4: Análises Específicas
    # ------------------------------------------------------------------------
    print("\n[ETAPA 4] ANÁLISES ESPECÍFICAS")
    print("-" * 50)
    
    # 4.1 Correlação Estresse vs Sono
    analyze_stress_sleep_correlation(input_file)
    
    # 4.2 Distribuição de IMC por Profissão
    plot_bmi_distribution_top_n_chart(input_file, n=7)
    
    # 4.3 Risco de Patologias
    plot_patology_risk_chart()
    
    # ------------------------------------------------------------------------
    # RESUMO FINAL
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANÁLISE COMPLETA FINALIZADA COM SUCESSO!")
    print("=" * 80)
    print("\nArquivos gerados:")
    print("  1. sleep_health_processed_balanced.csv - Dados processados e balanceados")
    print("  2. label_mappings.json - Mapeamento das categorias")
    print("  3. occupation_sleep_analysis_balanced.png - Gráfico de análise por profissão")
    print("  4. correlacao_estresse_sono.png - Correlação estresse vs qualidade do sono")
    print("  5. distribuicao_imc_top7_registros.png - Distribuição de IMC por profissão")
    print("  6. patologias_top7_registros_ordenado.png - Risco de patologias do sono")
    print("\n" + "=" * 80)


# ============================================================================
# PONTO DE ENTRADA PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Executa a análise completa
    run_complete_analysis('Sleep_health_and_lifestyle_dataset_optimized.csv')