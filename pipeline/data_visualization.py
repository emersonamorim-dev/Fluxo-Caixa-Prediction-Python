import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Certifique-se de ajustar o caminho conforme necessário
df = pd.read_csv('../data/processed/data.csv')

def plot_data(df):
    if 'Ano/Mês' not in df.columns:
        print("A coluna 'Ano/Mês' não existe no DataFrame.")
        return

    x = np.arange(len(df['Ano/Mês']))
    # Ajustei o tamanho para maior clareza
    plt.figure(figsize=(18, 10))  
    plt.plot(x, df['Total Entradas (R$)'], label='Total Entradas (R$)', marker='o')
    plt.plot(x, df['Total Saídas (R$)'], label='Total Saídas (R$)', marker='o')

    plt.xticks(x, df['Ano/Mês'], rotation=45)  

    # Anotar os valores numéricos
    for i in x:
        offset_y = 15 if i % 2 == 0 else -25
        offset_y_negative = -25 if i % 2 == 0 else 15

        plt.annotate(f"{df['Total Entradas (R$)'].iloc[i]:,.0f}", 
                     (i, df['Total Entradas (R$)'].iloc[i]), 
                     textcoords="offset points", 
                     xytext=(0, offset_y), 
                     ha='center', 
                     fontsize=9, 
                     backgroundcolor='white')
        
        plt.annotate(f"{df['Total Saídas (R$)'].iloc[i]:,.0f}", 
                     (i, df['Total Saídas (R$)'].iloc[i]), 
                     textcoords="offset points", 
                     xytext=(0, offset_y_negative), 
                     ha='center', 
                     fontsize=9, 
                     backgroundcolor='white')

        plt.plot([i, i], 
                 [df['Total Entradas (R$)'].iloc[i], df['Total Saídas (R$)'].iloc[i]], 
                 'k--', 
                 alpha=0.7)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_data(df)


