import matplotlib.pyplot as plt
import seaborn as sns

def plot_features_importance(model, features):
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = sorted(range(len(importances)), key=lambda k: importances[k])
            plt.figure(figsize=(10, 6))
            plt.title("Importância dos Recursos")
            bars = plt.barh(range(len(indices)), importances[indices], color='b', align='center')
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            
            # Adicionando anotações
            for bar in bars:
                plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                         f'{bar.get_width():.3f}', 
                         va='center')
                
            plt.gca().invert_yaxis()
            plt.show()
        else:
            print("O modelo não possui atributo 'feature_importances_'")
    except Exception as e:
        print(f"Erro ao plotar a importância dos recursos: {e}")

def plot_predictions(y_true, y_pred):
    try:
        plt.figure(figsize=(10, 6))
        
         # Pontos vermelhos para as previsões
        plt.plot(y_true, y_pred, 'ro') 
        
        # Linha azul para os valores reais
        plt.plot(y_true, y_true, 'b-')  

        # Adicionando linhas de oscilação e anotações
        for i in range(len(y_true)):
            plt.plot([y_true.iloc[i], y_true.iloc[i]], [y_true.iloc[i], y_pred[i]], 'k--')
            
            # Adicionado um offset para o valor de y para melhor visualização
            plt.text(y_true.iloc[i], y_pred[i] + (max(y_true) - min(y_true)) * 0.02, 
                     f'{y_pred[i]:.0f}', fontsize=9, ha='center', va='bottom')

        plt.xlabel("Valores Reais")
        plt.ylabel("Previsões")
        plt.title("Valores Reais vs Previsões")
        plt.show()
    except Exception as e:
        print(f"Erro ao plotar previsões vs valores reais: {e}")
