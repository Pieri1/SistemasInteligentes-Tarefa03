# ===============================================================================

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mptch
import sklearn as sk
from collections import Counter

# ===============================================================================

# Plota os datasets
def plot_graph(df, x, y):

    cores = {0: 'green', 1: 'yellow', 2: 'red', 3: 'black'}
    contagem = Counter(df['tri'])
    print("\nNúmero de vítimas por classificação de Triagem:")
    for k in sorted(contagem.keys()):
        print(f"  {k} ({cores[k]}): {contagem[k]}")
    print("")# Espaçamento

    plt.figure(figsize=(16, 10))
    cores_pontos = df['tri'].map(cores)

    plt.scatter(x, y, color=cores_pontos, edgecolors='black', alpha=0.7)
    plt.title('Distribuição percentual da probabilidade de sobrevivência')
    plt.ylabel('Probabilidade de Sobrevivência')
    plt.xlabel(x.name)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    legendas = [mptch.Patch(color=cores[k], label=f'{k}') for k in sorted(cores.keys())]
    plt.legend(handles=legendas, title='Classificação da Triagem')

    plt.show()

# -------------------------------------------------------------------------------

# Plota confusion matrix
def plot_cmatrix(y_test, y_pred_test, report):
    sk.metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, cmap='viridis')
    plt.title('Matriz de Confusão')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    if report:
        print(sk.metrics.classification_report(y_test, y_pred_test))

# -------------------------------------------------------------------------------

# Plota arvore de decisão
def plot_tree(modelo, x_train, y_train):
    feature_names = x_train.columns.tolist()
    class_names = [str(c) for c in sorted(y_train.unique())]

    plt.figure(figsize=(8, 6))
    sk.tree.plot_tree(modelo, feature_names=feature_names, filled=True, rounded=True, class_names=class_names, fontsize=8)
    plt.show()

# -------------------------------------------------------------------------------

# Plota desempenho das parametrizações RN
def plot_vies_var (num_params, train_scores, vld_scores):

    plt.figure(figsize=(10, 6))
    colors = [
        ["darkblue", "lightblue"],    # param 0
        ["purple", "pink"],           # param 1
        ["darkgreen", "lightgreen"],  # param 2
    ]

    for i in range(num_params):
        plt.plot(range(1, len(train_scores[i]) + 1), train_scores[i], label=f"{i} Train Neg MSE", marker='o',
                 color=colors[i][0])
        plt.axhline(train_scores[i].mean(), color=colors[i][0], linestyle='--',
                    label=f"{i} Train.mean: {train_scores[i].mean():.2f}")
        plt.plot(range(1, len(vld_scores[i]) + 1), vld_scores[i], label=f"{i} Valid Neg MSE", marker='o',
                 color=colors[i][1])
        plt.axhline(vld_scores[i].mean(), color=colors[i][1], linestyle='--',
                    label=f"{i} Valid.mean: {vld_scores[i].mean():.2f}")

    plt.xlabel("Fold")
    plt.ylabel("Neg MSE")
    plt.title("Training and Validation Scores (Bias and Variance)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(np.arange(1, 3 + 1, 1))
    plt.grid()
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------------

# Plota um grafo com pesos da biblioteca networkx
def plot_nxgraph (graph):
    pos = {(x, y): (y, -x) for x, y in graph.nodes()}

    plt.figure(figsize=(6, 6))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=1500,
        node_color="lightblue",
        font_weight="bold",
        arrows=True,
        arrowsize=25,
        connectionstyle="arc3,rad=0.2",  # curva as retas
        edgecolors="black"
    )

    for (u, v), data in nx.get_edge_attributes(graph, 'weight').items():

        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels={(u, v): data},
            label_pos=0.5,
            font_size=9,
            rotate=False
        )

    plt.title("Grafo Direcionado", fontsize=14)
    plt.axis("off")
    plt.show()