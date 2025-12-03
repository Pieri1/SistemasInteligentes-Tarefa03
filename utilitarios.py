# ===============================================================================

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg') # Configura backend não-interativo para evitar conflitos com Pygame/Tkinter
import matplotlib.pyplot as plt
import matplotlib.patches as mptch
import sklearn as sk
from collections import Counter
from constants import VS

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

# -------------------------------------------------------------------------------

# Plota gráfico simples usando contagens diretas por triagem
def plot_saved_vs_found_counts(total_counts, saved_counts, save_path=None):
    labels_map = {0: 'Verde (0)', 1: 'Amarelo (1)', 2: 'Vermelho (2)', 3: 'Preto (3)'}
    categories = [0, 1, 2, 3]

    found_vals = [int(total_counts.get(c, 0)) for c in categories]
    saved_vals = [int(saved_counts.get(c, 0)) for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, found_vals, width, label='Total (env)', color='gray', alpha=0.6)
    rects2 = ax.bar(x + width/2, saved_vals, width, label='Salvas', color='green', alpha=0.8)

    ax.set_ylabel('Quantidade')
    ax.set_title('Vítimas Totais vs Salvas por Triagem')
    ax.set_xticks(x)
    ax.set_xticklabels([labels_map.get(c, str(c)) for c in categories])
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()

# -------------------------------------------------------------------------------

def draw_rescuer_traces(env, resc_minds, save_path='rescuer_traces.png'):
    """Desenha o grid do ambiente e sobrepõe os trajetos dos socorristas.
    - env: Environment
    - resc_minds: lista de mentes dos socorristas (RescuerMind)
    """
    grid_w = env.dic.get('GRID_WIDTH', 94)
    grid_h = env.dic.get('GRID_HEIGHT', 94)

    # Render usando matplotlib
    fig, ax = plt.subplots(figsize=(10, 9))

    # Desenha obstáculos em escala de cinza
    for x in range(grid_w):
        for y in range(grid_h):
            obst = env.obst[x][y]
            if obst == VS.OBST_WALL:
                color = (0, 0, 0)
            elif obst == VS.OBST_NONE:
                color = (1, 1, 1)
            else:
                # escala de cinza baseada no fator do terreno
                perc = min(max(obst / 3.0, 0.0), 1.0)
                light = (1 - perc) * 1.0 + perc * 0.4
                color = (light, light, light)
            rect = plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor=(0.9, 0.9, 0.9), linewidth=0.3)
            ax.add_patch(rect)

    # Base
    bx, by = env.dic.get('BASE', [grid_w//2, grid_h//2])
    base_rect = plt.Rectangle((bx, by), 1, 1, fill=False, edgecolor=(0, 1, 1), linewidth=2)
    ax.add_patch(base_rect)

    # Paleta de cores fixa para os socorristas (em ordem):
    fixed_colors_rgb = [
        (0, 102, 204),   # azul
        (102, 204, 0),   # verde
        (204, 0, 51),    # vermelho escuro
    ]

    # Trajetos e vítimas salvas por agente
    for idx, mind in enumerate(resc_minds):
        # Seleciona cor fixa baseada no índice do socorrista; cicla se houver mais de 3
        rgb = fixed_colors_rgb[idx % len(fixed_colors_rgb)]
        color = (rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
        path = getattr(mind, 'trace_path', [])
        # desenha segmentos entre pontos consecutivos
        for i in range(1, len(path)):
            x0, y0 = path[i-1]
            x1, y1 = path[i]
            ax.plot([x0 + 0.5, x1 + 0.5], [y0 + 0.5, y1 + 0.5], color=color, linewidth=2)

        # marca vítimas salvas por este agente
        try:
            saved_by_agent = []
            for vid, savers in enumerate(env.saved):
                if any(getattr(phy.mind, 'NAME', None) == mind.NAME for phy in savers):
                    saved_by_agent.append(env.victims[vid])
            for (vx, vy) in saved_by_agent:
                ax.plot(vx + 0.5, vy + 0.5, marker='o', markersize=6, markerfacecolor=color, markeredgecolor='black')
        except Exception:
            pass

    ax.set_xlim(0, grid_w)
    ax.set_ylim(0, grid_h)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # para corresponder ao sistema pygame
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Mapa de trajetos salvo em: {save_path}")