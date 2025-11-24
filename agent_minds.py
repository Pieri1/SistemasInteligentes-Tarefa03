"""Made with 22/10/2025 version of env files on teacher's github"""
# ===============================================================================

import os
import csv
import numpy as np
import pandas as pd
import sklearn as sk
import random
import math
# import utilitarios as util
from constants import VS
from abstract_agent import AbstAgent
import networkx as nx


# ===============================================================================

# Class ExplorerMind
"""Implements the default behaviour for the explorer agents"""

class ExplorerMind(AbstAgent):
    def __init__(self, env, config_file, walk_sequence=None):
        super().__init__(env, config_file)
        self.highest_cost_action = max(self.COST_LINE * 3.0, self.COST_DIAG * 3.0, self.COST_READ, self.COST_FIRST_AID)
        self.walk_sequence = self.create_walk_sequence if not walk_sequence else walk_sequence
        self.already_read_vitals = 0 # marks if it's last action was reading vital signals
        self.victims = [] # victims found by this agent
        self.mental_map = nx.DiGraph() # the map resulted of this agent's exploration
        self.cheapest_path_to_base = [] # current cheapest path to base according to the mental map
        self.cheapest_cost_to_base = 0.0 # the cost of the cheapest path to base

        # Starts the stack with the base
        self.backtracking_stack = [{"Cost": 0,
                                    "Direction": (0, 0),
                                    "Coord": (env.dic["BASE"][0], env.dic["BASE"][1]),
                                    "Untried_Nbrs": self.create_untried_nbrs()}]

        # Marks the base as already visited
        self.get_env().visited[env.dic["BASE"][0]][env.dic["BASE"][1]].append(self._AbstAgent__phy)

        self.update_mental_map(self.backtracking_stack[0])
        self.update_cheapest_path_to_base()
        #util.plot_nxgraph(self.mental_map)

        print(f'{self.NAME} walk sequence: {walk_sequence}')

    def deliberate(self):

        # Starts going back to base if the cost to backtrack gets high enough
        if  (self.cheapest_cost_to_base + self.highest_cost_action) > self.get_rtime():
            return self.go_back_to_base()
        # Else, acts normally
        else:
            current_node = self.backtracking_stack[-1]
            x, y = current_node["Coord"]

            if (self.check_for_victim() != VS.NO_VICTIM) and not self.already_read_vitals:
                vital_signals = self.read_vital_signals()
                vital_signals[0] = (x,y)
                self.victims.append(vital_signals)
                print (f'Victim Found: {vital_signals}')
                self.already_read_vitals = 1
                return True

            else:
                while current_node["Untried_Nbrs"]:
                    direction = current_node["Untried_Nbrs"].pop(0)
                    dx, dy = self.AC_INCR[direction]
                    new_x, new_y = x + dx, y + dy

                    if self._AbstAgent__phy not in self.get_env().visited[new_x][new_y]:
                        result = self.walk(dx,dy)
                        if result == VS.EXECUTED:
                            self.already_read_vitals = 0
                            self.update_backtracking_stack(dx, dy, new_x, new_y)
                            self.update_mental_map(self.backtracking_stack[-1])
                            self.update_cheapest_path_to_base()
                            return True
                        elif result == VS.TIME_EXCEEDED:
                            return False

                result = self.backtrack()
                return result

#-------------------------------------------------------------------------------

    """Goes back one space in the backtracking stack"""

    def backtrack (self):

        if len(self.backtracking_stack) > 1:
            node = self.backtracking_stack.pop()
            dx, dy = node["Direction"]
            dx, dy = dx * -1, dy * -1  # gets the inverse direction
            #print(f'Going Back || cost: {node["Cost"]}, dir: {dx, dy}, coord: ({node["Coord"]})')

            result = self.walk(dx, dy)
            return True if result != VS.TIME_EXCEEDED else False
        else:
            return False

# -------------------------------------------------------------------------------

    """Updates the backtracking stack based on the direction, coordinates and cost
     of the last walk operation that was not a backtrack"""

    def update_backtracking_stack(self, dx, dy, new_x, new_y):

        if dx != 0 and dy != 0:  # diagonal
            base = self.COST_DIAG
        else:  # walk vertical or horizontal
            base = self.COST_LINE

        cost = base * self.get_env().obst[new_x][new_y]

        node = {"Cost": cost, "Direction": (dx, dy), "Coord": (new_x, new_y),"Untried_Nbrs": self.create_untried_nbrs()}
        #print(node)
        #print(self.backtracking_stack[-1])
        self.backtracking_stack.append(node)


#-------------------------------------------------------------------------------

    """Creates a shifted list of number ranging from 0 to 7 equal to the agents' 
    action sequence, removes the obstacles from the list, then returns the list"""

    def create_untried_nbrs(self):

        untried_nbrs = self.walk_sequence
        obstacles = self.check_walls_and_lim()

        untried_nbrs = [d for d in untried_nbrs if obstacles[d] == 0]
        #print(untried_nbrs)
        return untried_nbrs

#-------------------------------------------------------------------------------

    """Creates a random walk sequence automatically based on a random shift of 
    AC_INCR's number sequence"""

    def create_walk_sequence(self):
        walk_sequence = []

        shift = random.randint(0, 14)

        for i in range(8):
            walk_sequence.append(shift % 8)
            shift += 1

        return walk_sequence

#-------------------------------------------------------------------------------

    """Updates the mental map with hopefully the last node added in the backtracking stack
    automatically connecting to it all it's possible neighbours, and travel costs"""

    def update_mental_map (self, node):
        x, y = node["Coord"]
        possible_nbrs = node["Untried_Nbrs"]

        for direction in possible_nbrs:
            dx, dy = self.AC_INCR[direction]

            if dx != 0 and dy != 0:  # diagonal
                base = self.COST_DIAG
            else:  # walk vertical or horizontal
                base = self.COST_LINE

            nbr_x, nbr_y = x + dx, y + dy

            self.mental_map.add_node((x, y))
            #adds two edges with different costs, one to go, and one to come back
            cost = base * self.get_env().obst[nbr_x][nbr_y]
            self.mental_map.add_edge((x,y),(nbr_x,nbr_y), weight=cost, direction=(dx,dy))

            cost = base * self.get_env().obst[x][y]
            self.mental_map.add_edge((nbr_x, nbr_y), (x, y), weight=cost, direction=(-dx,-dy))

# -------------------------------------------------------------------------------

    def update_cheapest_path_to_base (self):
        x_base, y_base = self.backtracking_stack[0]["Coord"]
        current_x, current_y = self.backtracking_stack[-1]["Coord"]

        def heuristic(a, b): # Euclidean distance
            (x1, y1), (x2, y2) = a, b
            return math.hypot(x2 - x1, y2 - y1)

        path = nx.astar_path(self.mental_map, (current_x,current_y), (x_base,y_base), heuristic=heuristic)
        cost = nx.path_weight(self.mental_map, path, weight='weight')
        #print(f'Shortest Path to base: {path}')
        #print(f'Cost: {cost} Rtime: {self.get_rtime()}')
        self.cheapest_path_to_base = path
        self.cheapest_cost_to_base = cost


# -------------------------------------------------------------------------------

    """Method used to travel the current cheapest path to base, one at a time
    until the agent reaches the base (size equal to 1)"""

    def go_back_to_base(self):
        if len(self.cheapest_path_to_base) > 1: #base is the last item of the list
            current_x, current_y = self.cheapest_path_to_base.pop(0)
            target_x, target_y = self.cheapest_path_to_base[0]
            dx,dy = self.mental_map[(current_x, current_y)][(target_x, target_y)]['direction']
            result = self.walk(dx,dy)
            return True if result != VS.TIME_EXCEEDED else False
        else:
            return False

# -------------------------------------------------------------------------------

    def get_mental_map(self):
        return self.mental_map

# ===============================================================================

# Class RescuerMind
"""Implements the default behaviour for the rescuer agents, its 
    initialization works different if they are the Master agent 
    or not, if they aren't it means that the prediction model, 
    predicted tri and cluster labels, need to be passed by the 
    current Master Agent"""

class RescuerMind(AbstAgent):
    def __init__(self, env, config_file, data_folder, exploration_map, pred_model, tri_pred, sobr_pred, cluster_labels, Master_Agent=False):
        super().__init__(env, config_file)
        self.exploration_map = exploration_map
        self.data_folder = data_folder
        self.training_data_set_path = os.path.join(self.data_folder, "training_data.csv")
        self.found_data_set_path = os.path.join(self.data_folder, "data_found.csv")
        self.env_victims_found_path = os.path.join(self.data_folder, "env_victims_found.txt")
        self.victims_found = [] #stores the coordinates of all found victims in exploration phase
        self.assigned_victims = [] #stores the coordinates of victims that the agent was assigned to rescue

        #hard-coded solution to assign a cluster to the agent based on their self.NAME
        self.assigned_cluster = (int(''.join([c for c in self.NAME if c.isdigit()])) - 1) # returns the number in their name -1

        #fills victims_found
        with open(self.env_victims_found_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                x = int(row[0])
                y = int(row[1])
                self.victims_found.append((x, y))

        #trains a new model if they are the master agent
        if Master_Agent:
            self.prediction_model = self.learn_model(self.training_data_set_path)
            self.tri_predicted = self.predict_data_set(self.found_data_set_path, self.prediction_model)
            self.prediction_model_sobr = self.learn_model_sobr(self.training_data_set_path)
            self.sobr_predicted = self.predict_sobr(self.found_data_set_path, self.prediction_model_sobr)
            self.cluster_labels = self.clusterize(3) #creates a number of clusters equal to the number of rescuer agents
        else:
            self.prediction_model = pred_model
            self.tri_predicted = tri_pred
            self.sobr_predicted = sobr_pred
            self.cluster_labels = cluster_labels

        #appends in the list, victims that are on this agent's assigned cluster
        i = 0
        for label in self.cluster_labels:
            if label == self.assigned_cluster:
                x, y = self.victims_found[i]
                self.assigned_victims.append((x,y))
            i += 1

        print(f'Assigned Cluster to {self.NAME}: {self.assigned_cluster}')
        print(f'Assigned Victims of Cluster {self.assigned_cluster}:\n{self.assigned_victims}\n')

# -------------------------------------------------------------------------------

    def deliberate(self):
        pass

# -------------------------------------------------------------------------------

    """training algorithm from first assigment, trains on the same random seed"""

    def learn_model(self, data_set_path):

        # Test Parameters--------------
        parametros = {
            'criterion': ['entropy'],
            'max_depth': [8],
            'min_samples_leaf': [64, 256, 512]
        }

        # Loading Data Frame-----------------
        df = pd.read_csv(data_set_path)
        x = df.drop(columns=['gcs', 'avpu', 'tri', 'sobr'])
        y = df['tri']
        #util.plot_graph(df, df['tri'], df['sobr'])

        # Separation of test and train data--------------
        x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size=0.2, random_state=42)
        modelo = sk.tree.DecisionTreeClassifier(random_state=42)
        print(f'Quantidade de amostras pra treino: {len(x_train)}')

        # Cross Validation Training---------------
        clf = sk.model_selection.GridSearchCV(modelo, parametros, cv=3, scoring='f1_macro', verbose=4, return_train_score=True)
        clf.fit(x_train,y_train)

        # Choosing the best model------------------
        top3_indices = np.argsort(clf.cv_results_['mean_test_score'])[::-1][:3] #takes the 3 with the best f-score
        menor_erro = float('inf')
        best_modelo = None
        best_idx = None

        for rank, idx in enumerate(top3_indices):
            params_idx = clf.cv_results_['params'][idx]
            print(f'\n=== Modelo {idx} === \n{params_idx}')

            modelo_instancia = sk.tree.DecisionTreeClassifier(**params_idx, random_state=42)
            modelo_instancia.fit(x_train, y_train)

            variancia = self.variance_folds(clf, idx)
            vies_medio = abs(clf.cv_results_['mean_test_score'][idx] - clf.cv_results_['mean_train_score'][idx])
            print(f'Variancia = {variancia:.6f}, Vies = {vies_medio:.6f}')

            estimativa_erro = variancia + vies_medio
            print(f'Estimativa de erro: {estimativa_erro}')

            if estimativa_erro < menor_erro:
                if clf.cv_results_['mean_test_score'][idx] > 0.75: #filtra underfitteds
                    menor_erro = estimativa_erro
                    best_modelo = modelo_instancia
                    best_idx = idx

        if not best_modelo: # if none is chosen, takes the one with the highest mean f-score
            best_modelo = clf.best_estimator_
            best_idx = clf.best_index_

        print(f'\nMelhor modelo geral escolhido foi o {best_idx}: \n{clf.cv_results_['params'][best_idx]}')

        # Printing the Results--------------
        # Accuracies
        y_pred_train = best_modelo.predict(x_train)
        acc_train = sk.metrics.accuracy_score(y_train, y_pred_train) * 100
        print(f'\nAcuracia com dados de treino: {acc_train:.2f}%')

        y_pred_test = best_modelo.predict(x_test)
        acc_test = sk.metrics.accuracy_score(y_test, y_pred_test) * 100
        print(f'Acuracia com dados de teste: {acc_test:.2f}%')

        #util.plot_tree(best_modelo,x_train, y_train)
        #util.plot_cmatrix(y_test, y_pred_test, False)

        # Retraining with the whole data set----------------------
        best_modelo.fit(x,y)
        y_pred_test = best_modelo.predict(x)
        acc_test = sk.metrics.accuracy_score(y, y_pred_test) * 100
        print(f'Acuracia do modelo retreinado: {acc_test:.2f}%')

        return best_modelo

# -------------------------------------------------------------------------------

    """MLP regressor training algorithm from first assigment, trains on the same random seed"""

    def learn_model_sobr(self, data_set_path):

        df = pd.read_csv(data_set_path)

        X = df.drop(columns=["gcs", "avpu", "tri", "sobr"])
        y = df["sobr"]

        X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = sk.neural_network.MLPRegressor(
            hidden_layer_sizes=(10, 10, 10, 10, 10, 10),
            activation="tanh",
            solver="sgd",
            learning_rate="adaptive",
            learning_rate_init=0.025,
            max_iter=2000,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = sk.metrics.mean_squared_error(y_test, y_pred)
        print(f"[SOBR-MLP] MSE Test: {mse:.4f}")

        model.fit(X, y)

        return model

# -------------------------------------------------------------------------------

    def predict_data_set(self, data_set_path, prediction_model):

        df_final = pd.read_csv(data_set_path)
        x = df_final.drop(columns=['gcs', 'avpu'])
        y_pred_test = prediction_model.predict(x)
        print(f'==== Predicted the following [tri]s ====\n{y_pred_test}\n')
        return y_pred_test

# -------------------------------------------------------------------------------

    def predict_sobr(self, data_set_path, model):

        df = pd.read_csv(data_set_path)

        X = df.drop(columns=["gcs", "avpu"])   # igual ao treino

        sobr_pred = model.predict(X)

        print(f"[SOBR-MLP] Predicted SOBR for found victims:")
        print(sobr_pred)

        return sobr_pred    

# -------------------------------------------------------------------------------

    def variance_folds(self, clf, index):

        fold_scores = [
            clf.cv_results_[f'split{i}_test_score'][index]
            for i in range(clf.cv)]

        return np.var(fold_scores)

# -------------------------------------------------------------------------------

    def clusterize(self, n_clusters):

        import numpy as np
        from sklearn.cluster import KMeans

        pts = np.array([list(coord) for coord in self.victims_found])
        n = len(pts)
        if n == 0:
            return np.array([])

        # 1) Run KMeans to get centroids
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(pts)
        centers = kmeans.cluster_centers_

        # 2) compute desired sizes: distribute remainder among first clusters
        base = n // n_clusters
        rem = n % n_clusters
        capacities = [base + (1 if i < rem else 0) for i in range(n_clusters)]

        # 3) compute all distances (n_points x n_clusters)
        dists = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)

        # 4) create list of (dist, point_idx, cluster_idx) and sort
        rows = []
        for i in range(n):
            for c in range(n_clusters):
                rows.append((dists[i, c], i, c))
        rows.sort(key=lambda x: x[0])

        # 5) greedy assignment respecting capacities
        labels = -1 * np.ones(n, dtype=int)
        filled = [0] * n_clusters
        for dist, i, c in rows:
            if labels[i] != -1:
                continue  # already assigned
            if filled[c] < capacities[c]:
                labels[i] = c
                filled[c] += 1

        # 6) if any point unassigned (edge cases), assign to nearest cluster with room
        for i in range(n):
            if labels[i] == -1:
                # assign to nearest cluster that has capacity
                order = np.argsort(dists[i])
                for c in order:
                    if filled[c] < capacities[c]:
                        labels[i] = c
                        filled[c] += 1
                        break

        # debug prints
        unique, counts = np.unique(labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"Cluster {cluster_id} has {count} elements")
        print(f"capacities = {capacities}, filled = {filled}")

        return labels

# ===============================================================================