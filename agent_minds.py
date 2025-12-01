"""Made with 22/10/2025 version of env files on teacher's github"""
# ===============================================================================

import os
import csv
import numpy as np
import pandas as pd
import sklearn as sk
import random
import math
import pygad
# import utilitarios as util
from constants import VS, DEFAULT_RETURN_MARGIN
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
    def __init__(self, env, config_file, data_folder, exploration_map,
                 pred_model=None, tri_pred=None, sobr_pred=None, cluster_labels=None,
                 Master_Agent=False, assigned_cluster=None, n_clusters=None):
        super().__init__(env, config_file)
        self.exploration_map = exploration_map
        self.data_folder = data_folder
        self.training_data_set_path = os.path.join(self.data_folder, "training_data.csv")
        self.found_data_set_path = os.path.join(self.data_folder, "data_found.csv")
        self.env_victims_found_path = os.path.join(self.data_folder, "env_victims_found.txt")
        self.victims_found = [] #stores the coordinates of all found victims in exploration phase
        self.assigned_victims = [] #stores the coordinates of victims that the agent was assigned to rescue
        self.base_coord = (env.dic["BASE"][0], env.dic["BASE"][1])

        # assigned_cluster can be provided externally (preferred). Fallback to legacy
        # name-parsing behaviour for backwards compatibility.
        if assigned_cluster is not None:
            self.assigned_cluster = int(assigned_cluster)
        else:
            try:
                self.assigned_cluster = (int(''.join([c for c in self.NAME if c.isdigit()])) - 1)
            except Exception:
                self.assigned_cluster = None

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
            # Use provided n_clusters when available; otherwise default to 1 to avoid crash
            try:
                use_n = int(n_clusters) if n_clusters is not None else 1
            except Exception:
                use_n = 1
            self.cluster_labels = self.clusterize(use_n)
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

        #where we would call calculate_optimal_sequence, although here i'm executing it on top of all victims
        #ideally we would execute it for each different tri group
        self.rescue_path, self.rescue_path_cost = self.calculate_optimal_sequence(self.assigned_victims, self.base_coord)
        # persistent position tracked by the mind (keeps in sync with physical part)
        try:
            self.position = (self._AbstAgent__phy.x, self._AbstAgent__phy.y)
        except Exception:
            # fallback if physical reference not available yet
            self.position = self.base_coord
        # safety margin (in rtime units) to reserve for returning to base
        # sourced from constants to avoid scattered magic numbers
        self.return_margin = getattr(self, 'return_margin', DEFAULT_RETURN_MARGIN)
    
# -------------------------------------------------------------------------------

    def deliberate(self):
        """
        Deliberate method for the RESCUER agent.
        Follows the rescue path generated by the GA, rescues victims, and ensures
        the agent returns to base before the battery becomes critical.

        Mirrors the structure of the explorer's deliberate() for consistency:
        - States: RESCUING → RETURNING → FINISHED
        """

        # -------------------------------
        # Initialize stateful variables
        # -------------------------------
        if not hasattr(self, "state"):
            self.state = "RESCUING"
            self.rescue_step_index = 0
            print(f"{self.NAME}: Starting RESCUE operation.")
            # ensure mind has a current position tracked
            try:
                self.position = (self._AbstAgent__phy.x, self._AbstAgent__phy.y)
            except Exception:
                self.position = self.base_coord

        # always sync mind position with physical part at the start of deliberation
        try:
            self.position = (self._AbstAgent__phy.x, self._AbstAgent__phy.y)
        except Exception:
            pass

        # If no victims or no path → nothing to do
        if not self.rescue_path or not self.assigned_victims:
            self.state = "RETURNING"

        # -------------------------------------------------------
        # BATTERY CHECK — identical behavior to the EXPLORER
        # -------------------------------------------------------
        def enough_battery_to_continue():
            # Simple heuristic: return to base if battery < distance_to_base * cost_per_step
            # Uses the mind-tracked position (kept in sync with physical agent)
            current = self.position
            _, cost_back = self.get_cheapest_path(current, self.base_coord)
            # use the physical part remaining time (rtime) via the public getter
            # use a configurable margin to increase safety (reserve rtime for unexpected costs)
            margin = getattr(self, 'return_margin', 8)
            return self.get_rtime() > cost_back + margin  # margin

        # ================================================
        # PHASE 1 — RESCUING
        # ================================================
        if self.state == "RESCUING":

            # Battery too low → interrupt rescue and go home
            if not enough_battery_to_continue():
                print(f"{self.NAME}: Battery low → returning to base BEFORE finishing rescue.")
                self.state = "RETURNING"
                # Precompute path to base
                self.return_path, _ = self.get_cheapest_path(self.position, self.base_coord)
                self.return_index = 0

            else:
                # Normal rescue progression
                if self.rescue_step_index >= len(self.rescue_path):
                    print(f"{self.NAME}: Finished rescue path. Returning to base.")
                    self.state = "RETURNING"
                    self.return_path, _ = self.get_cheapest_path(self.position, self.base_coord)
                    self.return_index = 0
                else:
                    # Continue toward next waypoint
                    target = self.rescue_path[self.rescue_step_index]

                    # If already at that coordinate → progress
                    if self.position == target:

                        # RESCUE LOGIC
                        if target in self.assigned_victims:
                            print(f"{self.NAME}: Victim rescued at {target}.")
                            # Attempt to perform first aid via the public API
                            try:
                                res = self.first_aid()
                            except Exception:
                                res = False

                            # If time exceeded during first aid, signal termination
                            if res == VS.TIME_EXCEEDED:
                                print(f"{self.NAME}: TIME_EXCEEDED while giving first aid at {target}.")
                                return False

                            # If first aid was successful, ensure environment.saved is updated
                            # accept both boolean True and VS.EXECUTED from first_aid()
                            if res is True or res == VS.EXECUTED:
                                try:
                                    # find victim id and append physical agent to saved if not present
                                    vic_id = self.get_env().victims.index(target)
                                    phy = self._AbstAgent__phy
                                    if phy not in self.get_env().saved[vic_id]:
                                        self.get_env().saved[vic_id].append(phy)

                                    # --- record saved sequence for this rescuer (keep order) ---
                                    if not hasattr(self, 'saved_sequence'):
                                        self.saved_sequence = []
                                    self.saved_sequence.append(vic_id)
                                except Exception:
                                    # don't crash on bookkeeping failures
                                    pass

                                # Spend an extra 1 unit of battery per explicit save
                                try:
                                    self._AbstAgent__phy._rtime -= 1.0
                                except Exception:
                                    pass

                            # remove victim from assigned list regardless of bookkeeping
                            self.assigned_victims.remove(target)

                        # Move to next waypoint
                        self.rescue_step_index += 1

                        # If finished → prepare return
                        if self.rescue_step_index >= len(self.rescue_path):
                            self.state = "RETURNING"
                            self.return_path, _ = self.get_cheapest_path(self.position, self.base_coord)
                            self.return_index = 0
                            return None

                        target = self.rescue_path[self.rescue_step_index]

                    # Issue movement
                    return self.moveTo(*target)

        # ================================================
        # PHASE 2 — RETURNING TO BASE (Explorer-like)
        # ================================================
        if self.state == "RETURNING":

            # Safety: compute return path if missing
            if not hasattr(self, "return_path"):
                self.return_path, _ = self.get_cheapest_path(self.position, self.base_coord)
                self.return_index = 0

            # If arrived at base
            if self.position == self.base_coord:
                print(f"{self.NAME}: Arrived at base. Mission complete.")
                self.state = "FINISHED"
                return False

            # Follow return path step-by-step
            if self.return_index < len(self.return_path):
                target = self.return_path[self.return_index]

                if self.position == target:
                    self.return_index += 1
                    if self.return_index >= len(self.return_path):
                        return None
                    target = self.return_path[self.return_index]

                return self.moveTo(*target)

            # If somehow path ended but not at base
            print(f"{self.NAME}: Warning — return path exhausted, recomputing.")
            self.return_path, _ = self.get_cheapest_path(self.position, self.base_coord)
            self.return_index = 0
            return None

        # ================================================
        # PHASE 3 — FINISHED (Explorer-like final state)
        # ================================================
        if self.state == "FINISHED":
            return False


# -------------------------------------------------------------------------------

    def moveTo(self, tx, ty):
        """Move the agent one step toward target coordinate (tx, ty).
        Assumes target is adjacent (difference in [-1,0,1]). Updates
        self.position when movement succeeds."""
        try:
            cur_x, cur_y = (self._AbstAgent__phy.x, self._AbstAgent__phy.y)
        except Exception:
            cur_x, cur_y = self.position

        dx = tx - cur_x
        dy = ty - cur_y

        # normalize to single-step increments in case a larger delta is passed
        if dx != 0:
            dx = int(dx / abs(dx))
        if dy != 0:
            dy = int(dy / abs(dy))

        result = self.walk(dx, dy)
        if result == VS.EXECUTED:
            # sync mind position with physical
            try:
                self.position = (self._AbstAgent__phy.x, self._AbstAgent__phy.y)
            except Exception:
                self.position = (cur_x + dx, cur_y + dy)

            # After moving, ensure we still have enough battery to return.
            # If not, force the state to RETURNING and precompute return path.
            try:
                margin = getattr(self, 'return_margin', 8)
                _, cost_back = self.get_cheapest_path(self.position, self.base_coord)
                if self.get_rtime() <= cost_back + margin:
                    print(f"{self.NAME}: Battery reaching threshold after move → forcing RETURNING to base.")
                    self.state = "RETURNING"
                    self.return_path, _ = self.get_cheapest_path(self.position, self.base_coord)
                    self.return_index = 0
            except Exception:
                # if path computation fails, don't crash here; the main loop will retry next deliberate
                pass

        return result


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

# -------------------------------------------------------------------------------

    def calculate_optimal_sequence (self, victims, start):

        # Building adjency and path matrixes--------------
        adj_matrix = [] # stores the cheapest costs
        path_matrix = [] # stores the cheapest paths

        # includes start in if it's not inside victims
        if start not in victims:
            all_locations = [start] + victims
        else:
            all_locations = victims

        for row in all_locations:
            adj_matrix_row = []
            path_matrix_row = []
            for col in all_locations:
                path, cost = self.get_cheapest_path(row, col)
                adj_matrix_row.append(cost)
                path_matrix_row.append(path)
            adj_matrix.append(adj_matrix_row)
            path_matrix.append(path_matrix_row)

        adj_matrix = np.array(adj_matrix)
        #print(path_matrix)
        #print(adj_matrix)


        # Initialization of variables-----------------
        num_nodes = adj_matrix.shape[0]
        start_node = all_locations.index(start) # the only index that will never appear in any gene (allways 0 if start not in victims)
        all_nodes = list(range(num_nodes))
        print(f'All_Nodes: {all_nodes}')


        # Genetic Algorithm's fitness function------------------
        def fitness_function(GA, solution, solution_idx):

            path = [start_node] + solution.tolist()

            # print(f'Index: {solution_idx}')
            # print(f'Populacao:\n{GA.population}')
            # print(f'path: {path}')

            cost = 0
            for i in range(len(path) - 1):
                cost += adj_matrix[path[i], path[i + 1]]
            return 1.0 / cost # PyGAD Maximizes


        # GA configuration-------------------
        gene_space = [n for n in all_nodes if n != start_node] # explained right bellow

        ga_instance = pygad.GA(
            num_generations=700,  # number of generations
            num_parents_mating=10,  # number of solutions that are selected as parents
            fitness_func=fitness_function,  # fitness function (pygad searches for highest)
            sol_per_pop=20,  # solutions per generation

            num_genes=num_nodes - 1,  # number of genes per solution (the number of nodes to visit)
            gene_space=gene_space,  # discreetly defines all the possible values for a gene
            allow_duplicate_genes=False, # False because we want all possible values of gene_space
            gene_type=int,

            parent_selection_type="rws",  # selection by roulette
            keep_elitism=2, # the ammount of best solutions that go to the next generation (usually 5% of total pop)
            crossover_type="single_point",
            crossover_probability=0.8, # (usually between 0.7 and 0.9)
            mutation_type="random",
            mutation_probability=0.05 # (usually between 0.01 and 0.05)
        )


        # Runs the GA and gets the results----------------------
        ga_instance.run()

        # Gets the best solution
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        best_sequence = [start_node] + solution.tolist()

        # Gets the cost of the best solution
        best_cost = 0
        for i in range(len(best_sequence) - 1):
            best_cost += adj_matrix[best_sequence[i], best_sequence[i + 1]]

        # Gets the best path according to the best sequence
        best_path = []
        best_path += path_matrix[best_sequence[0]][best_sequence[1]]
        for i in range(1,len(best_sequence)-1):
            path_segment = path_matrix[best_sequence[i]][best_sequence[i + 1]]
            path_segment.pop(0) # removes the first element to avoid repetion
            best_path += path_segment

        print(f'Resulting Best Sequence: indx {solution_idx}\n{best_sequence}')
        print("Cost:", best_cost)
        print(f'Best Path: {best_path}')

        return best_path, best_sequence

# -------------------------------------------------------------------------------

    def get_cheapest_path(self, initial_pos, target_pos):


        src_x, src_y = initial_pos
        target_x, target_y = target_pos

        def heuristic(a, b):  # Euclidean distance
            (x1, y1), (x2, y2) = a, b
            return math.hypot(x2 - x1, y2 - y1)

        path = nx.astar_path(self.exploration_map, (src_x, src_y), (target_x, target_y), heuristic=heuristic)
        cost = nx.path_weight(self.exploration_map, path, weight='weight')
        # print(f'Shortest Path to base: {path}')
        # print(f'Cost: {cost} Rtime: {self.get_rtime()}')

        return path, cost

# ===============================================================================


# ===============================================================================
