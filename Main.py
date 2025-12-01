#===============================================================================

import pygame
import colorsys

import csv
import networkx as nx
from physical_agent import PhysAgent
from agent_minds import ExplorerMind, RescuerMind
from environment import Env
# clustering for saved victims
try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None
from constants import VS, GRID_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT
# import utilitarios as util

#===============================================================================

#Parameters
ENV_FOLDER = '94x94_408v/'
DATA_FOLDER = 'datasets/'

#===============================================================================

def get_exploration_map(expl_agents):

    mental_maps = []

    for agent in expl_agents:
        if agent.get_state() == VS.ENDED:
            mental_map = agent.get_mental_map()
            mental_maps.append(mental_map)

    exploration_map = nx.compose_all(mental_maps)
    return exploration_map

#-------------------------------------------------------------------------------

def write_victims_files(expl_agents):

    writen = []

    with open(DATA_FOLDER + "env_victims_found.txt", "w", newline="", encoding="utf-8") as file:
        for agent in expl_agents:
            if agent.get_state() == VS.ENDED:
                for row in agent.victims:
                    if row[0] not in writen:
                        writen.append(row[0])
                        x, y = row[0]
                        file.write(f'{x},{y}\n')

    writen = []

    with open( DATA_FOLDER + "data_found.csv", "w", newline="", encoding="utf-8") as file:
        for agent in expl_agents:
            if agent.get_state() == VS.ENDED:
                header = ['location', 'idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim', 'gcs', 'avpu']
                agent.victims.insert(0, header)

                writer = csv.writer(file)
                for row in agent.victims:
                    if row not in writen:
                        writen.append(row)
                        writer.writerow(row[1:])

#-------------------------------------------------------------------------------

def write_cluster_files(resc_agents):

    for agent in resc_agents:
        with open(DATA_FOLDER + f'cluster_{agent.assigned_cluster}.txt', "w", newline="", encoding="utf-8") as file:
            for row in agent.assigned_victims:
                x, y = row
                file.write(f'{x},{y}\n')

# -------------------------------------------------------------------------------

# 'labels' works fine as being either 'predicted_tri' or 'cluster_labels', from the recuer agent
def debug_draw(env, labels):

    # Set cell width and height (use constants to avoid hardcoding)
    cell_w = SCREEN_WIDTH / GRID_SIZE
    cell_h = SCREEN_HEIGHT / GRID_SIZE

    # Clear the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    screen.fill(VS.WHITE)

    # configuration for obstacles coloring
    # h,  s,   lc, ld:
    # 13, 100, 100, 65 red tonalities
    # 275,100, 100, 65 purple
    # 90, 35,  100, 40 green tonaliies
    #  0,  0,  100, 50 gray tonnalitites
    hue = 0  # Not relevant for grayscale; 0=Red, 120=green, 240=blue till 360
    saturation = 0  # 40 = Red  0 = Grayscale
    lightness_clear = 100  # 100 = White
    lightness_dark = 40  # 0 = Black


    # Draw the grid
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(x * cell_w, y * cell_h, cell_w, cell_h)
            pygame.draw.rect(screen, (230, 230, 230), rect, 1)

            if env.obst[x][y] == VS.OBST_WALL:
                rgb_int = VS.BLACK
            else:
                if env.obst[x][y] == VS.OBST_NONE:
                    rgb_int = VS.WHITE
                else:
                    perc = env.obst[x][y] / 3.0
                    lightness = (1 - perc) * lightness_clear + perc * lightness_dark

                    # convert HSL color to RGB
                    rgb_color = colorsys.hls_to_rgb(hue / 360.0, lightness / 100.0, saturation / 100.0)

                    # Convert RGB values to integers in the range [0, 255]
                    rgb_int = tuple(int(c * 255) for c in rgb_color)

            obst_rect = pygame.Rect(x * cell_w + 1, y * cell_h + 1, cell_w - 2, cell_h - 2)
            pygame.draw.rect(screen, rgb_int, obst_rect)


    # Draw a marker at the base (base coordinate assumed at center index GRID_SIZE//2)
    base_idx = GRID_SIZE // 2
    rect = pygame.Rect(base_idx * cell_w,
                       base_idx * cell_h, cell_w, cell_h)
    pygame.draw.rect(screen, VS.CYAN, rect, 4)

    victims = []

    with open('datasets/env_victims_found.txt', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            x = int(row[0])
            y = int(row[1])
            victims.append((x, y))


    # Draw the victims
    v = 0
    for victim in victims:
        victim_rect = pygame.Rect(victim[0] * cell_w + 1,
                                  victim[1] * cell_h + 1,
                                  cell_w - 1, cell_h - 1)
        c = labels[v]
        pygame.draw.ellipse(screen, VS.VIC_COLOR_LIST[c], victim_rect)
        v += 1

    pygame.image.save(screen, "grid_snapshot2.png")

    # Update the display
    pygame.display.update()

#===============================================================================

def main ():

    environment = Env(ENV_FOLDER, ENV_FOLDER)

    # Exploration Phase-------------------

    expl_agents = []

    expl_agent1 = ExplorerMind(environment, '94x94_408v/env_agent_config1.txt', [3, 4, 5, 6, 7, 0, 1, 2])
    expl_agent2 = ExplorerMind(environment, '94x94_408v/env_agent_config2.txt', [7, 6, 5, 4, 3, 2, 1, 0])
    expl_agent3 = ExplorerMind(environment, '94x94_408v/env_agent_config3.txt', [1, 2, 3, 4, 0, 7, 6, 5])
    expl_agents.append(expl_agent1)
    expl_agents.append(expl_agent2)
    expl_agents.append(expl_agent3)

    for agent in expl_agents:
        agent.set_state(VS.ACTIVE)

    environment.run()

    write_victims_files(expl_agents)
    exploration_map = get_exploration_map(expl_agents)
    
    # removing explorer agents from environment to free some space
    n_agents = len(expl_agents)
    for i in range(n_agents):
        environment.agents.pop(0)


    # Rescuing Phase---------------------

    # Define rescuer configuration files (one entry per rescuer). This makes the
    # number of rescuers explicit and easy to change without scattering literals.
    rescuer_configs = [
        '94x94_408v/env_agent_config4.txt',
        '94x94_408v/env_agent_config5.txt',
        '94x94_408v/env_agent_config6.txt'
    ]

    resc_agents = []

    # Create the master rescuer first and let it compute clusters with n_clusters
    # equal to the total number of rescuers (dynamic)
    total_rescuers = len(rescuer_configs)
    master_cfg = rescuer_configs[0]
    master = RescuerMind(environment, master_cfg, DATA_FOLDER, exploration_map,
                         pred_model=None, tri_pred=None, sobr_pred=None, cluster_labels=None,
                         Master_Agent=True, assigned_cluster=0, n_clusters=total_rescuers)
    resc_agents.append(master)

    # Create remaining rescuers, passing master's models and labels and assigning
    # each rescuer an index-based assigned_cluster (0..n-1)
    for idx, cfg in enumerate(rescuer_configs[1:], start=1):
        agent = RescuerMind(environment, cfg, DATA_FOLDER, exploration_map,
                            pred_model=master.prediction_model,
                            tri_pred=master.tri_predicted,
                            sobr_pred=master.sobr_predicted,
                            cluster_labels=master.cluster_labels,
                            Master_Agent=False,
                            assigned_cluster=idx)
        resc_agents.append(agent)

    for agent in resc_agents:
        agent.set_state(VS.ACTIVE)

    # Run the environment so rescuer agents' deliberate() is executed
    environment.run()

    # draw using master (first rescuer) cluster labels
    try:
        debug_draw(environment, master.cluster_labels)
    except Exception:
        # fallback: if master not available or has no labels, skip drawing
        pass

    write_cluster_files(resc_agents)

    # Final summary: print accumulated and per-agent results + concise stats
    print('\n=== FINAL SUMMARY ===')
    # Environment already provides detailed printing routines
    try:
        environment.print_acum_results()
    except Exception:
        pass

    try:
        environment.print_results()
    except Exception:
        pass

    # Concise summary: total saved, energy consumed per agent and saved coords
    total_saved = sum(1 for lst in environment.saved if lst)
    print(f"\nConcise summary:\n - Total victims saved: {total_saved} of {environment.nb_of_victims}")

    print('\n - Energy consumed per agent:')
    for phy in environment.agents:
        try:
            consumed = phy.mind.TLIM - phy._rtime
            print(f"   {phy.mind.NAME}: consumed {consumed:.2f} of {phy.mind.TLIM:.2f}")
        except Exception:
            # defensive: if attributes aren't present, skip
            print(f"   {getattr(phy.mind, 'NAME', repr(phy.mind))}: energy info not available")

    saved_indices = [i for i, lst in enumerate(environment.saved) if lst]
    if saved_indices:
        # Use the Master agent's precomputed clustering (one cluster per rescuer)
        # Build a coordinate -> cluster label mapping from the master's labels
        coord_to_label = {}
        try:
            master = resc_agents[0]
            for idx, coord in enumerate(master.victims_found):
                # master.cluster_labels aligns with master.victims_found
                coord_to_label[coord] = master.cluster_labels[idx]
        except Exception:
            coord_to_label = {}

        # Map cluster -> list of victim indices (use master's labels when available)
        cluster_map = {}
        for vid in saved_indices:
            coords = environment.victims[vid]
            lab = coord_to_label.get(coords, None)
            cluster_map.setdefault(lab, []).append(vid)

        # Map assigned cluster -> rescuer agent (should be one rescuer per cluster)
        cluster_to_rescuer = {}
        for agent in resc_agents:
            cluster_to_rescuer[agent.assigned_cluster] = agent

        print('\n - Saved victims by cluster (master labels) and assigned rescuer:')
        for lab in sorted(cluster_map.keys(), key=lambda x: (-1 if x is None else x)):
            agent = cluster_to_rescuer.get(lab, None)
            agent_name = getattr(agent, 'NAME', getattr(agent, 'NAME', 'Unassigned')) if agent else 'Unassigned'
            print(f"   Cluster {lab}: assigned to {agent_name}")
            for vid in cluster_map[lab]:
                coords = environment.victims[vid]
                rescuers = environment.saved[vid]
                names = [getattr(phy.mind, 'NAME', repr(phy.mind)) for phy in rescuers]
                print(f"     Victim {vid} at {coords}: saved by {', '.join(names)}")

            # Write per-rescuer saved-sequence files: seq_ag{agent_idx}_1.txt (id,x,y)
        for agent in resc_agents:
            try:
                agent_idx = (agent.assigned_cluster or 0) + 1
            except Exception:
                agent_idx = 0
            fname = DATA_FOLDER + f"seq_ag{agent_idx}_1.txt"
            seq = getattr(agent, 'saved_sequence', [])
            # ensure file exists; write lines in order: id,x,y
            with open(fname, "w", encoding="utf-8") as f:
                for vid in seq:
                    try:
                        x, y = environment.victims[vid]
                        f.write(f"{vid},{x},{y}\n")
                    except Exception:
                        # defensive: skip if mapping fails
                        continue
    else:
        print('\n - No victims were saved')


if __name__ == '__main__':
    main ()

#===============================================================================
