#===============================================================================

import pygame
import colorsys

import csv
import networkx as nx
from physical_agent import PhysAgent
from agent_minds import ExplorerMind, RescuerMind
from environment import Env
from constants import VS
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

    # Set cell width and height
    cell_w = 800 / 94
    cell_h = 700 / 94

    # Clear the screen
    screen = pygame.display.set_mode((800, 700))
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
    for x in range(94):
        for y in range(94):
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


    # Draw a marker at the base
    rect = pygame.Rect(46 * cell_w,
                       46 * cell_h, cell_w, cell_h)
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

    resc_agents = []

    resc_agent1 = RescuerMind(environment,'94x94_408v/env_agent_config4.txt', DATA_FOLDER, exploration_map,
                              None, None, None, Master_Agent=True)
    resc_agent2 = RescuerMind(environment, '94x94_408v/env_agent_config5.txt', DATA_FOLDER, exploration_map,
                              resc_agent1.prediction_model, resc_agent1.tri_predicted, resc_agent1.cluster_labels)
    resc_agent3 = RescuerMind(environment, '94x94_408v/env_agent_config6.txt', DATA_FOLDER, exploration_map,
                              resc_agent1.prediction_model, resc_agent1.tri_predicted, resc_agent1.cluster_labels)
    resc_agents.append(resc_agent1)
    resc_agents.append(resc_agent2)
    resc_agents.append(resc_agent3)

    for agent in resc_agents:
        agent.set_state(VS.ACTIVE)

    debug_draw(environment,resc_agent1.cluster_labels)

    write_cluster_files(resc_agents)



if __name__ == '__main__':
    main ()

#===============================================================================
