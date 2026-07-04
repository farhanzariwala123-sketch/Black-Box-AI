import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import heapq
import random
import time
import os

# --- IMPORT OUR AI BRAIN ---
from integrated_emotion_ai import process_customer_audio

# --- 1. A* PATHFINDING ALGORITHM ---
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
            
        for d in directions:
            neighbor = (current[0] + d[0], current[1] + d[1])
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                # Obstacle check
                if grid[neighbor] == 1 and neighbor != goal and neighbor != start:
                    continue
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
    return []

# --- 2. PRIORITY LOGIC ---
def calculate_priority(emotion, time_waiting):
    priority_map = {
        ('Negative', 'Most'): 1, ('Negative', 'Some'): 2, ('Neutral', 'Most'): 3,
        ('Negative', 'Just'): 4, ('Neutral', 'Some'): 5, ('Positive', 'Most'): 6,
        ('Positive', 'Some'): 7, ('Neutral', 'Just'): 8, ('Positive', 'Just'): 9
    }
    return priority_map.get((emotion, time_waiting), 10)

def categorize_time(minutes):
    if minutes >= 40: return 'Most'
    elif minutes >= 15: return 'Some'
    else: return 'Just'

# --- 3. MASTER SIMULATION STATE MACHINE ---
def run_simulation():
    print("\n[STATE 1] INITIALIZING WORLD...")
    grid = np.zeros((8, 7))
    kitchen_pos = (0, 3)
    
    tables_dict = {
        'T1': (2, 1), 'T2': (2, 3), 'T3': (2, 5),
        'T4': (4, 1), 'T5': (4, 3), 'T6': (4, 5),
        'T7': (6, 1), 'T8': (6, 3), 'T9': (6, 5)
    }
    for t in tables_dict.values():
        grid[t] = 1

    # Simulated database of audio files
    # Automatically generate a list of 20 audio files: ['test_1.wav', 'test_2.wav', ..., 'test_20.wav']
    audio_pool = [f"test_{i}.wav" for i in range(1, 21)]
    
    customers = []
    for t_id, pos in tables_dict.items():
        wait_mins = random.randint(1, 60)
        time_cat = categorize_time(wait_mins)
        audio_file = random.choice(audio_pool)
        
        customers.append({
            'id': t_id,
            'pos': pos,
            'wait_mins': wait_mins,
            'time_cat': time_cat,
            'audio': audio_file,
            'emotion': 'Unknown', # Starts unknown!
            'priority': 99
        })

    # Sort for Perception Patrol (Longest wait first)
    customers.sort(key=lambda x: x['wait_mins'], reverse=True)
    
    # --- UI SETUP ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 8))
    color_map = {'Unknown': 'grey', 'Negative': 'red', 'Neutral': 'orange', 'Positive': 'green'}
    
    def draw_grid(robot_pos, title):
        ax.clear()
        ax.set_xlim(-0.5, 6.5); ax.set_ylim(7.5, -0.5)
        ax.set_xticks(np.arange(-0.5, 7.5, 1)); ax.set_yticks(np.arange(-0.5, 8.5, 1))
        ax.grid(which='major', color='black', linestyle='-', linewidth=1)
        ax.set_xticklabels([]); ax.set_yticklabels([])
        
        # Kitchen
        ax.add_patch(patches.Rectangle((kitchen_pos[1]-0.5, kitchen_pos[0]-0.5), 1, 1, facecolor='lightblue'))
        ax.text(kitchen_pos[1], kitchen_pos[0], 'KITCHEN', ha='center', va='center', fontweight='bold', fontsize=8)
        
        # Tables
        for c in customers:
            pos = c['pos']
            col = color_map[c['emotion']]
            ax.add_patch(patches.Rectangle((pos[1]-0.5, pos[0]-0.5), 1, 1, facecolor=col, alpha=0.8))
            label = f"{c['id']}\n{c['wait_mins']}m\n{c['emotion']}" if c['emotion'] != 'Unknown' else f"{c['id']}\n{c['wait_mins']}m"
            ax.text(pos[1], pos[0], label, ha='center', va='center', color='white', fontweight='bold', fontsize=7)

        # Robot
        ax.plot(robot_pos[1], robot_pos[0], marker='s', color='blue', markersize=18, markeredgecolor='black')
        ax.set_title(title)
        plt.draw()
        plt.pause(0.3)

    # --- STATE 2: PERCEPTION PATROL ---
    print("\n[STATE 2] ROBOT PATROLLING TO LISTEN TO CUSTOMERS...")
    current_robot_pos = kitchen_pos
    
    for c in customers:
        path = astar(grid, current_robot_pos, c['pos'])
        for step in path:
            draw_grid(step, f"Patrolling: Going to {c['id']} to listen...")
            current_robot_pos = step
            
        print(f"\nRobot at {c['id']}. Listening to {c['audio']}...")
        audio_path = os.path.join('data', c['audio'])
        
        # RUN THE AI PIPELINE!
        if os.path.exists(audio_path):
            c['emotion'] = process_customer_audio(audio_path)
        else:
            c['emotion'] = random.choice(['Positive', 'Neutral', 'Negative']) # Fallback
            
        draw_grid(current_robot_pos, f"{c['id']} AI Result: {c['emotion']}!")
        plt.pause(1.0) # Pause so you can see the color change

    # --- STATE 3: KITCHEN INTERRUPT ---
    print("\n[STATE 3] RETURNING TO KITCHEN. FOOD IS READY!")
    path_home = astar(grid, current_robot_pos, kitchen_pos)
    for step in path_home:
        draw_grid(step, "Returning to Kitchen...")
    
    # Randomly select 3 tables that have food ready
    ready_tables = random.sample(customers, 3)
    print(f"Kitchen says food is ready for: {[t['id'] for t in ready_tables]}")
    
    # --- STATE 4: THE BRAIN ---
    print("\n[STATE 4] CALCULATING PRIORITY USING AI RESULTS...")
    for t in ready_tables:
        t['priority'] = calculate_priority(t['emotion'], t['time_cat'])
        print(f"{t['id']} -> {t['emotion']} + {t['time_cat']} = Priority {t['priority']}")
        
    ready_tables.sort(key=lambda x: x['priority'])
    
    # --- STATE 5: EXECUTION (STRICT PRIORITY ROUTING) ---
    print("\n[STATE 5] DELIVERING FOOD IN STRICT PRIORITY ORDER!")
    current_robot_pos = kitchen_pos
    
    for target in ready_tables:
        path = astar(grid, current_robot_pos, target['pos'])
        for step in path:
            draw_grid(step, f"Delivering to {target['id']} (Priority {target['priority']})")
            current_robot_pos = step
            
        print(f"Food delivered to {target['id']}!")
        # Reset table
        target['emotion'] = 'Unknown'
        target['wait_mins'] = 0
        draw_grid(current_robot_pos, f"Food delivered to {target['id']}!")
        plt.pause(1.0)

    # Return Home
    path_home = astar(grid, current_robot_pos, kitchen_pos)
    for step in path_home:
        draw_grid(step, "Mission Complete. Returning to Kitchen.")
        
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_simulation()