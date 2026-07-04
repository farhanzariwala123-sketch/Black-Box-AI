import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import heapq
import itertools
import time

# --- 1. PRIORITY LOGIC ---
def calculate_priority(emotion, time_waiting):
    priority_map = {
        ('Negative', 'Most'): 1, ('Negative', 'Some'): 2, ('Neutral', 'Most'): 3,
        ('Negative', 'Just'): 4, ('Neutral', 'Some'): 5, ('Positive', 'Most'): 6,
        ('Positive', 'Some'): 7, ('Neutral', 'Just'): 8, ('Positive', 'Just'): 9
    }
    return priority_map.get((emotion, time_waiting), 10)

# --- 2. A* PATHFINDING ALGORITHM ---
def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    # Up, Down, Left, Right
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
            
            # Check boundaries
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                # Obstacle check: allow entering start and goal table, but not other tables
                if grid[neighbor] == 1 and neighbor != goal and neighbor != start:
                    continue
                    
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []

# --- 3. RESTAURANT ENVIRONMENT & SIMULATION ---
def simulate_restaurant():
    # Create 8x7 Grid (8 rows, 7 cols)
    # 0 = Aisle, 1 = Obstacle/Table
    grid = np.zeros((8, 7))
    kitchen_pos = (0, 3) # Top Center
    
    # Using 0-indexed positioning (Row 3 Col 2 is (2,1))
    tables = {
        'T1': (2, 1), 'T2': (2, 3), 'T3': (2, 5),
        'T4': (4, 1), 'T5': (4, 3), 'T6': (4, 5),
        'T7': (6, 1), 'T8': (6, 3), 'T9': (6, 5)
    }
    
    for t in tables.values():
        grid[t] = 1

    # Simulated Data from your Audio AI / Text AI outputs
    customers = [
        {'id': 'T1', 'pos': tables['T1'], 'emotion': 'Positive', 'time': 'Some'}, # Green
        {'id': 'T2', 'pos': tables['T2'], 'emotion': 'Neutral', 'time': 'Just'},  # Orange
        {'id': 'T3', 'pos': tables['T3'], 'emotion': 'Negative', 'time': 'Some'}, # Red (Priority 2)
        {'id': 'T4', 'pos': tables['T4'], 'emotion': 'Positive', 'time': 'Just'}, # Green
        {'id': 'T5', 'pos': tables['T5'], 'emotion': 'Neutral', 'time': 'Most'},  # Orange (Priority 3)
        {'id': 'T6', 'pos': tables['T6'], 'emotion': 'Negative', 'time': 'Most'}, # Red (Priority 1)
        {'id': 'T7', 'pos': tables['T7'], 'emotion': 'Positive', 'time': 'Most'}, # Green
        {'id': 'T8', 'pos': tables['T8'], 'emotion': 'Neutral', 'time': 'Some'},  # Orange
        {'id': 'T9', 'pos': tables['T9'], 'emotion': 'Negative', 'time': 'Just'}  # Red
    ]
    
    # Sort by priority and take top 3 targets
    customers.sort(key=lambda x: calculate_priority(x['emotion'], x['time']))
    targets = customers[:3]
    print(f"Robot loading 3 tokens. Targets: {[t['id'] for t in targets]}")

    # --- 4. MULTI-STOP ROUTING (TSP) ---
    # Find the shortest sequence to visit all 3 tables and return to kitchen
    best_path = []
    best_dist = float('inf')
    
    # Generate all possible visit orders for the 3 tables
    for perm in itertools.permutations(targets):
        current_dist = 0
        current_full_path = []
        route_points = [kitchen_pos] + [c['pos'] for c in perm] + [kitchen_pos]
        
        valid = True
        for i in range(len(route_points)-1):
            p = astar(grid, route_points[i], route_points[i+1])
            if not p:
                valid = False
                break
            current_dist += len(p)
            # Add path segments, avoiding duplicate coordinates at overlapping stops
            if i < len(route_points)-2:
                current_full_path.extend(p[:-1])
            else:
                current_full_path.extend(p)
                
        if valid and current_dist < best_dist:
            best_dist = current_dist
            best_path = current_full_path

    # --- 5. ANIMATION SETUP ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 8))
    
    color_map = {'Negative': 'red', 'Neutral': 'orange', 'Positive': 'green'}
    
    for step in best_path:
        ax.clear()
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(7.5, -0.5)
        ax.set_xticks(np.arange(-0.5, 7.5, 1))
        ax.set_yticks(np.arange(-0.5, 8.5, 1))
        ax.grid(which='major', color='black', linestyle='-', linewidth=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Draw Kitchen
        kitchen_rect = patches.Rectangle((kitchen_pos[1]-0.5, kitchen_pos[0]-0.5), 1, 1, facecolor='lightblue')
        ax.add_patch(kitchen_rect)
        ax.text(kitchen_pos[1], kitchen_pos[0], 'KITCHEN', ha='center', va='center', fontweight='bold', fontsize=8)
        
        # Draw Tables
        for c in customers:
            pos = c['pos']
            col = color_map[c['emotion']]
            rect = patches.Rectangle((pos[1]-0.5, pos[0]-0.5), 1, 1, facecolor=col, alpha=0.8)
            ax.add_patch(rect)
            ax.text(pos[1], pos[0], f"{c['id']}", ha='center', va='center', color='white', fontweight='bold')

        # Draw Robot (Small Red Box Marker 's')
        ax.plot(step[1], step[0], marker='s', color='red', markersize=18, markeredgecolor='black')
        
        ax.set_title(f"Delivering 3 Tokens | Optimal Multi-Target Route")
        plt.draw()
        plt.pause(0.2) # Adjust animation speed here
        
    print("All 3 tokens delivered! Robot returned to the kitchen.")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    simulate_restaurant()