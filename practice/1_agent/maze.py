import numpy as np
from collections import deque

class NumpyMazeMaster:
    def __init__(self, size=(10, 10)):
        # Create a random grid: 80% open (0), 20% walls (1)
        self.grid = np.random.choice([0, 1], size=size, p=[0.8, 0.2])
        self.dims = size
        
        # Ensure start and end are open
        self.grid[0, 0] = 0
        self.grid[-1, -1] = 0

    def solve_and_mark(self, start, end, method='bfs'):
        """
        Solves the maze and returns a grid with the path marked as 2.
        """
        # Reuse the logic from the Python section (simplified here)
        queue = deque([(start, [start])])
        visited = {start}
        final_path = None

        while queue:
            # BFS Logic
            (r, c), path = queue.popleft() if method == 'bfs' else queue.pop()
            
            if (r, c) == end:
                final_path = path
                break

            # Neighbors: Up, Down, Left, Right
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.dims[0] and 0 <= nc < self.dims[1]:
                    if self.grid[nr, nc] != 1 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [(nr, nc)]))
        
        # --- NumPy Visualization Logic ---
        result_grid = self.grid.copy()
        if final_path:
            print(f"Path found with {len(final_path)} steps.")
            # Extract rows and cols from path list to index numpy array
            rows = [p[0] for p in final_path]
            cols = [p[1] for p in final_path]
            
            # Mark the path as '8' (visually distinct)
            result_grid[rows, cols] = 8 
        else:
            print("No path possible.")
            
        return result_grid

# --- Usage ---
maze_bot = NumpyMazeMaster(size=(6, 6))
print("Original Map (1=Wall):")
print(maze_bot.grid)

print("\nSolved Map (8=Path):")
solved_map = maze_bot.solve_and_mark((0,0), (5,5), method='bfs')
print(solved_map)