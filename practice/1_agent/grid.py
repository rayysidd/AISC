import random

class Agent:
    def __init__(self,rows,cols,battery):
        self.grid=[[random.choice([0,1,2]) for _ in range(cols)] for _ in range(rows)]
        self.pos=[0,0]
        self.battery=battery
        self.rows=rows
        self.cols=cols
        
    def get_percept(self):
        # What is in the current tile?
        r, c = self.pos
        return self.grid[r][c]
    
    def act(self):
        if self.battery <= 0:
            print("Battery dead. Stopping.")
            return False

        status = self.get_percept()

        # LOGIC: Modify this based on specific problem (Clean dirt? Check wall?)
        if status == 2: 
            print(f"Found target at {self.pos}. Cleaning...")
            self.grid[self.pos[0]][self.pos[1]] = 0
            self.battery -= 2 # Higher cost for action
        else:
            print(f"Nothing to do at {self.pos}.")
            self.battery -= 1 # Movement cost

        return True

    def move(self, direction):
        # direction: [d_row, d_col] e.g., [0, 1] is Right
        new_r = self.pos[0] + direction[0]
        new_c = self.pos[1] + direction[1]

        # Boundary and Obstacle Check
        if 0 <= new_r < self.rows and 0 <= new_c < self.cols:
            if self.grid[new_r][new_c] != 1: # Assuming 1 is an obstacle
                self.pos = [new_r, new_c]
            else:
                print("Blocked by obstacle.")