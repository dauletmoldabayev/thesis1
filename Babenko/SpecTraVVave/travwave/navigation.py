from __future__ import division

from .discretization import resample

def ortho_direction(p1, p2, step):   # takes in bifurcation points p1, p2 and step_size
                                     # computes point 'p3' and orthogonal direction vector 'd'
    """
    Returns pstar such that
        p3 = p2 + step*(p2-p1)
    """
    dp = (p2[0] - p1[0], p2[1] - p1[1])
    # orthogonal direction
    direction = (-dp[1], dp[0])

    p3 = (p2[0] + dp[0]*step, p2[1] + dp[1]*step)
    return p3, direction

class Navigator(object):
    """
    Runs the Newton iterator and stores the computed solutions.
    """

    def __init__(self, solve, size=32):  # 'solve' is the 'solve'-method of object of class Solver
        """
        solve: solve function
        size: size for the navigation
        """
        self.solve = solve   # Solver object
        self.size = size     # number of grid points that is inherent from Discretization object

    # the indices for velocity and amplitude
    velocity_, amplitude_ = (0, 1)

    def __getitem__(self, index):    # internal routine for accessing stored solutions by index number
        return self._stored_values[index]

    def __len__(self):    # returns the number of stored solutions
        return len(self._stored_values)

    def initialize (self, current, p, p0): # 'current' = initial guess for wave
                                           # 'p' = bifurcation point (c0, a=0), 'p0' = (c0, a = -epsilon)
                                           # 'epsilon' is the step_size to move up on bifurcation curve
        """
        Creates a list for solutions and stores the first solution (initial guess).
        """
        self._stored_values = []
        variables = [0]     # integration constant B
        self._stored_values.append({'solution': resample(current, self.size), 'integration constant': variables, 'current': p, 'previous': p0 }) # the first stored solution is the set of initial guess parameters, i.e. (wave, B, p, p0)

    def compute_direction(self, p1, p2, step):
        """
        Strategy for a new parameter direction search.
        """
        return ortho_direction(p1, p2, step)

    def run(self, N):
        """
        Iterates the solver N times, navigating over the bifurcation branch and storing found solutions.
        """
        for i in range(N):
            self.step() # runs Newton iterator one time

    def prepare_step(self, step, p2, p1): # computes p3 and direction for new iteration of Newton solver
        """
        Return the necessary variables to run the solver.
        """
        p3, direction = self.compute_direction(p1, p2, step)
        return p3, direction

    def two_parameter_points(self, index): # returns bifurcation points p2 and p1 from stored solutions by index
        p2 = self[index]['current']
        p1 = self[index]['previous']
        return p2, p1

    def run_solver(self, current, p3, direction): # initiates the Newton iteration on given data
        new_wave, variables, pstar = self.solve(current, p3, direction)
        return new_wave, variables, pstar

    def refine(self, resampling, sol, p, direction): # takes in wave solution on sparse grid, refines it and gives in to 
                                                     # the Newton solver. Then returns the computed wave solution on refined
                                                     # grid, constant B and updated p-bifurcation point.
        """
        Refine from solution `sol` at parameter `p` in direction `direction` in the parameter space.
        """
        sol_ = resample(sol, resampling)
        new, variables, p_ = self.run_solver(sol_, p, direction)
        return new, variables, p_

    def refine_at(self, resampling, index=-1):     # refine a wave solution at a point on bifurcation curve
                                                   # the point is specified by index
        """
        Refine using a direction orthogonal to the last two parameter points.
        """
        p2, p1 = self.two_parameter_points(index)
        p, dir = self.prepare_step(0, p2, p1)
        current = self[index]['solution']
        return self.refine(resampling, current, p, dir)

    def step(self):   # extracts data from stored solutions and passes them to Newton solver
                      # then stores the new solution
        p2, p1 = self.two_parameter_points(index=-1)
        p3, direction = self.prepare_step(1., p2, p1)
        current = self[-1]['solution']
        new, variables, pstar = self.run_solver(current, p3, direction)
        self._stored_values.append({'solution': new, 'integration constant': variables, 'current': pstar, 'previous': p2})

