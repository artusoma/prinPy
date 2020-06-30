# Import some modules
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.interpolate

def distg(pts, v1, v2):
    '''
    Returns the minimum of points from line and points from vertex
    pts: points to calulate distance from
    v1: vertex j
    v2: vertex j+1
    '''
    D1 = np.linalg.norm(v2 - pts, axis = 1)
    D2 = np.abs(np.cross(v2-v1, v1-pts)) / np.linalg.norm(v2-v1)

    error_t = [np.min([i,j]) for i,j in zip(D1, D2)]

    return sum(error_t)/len(error_t)

def points_in(pts, r1, p):
    '''
    Gets points in r1 from p
    '''
    distances = np.linalg.norm(p - pts, axis = 1)
    return pts[(distances < (r1))]

def points_out(pts, r1, p):
    '''
    Gets points out r1 from p
    '''
    distances = np.linalg.norm(p - pts, axis = 1)
    return pts[(distances > r1)]

def points_btw(pts, r1, r2, p):
    '''
    Gets points between r1 < r2 from p
    '''
    distances = np.linalg.norm(p - pts, axis = 1)
    return pts[(distances < r2) & (distances > r1)]

def reset(x,y):
    dat = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis = 1)
    dat = dat[dat[:,0].argsort()]
    return dat, [dat[0,:]]

def proj_min(X, tck, pt):
    loc_ = scipy.interpolate.splev(X, tck)
    return np.linalg.norm(loc_ - pt)

class CLPCG:
    def __init__(self):
        fit_points = []
        spline_ticks = None
    
    def fit_points(self, x, y, e_max = .2, fmin_error = False):
        '''
        Implements CLPC-greedy algorithm
        Args:
            x: x-data
            y: y-data
            e_max: Max allowed error. If not met, another point P will be added
                to the curve. Authors suggest 1/4 to 1/2 of measurement error
            fmin_error: If true, will find appox. the min value of e_max for the 
                        algorithm to run. e_max must be greater than 0 but
                        less than the min value
        Returns:   
            points: collection of points that construct the straight line
                    segments
        '''
         # Combine x,y and sort
        data = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis = 1)
        data = data[data[:,0].argsort()]

        points = []      # points of principal curve
        points.append(data[0,:])     # Append first point
        pe = data[-1,:] # end point

        nc = True
        while nc:
            pt_found = False

            # Start drawing circle
            rl = 0 # lower radius bound
            rt = 2 * np.linalg.norm(pe - points[-1]) # upper bound

            while not pt_found:
                ri = rl + (rt - rl)/2

                # Get points inside circle
                in_c = points_in(data, ri, points[-1])

                # calculate pi+1 (p2)
                # Construct inner radius
                rj = ri * .9

                btw_c = points_btw(data, rj, ri, points[-1]) #get pts in circle

                
                if btw_c.shape[0] == 0:  # No points are in circle
                    if fmin_error:      # reset error limit
                        e_max = e_max / .99
                        data, points = reset(x,y)
                        break
                    else:
                        raise ValueError("e_max = %f is too small. Choose a " \
                                        "smaller e_max, or set fmin_error = " \
                                        "True ." % e_max)   
                else:
                    # candidate point is mean of points in circle sector
                    p2 = np.array([np.mean(btw_c[:,0]), np.mean(btw_c[:,1])])
                    e_i = distg(in_c, points[-1], p2) # calculate error

                if e_i >= e_max:
                    rt = ri
                else:
                    data = points_out(data, ri, points[-1])
                    points.append(p2)

                    # Check if all points have been calculated
                    if data.shape[0] == 0:
                        nc = False
                        break

                    pt_found = True
        
        if fmin_error:
            print('Min e_max = %f' % e_max)

        res_x = np.array([p[0] for p in points])
        res_y = np.array([p[1] for p in points])
        res = np.concatenate([res_x.reshape(-1,1), res_y.reshape(-1,1)], axis = 1)
        self.fit_points = res
        return res

    def calc_pc(self, x, y, e_max = .2, fmin_error = False):
        '''
        Calculates principal curve ticks
        Args: same as fit_points
        Returns: None
        '''
        res = self.fit_points(x, y, e_max, fmin_error)
        tck, u = scipy.interpolate.splprep(res.T, s = 0)
        self.spline_ticks = tck
        return

    def plot(self, ax = None):
        '''
        Plots the curve to a MPL axes object.
        Args:
            ax: option set of ax to plot to 
        '''
        if ax == None:
            fig, ax = plt.subplots()
        xy = scipy.interpolate.splev(np.linspace(0,1,100), self.spline_ticks)
        ax.plot(xy[0], xy[1], c = 'black')
        return

    def project(self, x, y):
        '''
        Projects points x,y to principal curve calculated by calc_pc
        Args:
            x: x-data to project
            y: y-data to project
        Returns:
            proj: projections of points onto curve between (0,1)
        '''
        # for each point min distance to curve
        data = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis = 1)

        proj = []
        for p in data:
            proj_dist = op.minimize(
                    proj_min,
                    x0 = [.5],
                    args = (self.spline_ticks, p),
                    method = 'Powell'
                    ).x
            proj.append(proj_dist)
        return proj

# Functions specific to the search alg, namely finding the 
# error line and the function we search with/minimize
def to_min_error(theta, pts, v1, r1):
    v2 = np.array([v1[0]+r1*np.cos(theta[0]), v1[1]+r1*np.sin(theta[0])])
    D1 = np.linalg.norm(v2 - pts, axis = 1)
    D2 = np.abs(np.cross(v2-v1, v1-pts)) / np.linalg.norm(v2-v1)

    error_t = [np.min([i,j]) for i,j in zip(D1, D2)]

    return sum(error_t)/len(error_t)

def error_line(theta, c, r1):
    p2 = np.array([c[0]+r1*np.cos(theta), c[1]+r1*np.sin(theta)])
    return p2

class CLPCS:
    def __init__(self):
        fit_points = []
        spline_ticks = None
    
    def fit_points(self, x, y, e_max = .2, rl = 0):
        '''
        Implements CLPC one dimensional search algorithm
        Args:
            x: x-data
            y: y-data
            e_max: Max allowed error. If not met, another point P will be added
                to the curve. Authors suggest 1/4 to 1/2 of measurement error
        Returns:   

        '''
        # Combine x,y and sort
        data = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis = 1)
        data = data[data[:,0].argsort()]

        points = []      # points of principal curve
        points.append(data[0,:])     # Append first point
        pe = data[-1,:] # end point

        nc = True
        while nc:
            pt_found = False

            # Start drawing circle
            rt = 2 * np.linalg.norm(pe - points[-1]) # upper bound

            while not pt_found:
                ri = rl + (rt - rl)/2

                # Get points inside circle
                in_c = points_in(data, ri, points[-1])
                
                if in_c.shape[0] == 0:  # No points are in circle
                    raise ValueError("e_max = %f is too small. Choose a " \
                                    "smaller e_max." % e_max)   
                    
                else:
                    # find min error
                    theta = op.minimize(
                        to_min_error,
                        x0 = [0],
                        args = (in_c, points[-1], ri),
                        method = 'Powell'
                        ).x
                    p2 = error_line(theta, points[-1], ri) 
                    e_i = distg(in_c, points[-1], p2) # calculate error
                
                if e_i >= e_max:
                    rt = ri
                else:
                    data = points_out(data, ri, points[-1])
                    points.append(p2)

                    # Check if all points have been calculated
                    if data.shape[0] == 0:
                        nc = False
                        break

                    pt_found = True
            
        res_x = np.array([p[0] for p in points])
        res_y = np.array([p[1] for p in points])
        res = np.concatenate([res_x.reshape(-1,1), res_y.reshape(-1,1)], axis = 1)
        return res

    def calc_pc(self, x, y, e_max = .2, rl = 0):
        '''
        Calculates principal curve ticks
        Args: same as fit_points
        Returns: None
        '''
        res = self.fit_points(x, y, e_max, rl)
        tck, u = scipy.interpolate.splprep(res.T, s = 0)
        self.spline_ticks = tck
        return

    def plot(self, ax = None):
        '''
        Plots the curve to a MPL axes object.
        Args:
            ax: option set of ax to plot to 
        '''
        if ax == None:
            fig, ax = plt.subplots()
        xy = scipy.interpolate.splev(np.linspace(0,1,100), self.spline_ticks)
        ax.plot(xy[0], xy[1], c = 'black')
        return

    def project(self, x, y):
        '''
        Projects points x,y to principal curve calculated by calc_pc
        Args:
            x: x-data to project
            y: y-data to project
        Returns:
            proj: projections of points onto curve between (0,1)
        '''
        # for each point min distance to curve
        data = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis = 1)

        proj = []
        for p in data:
            proj_dist = op.minimize(
                    proj_min,
                    x0 = [.5],
                    args = (self.spline_ticks, p),
                    method = 'Powell'
                    ).x
            proj.append(proj_dist)
        return proj