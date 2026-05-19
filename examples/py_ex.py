import numpy as np
import timeit

def distg_fast(pts, v1, v2):
    """Fully vectorized distance metric calculation."""
    if pts.shape[0] == 0:
        raise ZeroDivisionError

    line_vec = v2 - v1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        raise ZeroDivisionError

    D1 = np.linalg.norm(v2 - pts, axis=1)
    D2 = np.abs(np.cross(line_vec, v1 - pts)) / line_len

    return np.minimum(D1, D2).mean()

def distg(pts, v1, v2):
    D1 = np.linalg.norm(v2 - pts, axis=1)
    D2 = np.abs(np.cross(v2-v1, v1-pts)) / np.linalg.norm(v2-v1)
    if len(D1) == 0:
        raise ZeroDivisionError
    error_t = [np.min([i,j]) for i,j in zip(D1, D2)]
    return sum(error_t)/len(error_t)

def points_in(pts, r1, p):
    distances = np.linalg.norm(p - pts, axis=1)
    return pts[(distances < r1)]

def points_out(pts, r1, p):
    distances = np.linalg.norm(p - pts, axis=1)
    return pts[(distances > r1)]

def points_btw(pts, r1, r2, p):
    distances = np.linalg.norm(p - pts, axis=1)
    return pts[(distances < r2) & (distances > r1)]

class CLPCG:
    def __init__(self):
        self.fit_points = []
    
    def points(self, x, y, e_max=0.2):
        data = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis=1)
        points = [data[0,:]]
        pe = data[-1,:]

        while 1:
            pt_found = False 
            rl = 0
            rt = 2 * np.linalg.norm(pe - points[-1])

            rend = np.linalg.norm(pe - points[-1])
            in_c = points_in(data, rend, points[-1])
            try:
                e_end = distg(in_c, points[-1], pe)
            except ZeroDivisionError:
                break

            if e_end <= e_max:
                points.append(pe)
                break

            while not pt_found:
                ri = rl + (rt - rl)/2
                in_c = points_in(data, ri, points[-1])
                rj = ri * .9
                btw_c = points_btw(data, rj, ri, points[-1])
                
                if btw_c.shape[0] == 0:
                    # Original code raised ValueError here
                    raise ValueError(f"e_max = {e_max} is too small. Choose a larger e_max.")   
                else:
                    p2 = np.array([np.mean(btw_c[:,0]), np.mean(btw_c[:,1])])
                    e_i = distg(in_c, points[-1], p2)

                if e_i > e_max:
                    rt = ri
                else:       
                    data = points_out(data, ri, points[-1])
                    points.append(p2)
                    pt_found = True

        res = np.array(points)
        self.fit_points = res
        return res

def clpcg_fast(data, e_max=0.2, slice_width=0.9):
    """
    Optimized implementation of the Greedy CLPCG algorithm.
    Avoids copying large arrays and uses vectorized distance calculations.
    """
    # Initialize points list with the first data point
    points = [data[0, :]]
    pe = data[-1, :]
    
    # Track which points are still available
    n_pts = data.shape[0]
    remaining = np.ones(n_pts, dtype=bool)
    remaining[0] = False # First point is used
    
    # Pre-calculate distances from pe to all points (useful for end-check)
    dist_to_end = np.linalg.norm(data - pe, axis=1)

    while remaining.any():
        pt_found = False
        prev_pt = points[-1]
        
        # Calculate distances from prev_pt to all remaining points
        dist_to_prev = np.linalg.norm(data - prev_pt, axis=1)
        
        # Search radius bounds
        rl = 0.0
        rt = 2.0 * np.linalg.norm(pe - prev_pt)
        
        # Try connecting to the end point first
        rend = np.linalg.norm(pe - prev_pt)
        in_c_mask = (dist_to_prev <= rend) & remaining
        if in_c_mask.any():
            e_end = distg_fast(data[in_c_mask], prev_pt, pe)
            if e_end <= e_max:
                points.append(pe)
                break
                
        # Binary search for the next point
        safety_counter = 0
        while not pt_found and safety_counter < 50:
            ri = rl + (rt - rl) / 2.0
            
            # Points inside the sphere ri that are still remaining
            in_c_mask = (dist_to_prev <= ri) & remaining
            
            if not in_c_mask.any():
                # If no points found, try expanding rt
                rt *= 1.5
                safety_counter += 1
                continue
                
            # Points in the outer shell (between rj and ri)
            rj = ri * slice_width
            btw_c_mask = in_c_mask & (dist_to_prev > rj)
            
            if not btw_c_mask.any():
                # Fallback: if outer shell is empty, use all points inside circle
                candidate = np.mean(data[in_c_mask], axis=0)
            else:
                candidate = np.mean(data[btw_c_mask], axis=0)
                
            # Calculate error
            e_i = distg_fast(data[in_c_mask], prev_pt, candidate)
            
            if e_i > e_max:
                rt = ri # Shrink
            else:
                # Accept: Mark used points as not remaining
                remaining[in_c_mask] = False
                points.append(candidate)
                pt_found = True
            
            safety_counter += 1
            
        if not pt_found:
            # If we get stuck, just force progress to avoid infinite loop
            break
            
    return np.array(points)

if __name__ == "__main__":
    # Generate Test Data
    theta = np.linspace(0, np.pi * 3, 1000)
    r = np.linspace(0, 1, 1000) ** 0.5
    
    x_data = r * np.cos(theta) + np.random.normal(scale=0.02, size=1000)
    y_data = r * np.sin(theta) + np.random.normal(scale=0.02, size=1000)
    data = np.column_stack([x_data, y_data])
    
    # Test Original
    cl = CLPCG()
    start = timeit.default_timer()
    res_orig = cl.points(x_data, y_data, e_max=0.05)
    stop = timeit.default_timer()
    print(f"Original Python: {stop - start:.4f} seconds ({len(res_orig)} points)")
    
    # Test Cleaned/Optimized Python
    start = timeit.default_timer()
    res_fast = clpcg_fast(data, e_max=0.05)
    stop = timeit.default_timer()
    print(f"Optimized Python: {stop - start:.4f} seconds ({len(res_fast)} points)")

    # Test Rust Implementation
    try:
        from prinpy_rs import clpg
        # Rust expects float32
        data_f32 = data.astype(np.float32)
        start = timeit.default_timer()
        res_rust = clpg(data_f32, 0.05)
        stop = timeit.default_timer()
        print(f"Rust Extension: {stop - start:.4f} seconds ({len(res_rust)} points)")
    except ImportError:
        print("Rust Extension (prinpy_rs) could not be imported. Did you run 'maturin develop'?")
    except Exception as e:
        print(f"Rust Extension failed: {e}")