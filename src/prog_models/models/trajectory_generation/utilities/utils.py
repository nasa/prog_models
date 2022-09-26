# Utility functions for trajectory sim
from imports_ import np, stats, interp, dt, plt, animation, make_axes_locatable

# Functions
# ==========

def check_dist2bounds(val, bounds):
    n = len(val)
    assert len(bounds)==n, "check_dist2bounds: Number of values should match the number of bounds."
    are_close = [[False, False] for _ in range(n)]
    for ii in range(n): 
        are_close[ii][0] = bool(np.isclose(val[ii], bounds[ii][0]))
        if len(bounds[ii]) == 2:
            are_close[ii][1] = bool(np.isclose(val[ii], bounds[ii][1]))
    return are_close


def debugplot(x, t=None):
    plt.figure()
    if t is not None:   plt.plot(t, x)
    else:               plt.plot(x)
    plt.show()
    return

def from_relative_time_to_timesamp(t, datetime0):
    return [datetime0.timestamp() + t[item] for item in range(len(t))]

def from_relative_time_to_datetime(t, datetime0):
    timestamps = from_relative_time_to_timesamp(t, datetime0)
    return [dt.datetime.fromtimestamp(timestamps[item]) for item in range(len(timestamps))]

# Sampling-based transition from cartesian (x, y) to magnitude and direction
def xy2magdir(xm, ym, xs=None, ys=None, nsamps=1000):
    norm_rv = LHS(dist='normal')
    if (xs is not None and ys is not None) and (xs.sum()!=0 and ys.sum()!=0):
        xsamps    = norm_rv(ndims=1, nsamps=nsamps, loc=xm, scale=xs)
        ysamps    = norm_rv(ndims=1, nsamps=nsamps, loc=ym, scale=ys)
        mag_samps = np.sqrt(xsamps**2.0 + ysamps**2.0)
        dir_samps = np.arctan2(ysamps, xsamps) - 1.0/2.0*np.pi

        mag_m = np.mean(mag_samps, axis=0)
        mag_s = np.std(mag_samps, axis=0)
        dir_m = np.mean(dir_samps, axis=0)
        dir_s = np.std(dir_samps, axis=0)
        return mag_m, dir_m, mag_s, dir_s
    elif (xs is not None and ys is not None) and (not all(xs==np.zeros_like(xs)) and not all(ys==np.zeros_like(ys))):
        xsamps = norm_rv(ndims=1, nsamps=nsamps, loc=xm, scale=xs)
        ysamps = norm_rv(ndims=1, nsamps=nsamps, loc=ym, scale=ys)
        for samp in range(nsamps):
            xsamps[np.isnan(xsamps[samp, :])] = xm
            ysamps[np.isnan(ysamps[samp, :])] = ym
        
        mag_samps = np.sqrt(xsamps**2.0 + ysamps**2.0)
        dir_samps = np.arctan2(ysamps, xsamps) - 1.0/2.0*np.pi
        mag_m = np.mean(mag_samps, axis=0)
        mag_s = np.std(mag_samps, axis=0)
        dir_m = np.mean(dir_samps, axis=0)
        dir_s = np.std(dir_samps, axis=0)
        return mag_m, dir_m, mag_s, dir_s
    else:
        mag_m = np.hypot(xm, ym)
        dir_m = np.arctan2(ym, xm) - 1.0/2.0*np.pi
        return mag_m, dir_m, np.zeros_like(mag_m), np.zeros_like(dir_m)


def utc_to_local(utc_dt):
    return utc_dt.replace(tzinfo=dt.timezone.utc).astimezone(tz=None)


def gen_video(fig, ax, frames, **kwargs):
    params = dict(interval=500, blit=False, repeat_delay=500, 
                  xlabel='longitude, deg', ylabel='latitude, deg', fps=15, 
                  artist='Matteo Corbetta', fontsize=20, bitrate=1800, filename=None)
    params.update(kwargs)

    video = animation.ArtistAnimation(fig, frames, interval=params['interval'], blit=params['blit'], repeat_delay=params['repeat_delay'])
    ax.set_xlabel('longitude, deg', fontsize=14)
    ax.set_ylabel('latitude, deg', fontsize=14)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(frames[-1][0], cax=cax, orientation='vertical')
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='MatteoCorbetta'), bitrate=1800)
    if params['filename'] is None:  params['filename'] = 'video'
    video.save(params['filename'] + '.mp4', writer='ffmpeg')
    print('Video ' + params['filename'] + ' saved successfully.')


def gen_heatmaps_vs_time(u, v, t, longrid, latgrid, what='mag'):
    n = len(t)
    assert u.shape[0] == n, "time steps in t and u must coincide."
    assert v.shape[0] == n, "time steps in t and v must coincide."
    wind_heatmaps = []
    plot_lims = (min(longrid[0, :]), max(longrid[0, :]),
                 min(latgrid[:, 0]), max(latgrid[:, 0]))
    fig, ax = plt.subplots()
    for ii in range(n):
        if what=='mag':     values = np.sqrt(u[ii, :, :]**2.0 + v[ii,:,:]**2.0)
        elif what=='dir':   values = np.arctan2(u[ii,:,:], v[ii,:,:]) + 3.0/2.0 * np.pi
        timestamp = dt.datetime.utcfromtimestamp(t[ii]).strftime('%Y-%m-%d %H:%M:%S')
        ttl = plt.text(0.5, 1.01, timestamp, horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
        im = plt.imshow(values, origin='lower', aspect='auto', cmap=plt.get_cmap('jet'), extent=plot_lims)
        wind_heatmaps.append([im, ttl])  
    return fig, ax, wind_heatmaps

    
def compute_norm(x, m, s):
    if len(x.shape)==1:
        return (x-m)/s
    else:
        xnorm = np.zeros_like(x)
        for ci in range(x.shape[1]):
            xnorm[:, ci] = (x[:, ci] - m[ci]) / s[ci]
        return xnorm

def compute_mean(x):
    if len(x.shape)==1:     return np.mean(x)
    else:                   return np.mean(x, axis=0)

def compute_std(x):
    if len(x.shape)==1:     return np.std(x)
    else:                   return np.std(x, axis=0)

def normalize(x, m=None, s=None):
    if m is None:   m = compute_mean(x)
    if s is None:   s = compute_std(x)
    return compute_norm(x, m, s), m, s


def interpolate_2dgrid(X, n_interp, m_interp):
    _, m    = X.shape
    tmp     = np.zeros((n_interp, m))
    Xinterp = np.zeros((n_interp, m_interp))
    # First interpolate columns and then rows
    for col in range(m):                tmp[:, col]     = np.linspace(X[:, col][0], X[:, col][-1], n_interp)
    for row in range(n_interp):         Xinterp[row, :] = np.linspace(tmp[row, :][0], tmp[row, :][-1], m_interp)
    return Xinterp


def euler(f, h, x, u, params):
    return f(x, u, params)


def rk4(f, h, x, u, params):
    """
    Fourth-order Runge-Kutta method

    dxdt = rk4(f, h, x, u, params)

    Provide the value increment of variable x, dx/dt,
    evolving according to the dynamics:

        dx/dt = f(x)

    where f is dynamic function, x is the current value, and dt is the time-step size.

    Input:
    f               matlab function (anonymous function, script function, etc.) defining the system's dynamics
                    f must be able to receive n-dimensional vector x
    h               scalar, time step size
    x               n x 1 vector, independent variable
    u               m x 1 vector, input to the system
    params

    Output:
    dxdt            n x 1 vector, value increment of x in time dt.
    """
    # Compute intermediate points of integration
    k1 = f(x, u, params)               # Compute k1 -> k1 = f(x, u, params)
    k2 = f(x + h/2.0 * k1, u, params)  # Compute k2 -> k2 = f(x + h/2.0*k1, u, params);
    k3 = f(x + h/2.0 * k2, u, params)  # Compute k3 -> k3 = f(x + h/2.0*k2, u, params);
    k4 = f(x + h * k3, u, params)      # Compute k4 -> k4 = f(x + h*k3, u, params);
    return 1.0/6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)    # Compute dxdt: sum up individual contributions


# GPR utilities
# ==============

def interp_function(x, t, tq):
    return interp.interp1d(t, x, axis=-1)(tq)
    

def resample_params(params, bounds_, scale=0.1):
    p_list = list(params.keys())
    
    n = len(params)
    assert len(bounds_) == n, "resample_params: length of params and bounds must match."

    for p in p_list:
        sample_pi = params[p] + scale*params[p] * np.random.randn()
        sample_pi = max([bounds_[p][0], sample_pi])
        sample_pi = min([bounds_[p][1], sample_pi])
        params[p] = sample_pi
    return params


def init_hyperparams(X, y, gamma=0.5, delta=0.5):
    # Initialize GP hyperparameter search according to:
    # Ulapane et al. Hyper-Parameter Initialization for Squared Exponential Kernel-based Gaussian Process Regression
    _, m = X.shape
    dx = np.zeros((m,))
    dy = np.zeros((m,))
    for ii in range(m):
        S      = np.column_stack((X[:, ii], y))
        S      = S[np.argsort(S[:, 0])]
        dS     = np.diff(S, axis=0)
        dx[ii] = sum(abs(dS[:, 0])) / sum(dS[:, 0]!=0)
        dy[ii] = sum(abs(dS[:, 1])) / sum(dS[:, 1]!=0)
    ls    = np.mean(dx)
    amp   = max([np.mean(dy), 0.])
    noise = gamma * amp + delta * min(dy)
    return {'signal_amplitude': amp, 'lengthscale': ls, 'noise_var': noise}


# CLASSES
# ========

class ProgressBar():
    def __init__(self, n, prefix='', suffix='', decimals=1, print_length=100, fill='█', print_end = " ") -> None:
        self.n = n
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.print_length = print_length
        self.fill = fill
        self.print_end = print_end
        pass

    def __call__(self, iteration):
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (iteration / float(self.n)))
        filledLength = int(self.print_length * iteration // self.n)
        bar = self.fill * filledLength + '-' * (self.print_length - filledLength)
        print('\r%s |%s| %s%% %s' % (self.prefix, bar, percent, self.suffix), end = self.print_end)
        # Print New Line on Complete
        if iteration == self.n:
            print('')

            
class LHS():
    """
    Latin Hypercube Sampling Function

    Limitations:
    - independent dimensions
    - uniform and normal distributions only
    
    Example calls:
    - Generate 100 random variate samples in one dimension from a standard normal distribution
    
    norm_rv = LHS(dist='normal')
    r       = norm_rv(ndims=1, nsamps=100, loc=0, scale=1)
    
    - Generate 200 random variate samples in three dimensions from a uniform distribution with limits
    
    unif_rv = LHS(dist='uniform')
    u       = unif_rv(ndims=3, nsamps=200)
    """
    def __init__(self, dist='uniform') -> None:
        dist = dist.lower().replace(" ","").replace("_", "").replace("-","")
        if dist != 'uniform' and dist != 'normal':      raise Exception("Distribution type " + self.dist + " not recognized.")
        self.dist = dist
        pass

    def __call__(self, ndims=1, nsamps=1, **kwargs):
        norm_params = dict(loc=0, scale=1)
        norm_params.update(kwargs)
        if   self.dist == 'uniform':    return self.__uniform(ndims, nsamps)
        elif self.dist == 'normal':     return self.__normal(ndims, nsamps, norm_params['loc'], norm_params['scale'])
        else:                           return []
    
    def __normal(self, ndims=1, nsamps=1, loc=0.0, scale=1.0):
        if not hasattr(loc, '__len__'):     loc   = [loc, ] * ndims
        if not hasattr(scale, '__len__'):   scale = [scale, ] * ndims
        samples = self.__draw_samples(ndims=ndims, nsamps=nsamps, loc=loc, scale=scale)
        return self.shuffle_samples(samples)

    def __uniform(self, ndims=1, nsamps=1):
        samples = self.__draw_samples(ndims=ndims, nsamps=nsamps)
        return self.shuffle_samples(samples)

    def __draw_samples(self, ndims, nsamps, loc=0., scale=1.):
        u_low, u_high = self.__gen_unif_limits(nsamps)
        if self.dist=='uniform' or self.dist=='normal':
            samples = np.random.uniform(low=u_low, high=u_high, size=(ndims, nsamps)).T
            if self.dist == 'normal':       
                samples = stats.norm.ppf(samples, loc=loc, scale=scale)
        else:
            raise Exception("Distribution type " + self.dist + " not recognized (or not implemented yet).")
        return samples

    @staticmethod
    def __gen_unif_limits(nsamps):
        u = np.linspace(0.0, 1.0, nsamps+1)
        return u[:-1], u[1:]

    @staticmethod
    def shuffle_samples(samples):
        [np.random.shuffle(samples[:, ii]) for ii in range(samples.shape[1])]
        return samples

def resize_wind_grid(x, order='F'):
    grid_size = int(np.sqrt(x.shape[0]))
    if len(x.shape) == 1:
        return x.reshape((grid_size, grid_size), order=order)
    elif len(x.shape)==2:
        m = x.shape[-1]
        return x.reshape((grid_size, grid_size, m), order=order)

def compute_wind_magnitude(u, v):
    return np.sqrt(u**2.0 + v**2.0)


def trajectory_sample_generator(trajs, **kwargs):
    return [traj.gen_samples(**kwargs) for traj in trajs]

    
def is_segment_full(t):
    if t['lat']['start'].size + t['lat']['end'].size:   return True
    else:                                               return False


# VISUALIZE FUNCTIONS
# ==========================
def get_subplot_dim(num_subplots, rowfirst=True):
    """
    Compute the number of rows and columns (nrows, ncols) for a figure with multiple subplots.
    The function returns number of rows and columns given num_subplots. 
    Those numbers are computed sequentially until nrows * ncols >= num_subplots.
    By default, the function adds a new row first if the number of subplots has not been reached, then adds a new column.
    By passing rowfirst=False, the function will add a new column first if the number of subplot has not been reached, then a new row.

    nrows and ncols are initialized to 1. If num_subplots==1, then subplots are not needed, and the function returns nrows=ncols=1.
    The command fig.add_subplot(nrows,ncols,1) generates a normal plot (no subplots).

    Parameters
    ----------
    num_subplots : int
                   number of subplots the figure should contain
    rowfirst     : Boolean
                   whether to add a new row first or a new column first to increase the number of rows and columns if necessary.
                   Default is rowfirst=True.
    
    Returns
    -------
    nrows : int
            number of subplots along the rows (vertical axis) of the figure
    ncols : int
            number of subplots along the columns (horizontal axis) of the figure

    Example
    -------
    | states = np.random.randn(1000,5) # let us consider a state vector with 5 dimensions, and 1000 values of the states (one for each time step)
    | n_states = states.shape[-1]     # get the number of states (5)
    | print(get_subplot_dim(n_states)) # 3, 2
    | print(get_subplot_dim(n_states, rowfirst=False)) # 2, 3
    | 
    | fig = plt.figure()
    | ax = fig.add_subplot(nrows, ncols, 0)
    | # ...
    """
    nrows, ncols = 1, 1 # initialize number of rows and cols to 1.
    if rowfirst:
        while nrows * ncols < num_subplots:         
            nrows += 1
            if nrows * ncols < num_subplots:        
                ncols += 1
    else:
        while nrows * ncols < num_subplots:         
            ncols += 1
            if nrows * ncols < num_subplots:        
                nrows += 1
    return nrows, ncols



if __name__ == '__main__':

    import time

    x = np.linspace(0., 1.0, 100)
    wb = ProgressBar(n=len(x), prefix=' Nonsense computation .. ', suffix=' complete.')

    for ii in range(len(x)):
        wb(ii)
        # your model computation here
        time.sleep(0.2)
    wb(len(x))

