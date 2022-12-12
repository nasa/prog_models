# Aerodynamics effects


import numpy as np
import scipy.io as io
import scipy.interpolate as interp
import pandas as pd

import h5py


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import prog_models.models.uav_model.utilities.geometry as geom

# AUXILIARY CONVERSIONS
# =======================
ftps2mps = 0.3048   # feet per second to meters per second
lb2kg = 0.453592        # pound to kg
lbpftsq2kgpmsq = 4.88243    # pound per square feet to kg per square meter
slugpftcub2kgpmcub = 515.379    # slug per cubic foot to kg per cubic meter
inchplb2nm = 0.112984825        # inch per pound to newtown meters


# FUNCTIONS
# =======================
def read_csv_aerotest_file(filename=None):
    if filename is None:
        filename = 'data/Aerodynamics/3DRSOLO_FullVehicle_Table_1.csv'
    
    d          = pd.read_csv(filename)
    n_rotors   = 4
    directions = ['x', 'y', 'z']
    rotors     = [str(ii) for ii in range(1, n_rotors+1)]            

    F   = dict.fromkeys(directions)
    M   = dict.fromkeys(directions)
    UF  = dict.fromkeys(directions)
    UM  = dict.fromkeys(directions)
    UI  = dict.fromkeys(rotors)
    I   = dict.fromkeys(rotors)
    RPM = dict.fromkeys(rotors)
    
    q       = d[r'q (lb/ft\textsuperscript{2})'].values * lbpftsq2kgpmsq
    density = d[r'density (slug/ft\textsuperscript{3})'].values * slugpftcub2kgpmcub
    s       = d['Speed (ft/s)'].values * ftps2mps

    yaw   = d['Yaw (deg)'].values * np.pi/180.0
    pitch = d['Pitch (deg)'].values * np.pi/180.0
    for rotor in rotors:
        RPM[rotor] = d['RPM ' + rotor].values * 2.0*np.pi/60.0
        I[rotor]   = d['I' + rotor + ' (A)'].values
        UI[rotor]  = d['U-I'+rotor +' [A]'].values
    
    for direction in directions:
        F[direction]  = d['F'+direction+' (lb)'].values * lb2kg    
        UF[direction] = d['U-F'+direction+' [lb]'].values * lb2kg
        M[direction]  = d['M'+direction+' (in-lb)'].values * inchplb2nm
        UM[direction] = d['U-M'+direction+' [in-lb]'].values * inchplb2nm

    Vesc  = d['Vesc (V)'].values   # voltage
    UVesc = d['U-Vesc [V]'].values   # voltage

    drag        = d['D [lb]'].values * lb2kg
    rpm_all     = np.column_stack((RPM['1'], RPM['2'], RPM['3'], RPM['4']))
    current_all = np.column_stack((I['1'], I['2'], I['3'], I['4'], UI['1'], UI['2'], UI['3'], UI['4']))
    forces_all  = np.column_stack((F['x'], F['y'], F['z'], UF['x'], UF['y'], UF['z']))
    moments_all = np.column_stack((M['x'], M['y'], M['z'], UM['x'], UM['y'], UM['z']))
    data        = np.column_stack((s, q, density, pitch, yaw, Vesc, UVesc, rpm_all, current_all, forces_all, moments_all, drag))
    variables   = ['Speed (ft/s)', 'q (lb/ft2)', 'density (slug/ft3)', 'Pitch (deg)', 'Yaw (deg)', 'Vesc (V)', 'U-Vesc (V)', 
                   'RPM 1 (rad/s)', 'RPM 2 (rad/s)', 'RPM 3 (rad/s)', 'RPM 4 (rad/s)', 'I 1 (A)', 'I 2 (A)', 'I 3 (A)', 'I 4 (A)', 'U-I 1 (A)', 'U-I 2 (A)', 'U-I 3 (A)', 'U-I 4 (A)',
                   'Fx (lb)', 'Fy (lb)', 'Fz (lb)', 'U-Fx (lb)', 'U-Fy (lb)', 'U-Fz (lb)', 'Mx (in-lb)', 'My (in-lb)', 'Mz (in-lb)', 'U-Mx (in-lb)', 'U-My (in-lb)', 'U-Mz (in-lb)', 'D (lb)']
    
    return pd.DataFrame(data=data, index=[ii for ii in range(drag.shape[0])], columns=variables)


# Aerodynamic forces
# ====================
class DragModel():
    def __init__(self, bodyarea=None, Cd=None, air_density=None):
        self.area = bodyarea
        self.cd = Cd
        self.rho = air_density
        return

    def __call__(self, air_v):
        vsq = air_v * np.abs(air_v)
        return 0.5 * self.rho * self.area * self.cd * vsq
 



# def drag_force_body_frame(rho, a, cd, v):
#     vsq = v * np.abs(v)
#     d   = 0.5 * rho * a * cd * vsq
#     # D = - 1.0 / mass * np.dot(rotMat_body2earth(phi, theta, psi), np.dot(vel_bodyFrame_square, dragCoeff))    # N, drag forces in x, y, z
#     return d


# def drag_force_earth_frame(d_body, phi, theta, psi):
#     return np.dot( geom.rot_body2earth(phi, theta, psi), d_body)
    

"""
if __name__ == '__main__':
    
    print('Aerodynamic models')
    df = read_csv_aerotest_file()
    
    sns.pairplot(data=df)


    # plt.figure(),
    # plt.plot(s, F['x'], '.', label=r'$F_x$')
    # plt.plot(s, F['y'], '.', label=r'$F_y$')
    # plt.plot(s, F['z'], '.', label=r'$F_z$')
    # plt.xlabel('airspeed, m/s', fontsize=14)
    # plt.ylabel('force, kg', fontsize=14)
    # plt.legend(fontsize=14)



    plt.show()
"""