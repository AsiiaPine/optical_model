from typing import Tuple
import numpy as np

def divide_surface(surface_size: Tuple[int, int], n_rows, n_columns):
    '''
    The function divides the surface by n_row, n_columns and returns matrix of centers of the windows in form (x_i, y_i)
    '''
    R, C = surface_size
    r = R/n_columns
    c = C/n_rows

    # generator of coordinates of the windows centers
    r_array = np.arange(r/2, R, r)
    c_array = np.arange(c/2, C, c)

    x_matrix = np.repeat(r_array[np.newaxis, :], repeats=n_rows, axis=0)
    y_matrix = np.repeat(c_array[np.newaxis, :], repeats=n_columns, axis=0)

    return np.moveaxis(np.array([x_matrix, y_matrix.T]), 0, -1)

def make_blindness_matrix(surface_shape: Tuple[int, int], blind_shape):
    '''
    Returns a true map matrix of size=blind_shape with 0 values. The matrix is located in the center of blindness matrix.
    '''

    n_rows, n_columns = surface_shape
    blindness_matrix = np.ones(shape=(n_rows, n_columns))

    k, l = blind_shape
    start_row = (n_rows - k) // 2
    end_row = start_row + k
    start_col = (n_columns - l) // 2
    end_col = start_col + l

    # Matrix shows the Intensity of n LED sectors
    blindness_matrix[start_row:end_row, start_col:end_col] = 0
    return blindness_matrix

def calc_intensity(emitter_xy: [float, float],  receiver_xy: [float, float], l_0: float=3.5, I_0: float = 1.0):
    x_l, y_l = emitter_xy
    x_p, y_p = receiver_xy

    R_2 = (x_l - x_p)**2 + (y_l - y_p)**2 + l_0**2
    h = y_p - y_l
    h = h*np.sign(h)
    theta = np.arccos(h/np.sqrt(R_2))
    # from my calculations
    # return I_0*(np.sin(theta)**2)/(4*np.pi*R_2)
    
    # From the paper
    return I_0*(np.sin(theta))/(4*np.pi*R_2/(2*l_0))


def calc_intensity_matrix_to_point(pt_xy: [float, float], led_matrix, led_map=None, l_0: float=3.5, I_0: float = 1.0):
    if led_map is None:
        led_map = np.ones(shape=(led_matrix.shape[0], led_matrix.shape[1]))
        
    intensity = 0
    for i in range(led_matrix.shape[0]):
        for j in range(led_matrix.shape[1]):
            intensity+=calc_intensity(emitter_xy=led_matrix[i, j], receiver_xy=pt_xy) * led_map[i, j]
    return intensity


# def opts_
from scipy import integrate
import numpy as np


def calc_intensity_integration(emitter_x: float, emitter_y: float,  receiver_x: float, receiver_y: float, l_0: float=3.5, I_0: float = 1.0):
    x_l, y_l = emitter_x, emitter_y
    x_p, y_p = receiver_x, receiver_y

    R_2 = (x_l - x_p)**2 + (y_l - y_p)**2 + l_0**2
    h = y_p - y_l
    h = h*np.sign(h)
    theta = np.arccos(h/np.sqrt(R_2))
    # from my calculations
    # return I_0*(np.sin(theta)**2)/(4*np.pi*R_2)
    # From the paper
    return I_0*(np.sin(theta))/(4*np.pi*R_2/(2*l_0))

def calc_flux_on_area(source_matrix: np.ndarray, receiver_xy: tuple[float], dx:float, dy:float, led_shape: tuple[float]):
    x, y = receiver_xy
    S1, S2 = led_shape
    lims = [[-S1/2, S1/2], [-S2/2, S2/2], [x-dx/2, x+dx/2], [y-dy/2, y+dy/2]]

    # for i in range(source_matrix.shape[0]):
    #     for j in range(source_matrix.shape[1]):
    #         print(source_matrix[i, j], receiver_xy)
    return integrate.nquad(calc_intensity_integration, ranges=lims)
    