import math
import numpy as np
import matplotlib.pyplot 
import multiprocessing as mp
import pickle

from Tesseroid import Tesseroid


def cal_delta_x(phi_cal, lambda_cal, r_tess, phi_tess, lambda_tess):
    """
    Calculate delta_x in kernel function.
        \Delta_x = r' \left[\cos\varphi\sin\varphi' 
            - \sin\varphi\cos\varphi'\cos(\lambda' - \lambda)\right] 

    Parameters
    ----------
    phi_cal: float
        Latitude of computation point in radian.
    lambda_cal: float
        Longitude of computation point in radian.
    r_tess: float
        Radius of integration point in meter.
    phi_tess: float
        Latitude of integration point in radian.
    lambda_tess: float
        Longitude of integration point in radian.

    Returns
    -------
    delta_x: float
        delta_x in kernel function.
    """
    delta_x = r_tess * (math.cos(phi_cal) * math.sin(phi_tess) \
        - math.sin(phi_cal) * math.cos(phi_tess) \
        * math.cos(lambda_tess - lambda_cal)) 
    return delta_x


def cal_delta_y(lambda_cal, r_tess, phi_tess, lambda_tess):
    """
    Calculate delta_y in kernel function.
        \Delta_y = r' \cos \varphi' \sin(\lambda' - \lambda) 

    Parameters
    ----------    
    lambda_cal: float
        Longitude of computation point in radian.
    r_tess: float
        Radius of integration point in meter.
    phi_tess: float
        Latitude of integration point in radian.
    lambda_tess: float
        Longitude of integration point in radian.

    Returns
    -------
    delta_y: float
        delta_y in kernel function.
    """
    delta_y = r_tess * math.cos(phi_tess) * math.sin(lambda_tess - lambda_cal)

    return delta_y


def cal_delta_z(r_cal, phi_cal, lambda_cal, r_tess, phi_tess, lambda_tess):
    """
    Calculate delta_z in kernel function.
        \Delta_z = r' \cos \psi - r     
    
    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: float
        Latitude of computation point in radian.
    lambda_cal: float
        Longitude of compuattion point in radian.
    r_tess: float
        Radius of integration point in meter.
    phi_tess: float
        Latitude of integration point in radian.
    lambda_tess: float
        Longitude of integration point  in radian.

    Returns
    -------
    delta_z: float
        delta_z in kernel function.
    """
    cosPsi = math.sin(phi_cal) * math.sin(phi_tess) \
        + math.cos(phi_cal) * math.cos(phi_tess) \
        * math.cos(lambda_cal - lambda_tess)
    delta_z = (r_tess * cosPsi - r_cal)

    return delta_z


def cal_distance(r1, phi1, lambda1, r2, phi2, lambda2):
    """
    Calculate the distance in sphereical coordinate system.

    Parameters
    ----------
    r1: float
        Radius of point 1 in meter.
    phi1: float
        Latitude of point 1 in radian.
    lambda1: float
        Longitude of point 1 in radian.
    r2: float
        Radius of point 2 in meter.
    phi2: float
        Latitude of point 2 in radian.
    lambda2: float
        Longitude of point 2 in radian.

    Returns
    -------
    distance: float
        Distance between point 1 and point 2.
    """
    cosPsi = math.sin(phi1) * math.sin(phi2) \
        + math.cos(phi1) * math.cos(phi2) \
        * math.cos(lambda1 - lambda2)
    distance = math.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * cosPsi)

    return distance


def cal_V_kernel(r_cal, phi_cal, lambda_cal, \
    r_tess, phi_tess, lambda_tess, is_linear_density=False):
    """
    Calculate the kernel of gravitational potential.

    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: float
        Latitude of computation point in radian.
    lambda_cal: float
        Longitude of compuattion point in radian.
    r_tess: float
        Radius of integration point in meter.
    phi_tess: float
        Latitude of integration point in radian.
    lambda_tess: float
        Longitude of integration point in radian.
    is_linear_density: bool
        If the tesseroid have linear varying density.

    Returns
    -------
    kernel: float
        Kernel of gravitational potential.
    """
    ell = cal_distance(r_cal, phi_cal, lambda_cal,
        r_tess, phi_tess, lambda_tess)
    kernel = 1 / ell * r_tess**2 * math.cos(phi_tess)
    if is_linear_density:
        kernel *= r_tess
    
    return kernel


def cal_Vx_kernel(r_cal, phi_cal, lambda_cal, \
    r_tess, phi_tess, lambda_tess, is_linear_density=False):
    """
    Calculate the kernel of gravitational accelation Vx.

    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: float
        Latitude of computation point in radian.
    lambda_cal: float
        Longitude of compuattion point in radian.
    r_tess: float
        Radius of integration point in meter.
    phi_tess: float
        Latitude of integration point in radian.
    lambda_tess: float
        Longitude of integration point in radian.
    is_linear_density: bool
        If the tesseroid have linear varying density.

    Returns
    -------
    kernel: float
        Kernel of gravitational accelation Vx.
    """
    ell = cal_distance(r_cal, phi_cal, lambda_cal,
        r_tess, phi_tess, lambda_tess)
    delta_x = cal_delta_x(phi_cal, lambda_cal, r_tess, phi_tess, lambda_tess)
    kernel = delta_x / ell * r_tess**2 * math.cos(phi_tess)  
    if is_linear_density:
        kernel *= r_tess

    return kernel


def cal_Vy_kernel(r_cal, phi_cal, lambda_cal, \
    r_tess, phi_tess, lambda_tess, is_linear_density=False):
    """
    Calculate the kernel of gravitational accelation Vy.

    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: float
        Latitude of computation point in radian.
    lambda_cal: float
        Longitude of compuattion point in radian.
    r_tess: float
        Radius of integration point in meter.
    phi_tess: float
        Latitude of integration point in radian.
    lambda_tess: float
        Longitude of integration point in radian.
    is_linear_density: bool
        If the tesseroid have linear varying density.

    Returns
    -------
    kernel: float
        Kernel of gravitational accelation Vy.
    """
    ell = cal_distance(r_cal, phi_cal, lambda_cal,
        r_tess, phi_tess, lambda_tess)
    delta_y = cal_delta_y(lambda_cal, r_tess, phi_tess, lambda_tess)
    kernel = delta_y / ell * r_tess**2 * math.cos(phi_tess)
    if is_linear_density:
        kernel *= r_tess
    
    return kernel


def cal_Vz_kernel(r_cal, phi_cal, lambda_cal, \
    r_tess, phi_tess, lambda_tess, is_linear_density=False):
    """
    Calculate the kernel of gravitational accelation Vz.

    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: float
        Latitude of computation point in radian.
    lambda_cal: float
        Longitude of compuattion point in radian.
    r_tess: float
        Radius of integration point in meter.
    phi_tess: float
        Latitude of integration point in radian.
    lambda_tess: float
        Longitude of integration point in radian.
    is_linear_density: bool
        If the tesseroid have linear varying density.

    Returns
    -------
    kernel: float
        Kernel of gravitational accelation Vz.
    """
    ell = cal_distance(r_cal, phi_cal, lambda_cal,
        r_tess, phi_tess, lambda_tess)
    delta_z = cal_delta_z(r_cal, phi_cal, lambda_cal, 
        r_tess, phi_tess, lambda_tess)
    kernel = delta_z / ell**3 * r_tess**2 * math.cos(phi_tess)
    if is_linear_density:
        kernel *= r_tess

    return kernel


def cal_Vxx_kernel(r_cal, phi_cal, lambda_cal, \
    r_tess, phi_tess, lambda_tess, is_linear_density=False):
    """
    Calculate the kernel of gravitational gradient Vxx.

    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: float
        Latitude of computation point in radian.
    lambda_cal: float
        Longitude of compuattion point in radian.
    r_tess: float
        Radius of integration point in meter.
    phi_tess: float
        Latitude of integration point in radian.
    lambda_tess: float
        Longitude of integration point in radian.
    is_linear_density: bool
        If the tesseroid have linear varying density.

    Returns
    -------
    kernel: float
        Kernel of gravitational gradient Vxx.
    """
    
    ell = cal_distance(r_cal, phi_cal, lambda_cal,
        r_tess, phi_tess, lambda_tess)
    delta_x = cal_delta_x(phi_cal, lambda_cal, r_tess, phi_tess, lambda_tess)
    kernel = ((3 * (delta_x)**2 / (ell)**5) \
        - (1 / (ell)**3)) \
        * (r_tess**2 * math.cos(phi_tess))
    if is_linear_density:
        kernel *= r_tess
    
    return kernel


def cal_Vxy_kernel(r_cal, phi_cal, lambda_cal, \
    r_tess, phi_tess, lambda_tess, is_linear_density=False):
    """
    Calculate the kernel of gravitational gradient Vxy.

    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: float
        Latitude of computation point in radian.
    lambda_cal: float
        Longitude of compuattion point in radian.
    r_tess: float
        Radius of integration point in meter.
    phi_tess: float
        Latitude of integration point in radian.
    lambda_tess: float
        Longitude of integration point in radian.
    is_linear_density: bool
        If the tesseroid have linear varying density.

    Returns
    -------
    kernel: float
        Kernel of gravitational gradient Vxy.
    """
    ell = cal_distance(r_cal, phi_cal, lambda_cal,
        r_tess, phi_tess, lambda_tess)
    delta_x = cal_delta_x(phi_cal, lambda_cal, r_tess, phi_tess, lambda_tess)
    delta_y = cal_delta_y(lambda_cal, r_tess, phi_tess, lambda_tess)
    kernel = ((3 * (delta_x * delta_y) / (ell)**5)) \
        * (r_tess**2 * math.cos(phi_tess))
    if is_linear_density:
        kernel *= r_tess
    
    return kernel


def cal_Vxz_kernel(r_cal, phi_cal, lambda_cal, \
    r_tess, phi_tess, lambda_tess, is_linear_density=False):
    """
    Calculate the kernel of gravitational gradient Vxz.

    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: float
        Latitude of computation point in radian.
    lambda_cal: float
        Longitude of compuattion point in radian.
    r_tess: float
        Radius of integration point in meter.
    phi_tess: float
        Latitude of integration point in radian.
    lambda_tess: float
        Longitude of integration point in radian.
    is_linear_density: bool
        If the tesseroid have linear varying density.

    Returns
    -------
    kernel: float
        Kernel of gravitational gradient Vxz.
    """
    ell = cal_distance(r_cal, phi_cal, lambda_cal,
        r_tess, phi_tess, lambda_tess)
    delta_x = cal_delta_x(phi_cal, lambda_cal, r_tess, phi_tess, lambda_tess)
    delta_z = cal_delta_z(r_cal, phi_cal, lambda_cal, 
        r_tess, phi_tess, lambda_tess)
    kernel = ((3 * delta_x * delta_z / (ell)**5)) \
        * (r_tess**2 * math.cos(phi_tess))
    if is_linear_density:
        kernel *= r_tess
    
    return kernel


def cal_Vyy_kernel(r_cal, phi_cal, lambda_cal, \
    r_tess, phi_tess, lambda_tess, is_linear_density=False):
    """
    Calculate the kernel of gravitational gradient Vyy.

    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: float
        Latitude of computation point in radian.
    lambda_cal: float
        Longitude of compuattion point in radian.
    r_tess: float
        Radius of integration point in meter.
    phi_tess: float
        Latitude of integration point in radian.
    lambda_tess: float
        Longitude of integration point in radian.
    is_linear_density: bool
        If the tesseroid have linear varying density.

    Returns
    -------
    kernel: float
        Kernel of gravitational gradient Vyy.
    """
    ell = cal_distance(r_cal, phi_cal, lambda_cal,
        r_tess, phi_tess, lambda_tess)
    delta_y = cal_delta_y(lambda_cal, r_tess, phi_tess, lambda_tess)
    kernel = ((3 * (delta_y)**2 / (ell)**5) \
        - (1 / (ell)**3)) \
        * (r_tess**2 * math.cos(phi_tess))
    if is_linear_density:
        kernel *= r_tess
    
    return kernel


def cal_Vyz_kernel(r_cal, phi_cal, lambda_cal, \
    r_tess, phi_tess, lambda_tess, is_linear_density=False):
    """
    Calculate the kernel of gravitational gradient Vyz.

    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: float
        Latitude of computation point in radian.
    lambda_cal: float
        Longitude of compuattion point in radian.
    r_tess: float
        Radius of integration point in meter.
    phi_tess: float
        Latitude of integration point in radian.
    lambda_tess: float
        Longitude of integration point in radian.
    is_linear_density: bool
        If the tesseroid have linear varying density.

    Returns
    -------
    kernel: float
        Kernel of gravitational gradient Vyz.
    """
    ell = cal_distance(r_cal, phi_cal, lambda_cal,
        r_tess, phi_tess, lambda_tess)
    delta_y = cal_delta_y(lambda_cal, r_tess, phi_tess, lambda_tess)
    delta_z = cal_delta_z(r_cal, phi_cal, lambda_cal, 
        r_tess, phi_tess, lambda_tess)
    kernel = ((3 * delta_y * delta_z / (ell)**5)) \
        * (r_tess**2 * math.cos(phi_tess))
    if is_linear_density:
        kernel *= r_tess
    
    return kernel


def cal_Vzz_kernel(r_cal, phi_cal, lambda_cal, \
    r_tess, phi_tess, lambda_tess, is_linear_density=False):
    """
    Calculate the kernel of gravitational gradient Vzz.

    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: float
        Latitude of computation point in radian.
    lambda_cal: float
        Longitude of compuattion point in radian.
    r_tess: float
        Radius of integration point in meter.
    phi_tess: float
        Latitude of integration point in radian.
    lambda_tess: float
        Longitude of integration point in radian.
    is_linear_density: bool
        If the tesseroid have linear varying density.

    Returns
    -------
    kernel: float
        Kernel of gravitational gradient Vzz.
    """
    ell = cal_distance(r_cal, phi_cal, lambda_cal,
        r_tess, phi_tess, lambda_tess)
    delta_z = cal_delta_z(r_cal, phi_cal, lambda_cal, 
        r_tess, phi_tess, lambda_tess)
    kernel = ((3 * (delta_z)**2 / (ell)**5) \
        - (1 / (ell)**3)) \
        * (r_tess**2 * math.cos(phi_tess))
    if is_linear_density:
        kernel *= r_tess
      
    return kernel


def cal_kernel(r_cal, phi_cal, lambda_cal, \
    r_tess, phi_tess, lambda_tess, tag, is_linear_density):
    """
    Calculate the kernel of gravitational field.

    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: float
        Latitude of computation point in radian.
    lambda_cal: float
        Longitude of compuattion point in radian.
    r_tess: float
        Radius of integration point in meter.
    phi_tess: float
        Latitude of integration point in radian.
    lambda_tess: float
        Longitude of integration point in radian.
    tag: string
        Kernel function to be calculated. 
        tag \in {'V', 'Vx', 'Vy', 'Vz', 
        'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz'}
    is_linear_density: bool
        If the tesseroid have linear varying density.

    Returns
    -------
    kernel: float
        Kernel of gravitational field.
    """
    if tag=='V':
        kernel = cal_V_kernel(r_cal, phi_cal, lambda_cal, \
            r_tess, phi_tess, lambda_tess, is_linear_density)             
    elif tag=='Vx':
        kernel = cal_Vx_kernel(r_cal, phi_cal, lambda_cal, \
            r_tess, phi_tess, lambda_tess, is_linear_density)                   
    elif tag=='Vy':
        kernel = cal_Vy_kernel(r_cal, phi_cal, lambda_cal, \
            r_tess, phi_tess, lambda_tess, is_linear_density)                     
    elif tag=='Vz':
        kernel = cal_Vz_kernel(r_cal, phi_cal, lambda_cal, \
            r_tess, phi_tess, lambda_tess, is_linear_density)                   
    elif tag=='Vxx':
        kernel = cal_Vxx_kernel(r_cal, phi_cal, lambda_cal, \
            r_tess, phi_tess, lambda_tess, is_linear_density)                   
    elif tag=='Vxy':
        kernel = cal_Vxy_kernel(r_cal, phi_cal, lambda_cal, \
            r_tess, phi_tess, lambda_tess, is_linear_density)                   
    elif tag=='Vxz':
        kernel = cal_Vxz_kernel(r_cal, phi_cal, lambda_cal, \
            r_tess, phi_tess, lambda_tess, is_linear_density)                    
    elif tag=='Vyy':
        kernel = cal_Vyy_kernel(r_cal, phi_cal, lambda_cal, \
            r_tess, phi_tess, lambda_tess, is_linear_density)                   
    elif tag=='Vyz':
        kernel = cal_Vyz_kernel(r_cal, phi_cal, lambda_cal, \
            r_tess, phi_tess, lambda_tess, is_linear_density) 
    else:
        kernel = cal_Vzz_kernel(r_cal, phi_cal, lambda_cal, \
            r_tess, phi_tess, lambda_tess, is_linear_density) 
    return kernel


def subdivision(r_cal, phi_cal, lambda_cal, 
    r_min, r_max, phi_min, phi_max, lambda_min, lambda_max,
    ratio, density, density_gradient, max_node, tag, is_linear_density):
    """
    Subdivide the tesseroid.

    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: float
        Latitude of computation point in radian.
    lambda_cal: float
        Longitude of compuattion point in radian.
    r_min: float
        Min radius of tesseroid in meter.
    r_max: float
        Max radius of tesseroid in meter.
    phi_min: float
        Min latitude of tesseroid in radian.
    phi_max: float
        Max latitude of tesseroid in radian.
    lambda_min: float
        Min longitude of tesseroid in radian.
    lambda_max: float
        Max longitude of tesseroid in radian.   
    ratio: int
        Distance-size ratio, which is specified by the user. 
        The larger ratio is, the smaller tesseroid is divided, 
        and the higher accuracy of calculation is.
    density: float
        Density of tesseroid, unit: kg/m^3.
    density_gradient: float
        Density gradient of tesseroid, unit: kg/m^3/m.
    max_node: int
        Max node in $r$ direction, $\phi$ direction, 
        and $\lambda$ direction. In this program, 
        the nodes in radial, latitude, and longitude direction 
        take the same value.
    tag: string
        Kernel function to be calculated. 
        tag \in {'V', 'Vx', 'Vy', 'Vz', 
        'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz'}
    is_linear_density: bool
        If the tesseroid have linearly varying density.

    Returns
    -------
    result: float
        The gravitational field generated by tesseroid after subdivide.
    """
    result = 0
    phi_num = 2
    delta_phi = (phi_max - phi_min) / phi_num

    for phi_index in range(phi_num):
        phi_min_temp = phi_min + phi_index * delta_phi
        phi_max_temp = phi_min_temp + delta_phi
       
        lambda_num = 2
        delta_lambda = (lambda_max - lambda_min) / lambda_num
        for lambda_index in range(lambda_num):
            lambda_min_temp = lambda_min + delta_lambda * lambda_index
            lambda_max_temp = lambda_min_temp + delta_lambda

            result += cal_single2single_gravitational_field(r_cal, phi_cal, lambda_cal, 
                r_min, r_max, phi_min_temp, phi_max_temp, 
                lambda_min_temp, lambda_max_temp, 
                density, density_gradient, max_node, tag, ratio, is_linear_density)
    return result


def direct_cal_gravitational_field(r_cal, phi_cal, lambda_cal, 
    r_min, r_max, phi_min, phi_max, lambda_min, lambda_max, 
    density, density_gradient, max_node, tag, is_linear_density):
    """
    Directly calculate the gravitational field of tesseroid.

    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: float
        Latitude of computation point in radian.
    lambda_cal: float
        Longitude of compuattion point in radian.
    r_min: float
        Min radius of tesseroid in meter.
    r_max: float
        Max radius of tesseroid in meter.
    phi_min: float
        Min latitude of tesseroid in radian.
    phi_max: float
        Max latitude of tesseroid in radian.
    lambda_min: float
        Min longitude of tesseroid in radian.
    lambda_max: float
        Max longitude of tesseroid in radian.   
    ratio: int
        Distance-size ratio, which is specified by the user. 
        The larger ratio is, the smaller tesseroid is divided, 
        and the higher accuracy of calculation is.
    density: float
        Density of tesseroid, unit: kg/m^3.
    density_gradient: float
        Density gradient of tesseroid, unit: kg/m^3/m.
    max_node: int
        Max node in $r$ direction, $\phi$ direction, 
        and $\lambda$ direction. In this program, 
        the nodes in radial, latitude, and longitude direction 
        take the same value.
    tag: string
        Kernel function to be calculated. 
        tag \in {'V', 'Vx', 'Vy', 'Vz', 
        'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz'}
    is_linear_density: bool
        If the tesseroid have linearly varying density.

    Returns
    -------
    result: float
        Gravitational field generated by tesseroid.
    """
    result_constant = 0
    result_linear = 0
    roots, weights = np.polynomial.legendre.leggauss(max_node)
    if is_linear_density:
        for index_r in range(max_node):
            for index_phi in range(max_node):
                for index_lambda in range(max_node):
                    r_tess_temp = (roots[index_r] * (r_max - r_min) \
                        + r_max + r_min) / 2
                    phi_tess_temp = (roots[index_phi] * (phi_max - phi_min) \
                        + phi_max + phi_min) / 2
                    lambda_tess_temp = (roots[index_lambda] \
                        * (lambda_max - lambda_min) + lambda_max + lambda_min) / 2
                    result_constant += cal_kernel(r_cal, phi_cal, lambda_cal, \
                        r_tess_temp, phi_tess_temp, lambda_tess_temp, tag, is_linear_density=False) \
                        * weights[index_r] * weights[index_phi] * weights[index_lambda]
                    result_linear += cal_kernel(r_cal, phi_cal, lambda_cal, \
                        r_tess_temp, phi_tess_temp, lambda_tess_temp, tag, is_linear_density=True) \
                        * weights[index_r] * weights[index_phi] * weights[index_lambda]
    else:
        for index_r in range(max_node):
            for index_phi in range(max_node):
                for index_lambda in range(max_node):
                    r_tess_temp = (roots[index_r] * (r_max - r_min) \
                        + r_max + r_min) / 2
                    phi_tess_temp = (roots[index_phi] * (phi_max - phi_min) \
                        + phi_max + phi_min) / 2
                    lambda_tess_temp = (roots[index_lambda] \
                        * (lambda_max - lambda_min) + lambda_max + lambda_min) / 2
                    result_constant += cal_kernel(r_cal, phi_cal, lambda_cal, \
                        r_tess_temp, phi_tess_temp, lambda_tess_temp, tag, is_linear_density) \
                        * weights[index_r] * weights[index_phi] * weights[index_lambda]
    
    G =6.67384e-11
    result_constant *= G * (r_max - r_min) * (phi_max - phi_min) \
        * (lambda_max - lambda_min) / 8 * density
    if is_linear_density:
        result_linear *= G * (r_max - r_min) * (phi_max - phi_min) \
            * (lambda_max - lambda_min) / 8 * density_gradient
        return result_constant, result_linear
    else:
        return result_constant


def cal_single2single_gravitational_field(r_cal, phi_cal, lambda_cal, 
    r_min, r_max, phi_min, phi_max, lambda_min, lambda_max, 
    density, density_gradient, max_node, tag, ratio, is_linear_density):
    """
    Calculate gravitational field of a single tesseroid with size 
    (r_1, r_2) \times (phi_1, phi_2) \times (lambda_1, lambda_2)
    at point (r_cal, phi_cal, lambda_cal).

    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: float
        Latitude of computation point in radian.
    lambda_cal: float
        Longitude of compuattion point in radian.
    r_min: float
        Min radius of tesseroid in meter.
    r_max: float
        Max radius of tesseroid in meter.
    phi_min: float
        Min latitude of tesseroid in radian.
    phi_max: float
        Max latitude of tesseroid in radian.
    lambda_min: float
        Min longitude of tesseroid in radian.
    lambda_max: float
        Max longitude of tesseroid in radian.   
    density: float
        Density of tesseroid, unit: kg/m^3.
    density_gradient: float
        Density gradient of tesseroid, unit: kg/m^3/m.
    max_node: int
        Max node in $r$ direction, $\phi$ direction, 
        and $\lambda$ direction. In this program, 
        the nodes in radial, latitude, and longitude direction 
        take the same value.
    tag: string
        Kernel function to be calculated. 
        tag \in {'V', 'Vx', 'Vy', 'Vz', 
        'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz'}
    ratio: float
        Distance-size ratio, which is specified by the user. 
        The larger ratio is, the smaller tesseroid is divided, 
        and the higher accuracy of calculation is.
    is_linear_density: bool
        If the tesseroid have linearly varying density.

    Returns
    -------
    result: float
        The gravitational field generated by tesseroid.
    """
    r0 = r_max
    phi0 = (phi_max + phi_min) / 2
    lambda0 = (lambda_max + lambda_min) / 2
    ell0 = cal_distance(r_cal, phi_cal, lambda_cal, r0, phi0, lambda0)
    
    L_phi = r_max * (phi_max - phi_min)
    L_lambda = r_max * math.cos(phi_min) * (lambda_max - lambda_min)
    
    result_linear = 0
    if (ell0/L_phi>ratio) and (ell0/L_lambda>ratio):
        if is_linear_density:
            result_constant, result_linear = direct_cal_gravitational_field(r_cal, phi_cal, lambda_cal, 
                r_min, r_max, phi_min, phi_max, lambda_min, lambda_max, 
                density, density_gradient, max_node, tag, is_linear_density)
        else:
            result_constant = direct_cal_gravitational_field(r_cal, phi_cal, lambda_cal, 
                r_min, r_max, phi_min, phi_max, lambda_min, lambda_max, 
                density, density_gradient, max_node, tag, is_linear_density)
        return result_constant+result_linear
    else:
        result = subdivision(r_cal, phi_cal, lambda_cal, 
            r_min, r_max, phi_min, phi_max, lambda_min, lambda_max,
            ratio, density, density_gradient, max_node, tag, is_linear_density)
        return result
    

def cal_single_tess_gravitational_field(r_cal, phi_cal, lambda_cal, 
    r_min, r_max, phi_min, phi_max, lambda_min, lambda_max, 
    density, density_gradient, max_node, tag, ratio, is_linear_density):
    """
    Calculate the gravitational field of tesseroid at all calculation points.
    
    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: numpy.ndarray, float
        Latitude of computation point in radian.
    lambda_cal: numpy.ndarray, float
        Longitude of computation point in radian.
    r_min: float
        Min radius of tesseroid in meter.
    r_max: float
        Max radius of tesseroid in meter.
    phi_min: float
        Min latitude of tesseroid in radian.
    phi_max: float
        Max latitude of tesseroid in radian.
    lambda_min: float
        Min longitude of tesseroid in radian.
    lambda_max: float
        Max longitude of tesseroid in radian.   
    density: float
        Density of tesseroid, unit: kg/m^3.
    density_gradient: float
        Density gradient of tesseroid, unit: kg/m^3/m.
    max_node: int
        Max node in $r$ direction, $\phi$ direction, 
        and $\lambda$ direction. In this program, 
        the nodes in radial, latitude, and longitude direction 
        take the same value.
    tag: string
        Kernel function to be calculated. 
        tag \in {'V', 'Vx', 'Vy', 'Vz', 
        'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz'}
    ratio: float
        Distance-size ratio, which is specified by the user. 
        The larger ratio is, the smaller tesseroid is divided, 
        and the higher accuracy of calculation is.
    is_linear_density: bool
        If the tesseroid have linearly varying density.

    Returns
    -------
    result: numpy.ndarray
        Gravitational field of tesseroid at all calculation points.
    """

    gf = np.zeros((len(phi_cal), len(lambda_cal)))

    # loop every calculation in latitude direction
    for index_latitude in range(len(phi_cal)): 
        # loop every calculation in longitude direction
        for index_longitude in range(len(lambda_cal)): 
            temp = cal_single2single_gravitational_field(\
                r_cal, phi_cal[index_latitude], lambda_cal[index_longitude], \
                r_min, r_max, phi_min, phi_max, lambda_min, lambda_max, \
                density, density_gradient, max_node, tag, ratio, is_linear_density)
            gf[index_latitude, index_longitude] = temp
                
    return gf


def cal_gravitational_field(r_cal, phi_cal, lambda_cal, 
    r_min, r_max, phi_min, phi_max, lambda_min, lambda_max, 
    density, density_gradient, max_node, ratio, tag, 
    is_linear_density=False, is_parallel_computing=True):
    """
    Calculate the gravitational field of tesseroids.
    
    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: numpy.ndarray, float
        Latitude of computation point in radian.
    lambda_cal: numpy.ndarray, float
        Longitude of computation point in radian.
    r_min: numpy.ndarray, float
        Radius of tesseroid in meter.
    r_max: numpy.ndarray, float
        Radius of tesseroid in meter.
    phi_min: float
        Latitude of tesseroid in radian.
    phi_max: float
        Latitude of tesseroid in radian.
    lambda_min: float
        Longitude of tesseroid in radian.
    lambda_max: float
        Longitude of tesseroid in radian.
    density: numpy.ndarray, float
        Density of tesseroid in kg/m^3.
    density_gradient: numpy.ndarray, float
        Density gradient of tesseroid in kg/m^3/m.
    max_node: int
        Max node in $r$ direction, $\phi$ direction, 
        and $\lambda$ direction.
    ratio: int
        Distance-size ratio, which is specified by the user. 
        The larger ratio is, the smaller tesseroid is divided, 
        and the higher accuracy of calculation is.
    tag: string
        Kernel function to be calculated. 
        tag \in {'V', 'Vx', 'Vy', 'Vz', 
        'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz'}
    is_linear_density: bool
        If the tesseroid have linear varying density.
    is_parallel_computing: bool
        If parallel computing is turned on.

    Returns
    -------
    result: numpy.ndarray
        Gravitational field of large-scale tesseroid.
    """
    apparent_density = density - density_gradient * r_min
    tesseroids = []
    [phi_num, lambda_num] = r_min.shape
    delta_phi = (phi_max - phi_min) / phi_num
    delta_lambda = (lambda_max - lambda_min) / lambda_num

    gf = np.zeros((len(phi_cal), len(lambda_cal)))
    for idx_latitude in range(phi_num):
        phi_min_temp = phi_min + delta_phi * idx_latitude
        phi_max_temp = phi_min_temp + delta_phi
        for idx_longitude in range(lambda_num):
            lambda_min_temp = lambda_min + delta_lambda * idx_longitude
            lambda_max_temp = lambda_min_temp + delta_lambda

            tesseroids.append(Tesseroid(r_min[idx_latitude, idx_longitude], \
                    r_max[idx_latitude, idx_longitude], \
                    phi_min_temp, phi_max_temp, \
                    lambda_min_temp, lambda_max_temp, \
                    apparent_density[idx_latitude, idx_longitude], 
                    density_gradient[idx_latitude, idx_longitude]))
    
    if is_parallel_computing:
        pool = mp.Pool(mp.cpu_count())
        results = pool.starmap(cal_single_tess_gravitational_field, \
            [(r_cal, phi_cal, lambda_cal, 
            tess.r_min, tess.r_max, tess.phi_min, tess.phi_max, \
            tess.lambda_min, tess.lambda_max, \
            tess.apparent_density, tess.density_gradient, \
            max_node, tag, ratio, is_linear_density) 
            for tess in tesseroids])
        pool.close()
        for res in results:
            gf += res
    else:
        for tess in tesseroids:
            gf += cal_single_tess_gravitational_field(\
                r_cal, phi_cal, lambda_cal, \
                tess.r_min, tess.r_max, tess.phi_min, tess.phi_max, \
                tess.lambda_min, tess.lambda_max, \
                tess.apparent_density, tess.density_gradient, \
                max_node, tag, ratio, is_linear_density)
    return gf
    