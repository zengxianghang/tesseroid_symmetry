import numpy as np
import sympy as sy
import multiprocessing as mp
import math


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
    cosPsi = np.sin(phi1) * np.sin(phi2) \
        + np.cos(phi1) * np.cos(phi2) \
        * np.cos(lambda1 - lambda2)
    distance = np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * cosPsi)

    return distance


def cal_V_kernel(r_cal, phi_cal, lambda_cal, 
        r_tess, phi_tess, lambda_tess, 
        order, is_linear_density):
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
    order: int
        Differentiation of kernel function.
    is_linear_density: bool
        If the tesseroid have linearly varying density.
        
    Returns
    -------
    kernel: float
        Kernel of gravitational potential.
    """
    a, b, c, d, e, f = r_tess, r_cal, phi_tess, phi_cal, lambda_tess, lambda_cal
    # Common part in taylor series.
    temp1 = np.cos(c)*np.cos(d)*np.cos(e - f)
    temp2 = np.sin(c)*np.sin(d)
    temp3 = temp1 + temp2
    temp4 = a - b * (temp3)
    temp6 = np.cos(c)
    a2 = a * a
    a3 = a2 * a
    ell = a2 - 2*a*b*(temp3) + b * b
    ell25 = ell**(5/2)

    if is_linear_density:
        if order==0:
            if np.abs(temp3-1)<1e-15:
                # kernel = -1/6 * temp6 * ( 2 * a3 +  3 * a2 * b +  6 * a * b**2  -11 * b**3 + 6 * b**3 * np.log(b-a))
                kernel = -1/6 * temp6 * ( 2 * a3 + ( 3 * a2 * b + ( 6 * a * ( b )**( 2 ) + ( -11 * \
                    ( b )**( 3 ) + 6 * ( b )**( 3 ) * np.log( ( -1 * a + b ) ) ) ) ) )
                kernel = - 1 / 6 * temp6 * (2 * a3 + 3 * b * a2 + 6 * b*b * a + 6 * a3 * np.log(b-a))
            elif np.abs(temp3+1)<1e-15:
                kernel = 1/6 * temp6 * ( ( a + b ) * ( 2 * a2 + ( -5 * a * b + 11 * ( b )**( 2 ) ) ) + -6 * \
                    ( b )**( 3 ) * np.log( ( a + b ) ) )
                # kernel = 1/6 * temp6 * (2 * a3 - 3 * b * a2 + 6 * b*b * a - 6 * a3 * np.log(a+b))
            else:
                kernel = 1/6 * temp6 * ( np.sqrt(ell) * ( 2 * a2 + ( 5 * a * b * temp3 + ( b )**( 2 ) * ( \
                    -4 + 15 * ( temp3 )**( 2 ) ) ) ) + 3 * ( b )**( 3 ) * temp3 * ( -3 + 5 * ( temp3 \
                    )**( 2 ) ) * np.log( ( a + ( -1 * b * temp3 +   np.sqrt(ell) ) ) ) )
        elif order==1:
            kernel = a3*temp6/np.sqrt(ell)
        elif order==2:
            kernel = a3*(-temp4)*temp6/(ell)**(3/2) + 3*a2*temp6/np.sqrt(ell)
        elif order==3:
            kernel = a*(a2*(3*(temp4)**2/(ell) - 1)/(ell) - 6*a*(temp4)/(ell) + 6)*temp6/np.sqrt(ell)
        elif order==4:
            kernel = 3*(-a3*(temp4)*(5*(temp4)**2/(ell) - 3)/(ell)**2 + 3*a2*(3*(temp4)**2/(ell) - 1)/(ell) - 6*a*(temp4)/(ell) + 2)*temp6/np.sqrt(ell)
        elif order==5:
            kernel = 3*(a3*(35*(temp4)**4/(ell)**2 - 30*(temp4)**2/(ell) + 3)/(ell) - 12*a2*(temp4)*(5*(temp4)**2/(ell) - 3)/(ell) + 12*a*(3*(temp4)**2/(ell) - 1) - 8*a + 8*b*(temp3))*temp6/(ell)**(3/2)
        elif order==6:
            kernel = 15*(-a3*(temp4)*(63*(temp4)**4/(ell)**2 - 70*(temp4)**2/(ell) + 15)/(ell)**2 + 3*a2*(35*(temp4)**4/(ell)**2 - 30*(temp4)**2/(ell) + 3)/(ell) - 12*a*(temp4)*(5*(temp4)**2/(ell) - 3)/(ell) + 12*(temp4)**2/(ell) - 4)*temp6/(ell)**(3/2)
        elif order==7:
            kernel = 45*(a3*(231*(temp4)**6/(ell)**3 - 315*(temp4)**4/(ell)**2 + 105*(temp4)**2/(ell) - 5)/(ell) - 6*a2*(temp4)*(63*(temp4)**4/(ell)**2 - 70*(temp4)**2/(ell) + 15)/(ell) + 6*a*(35*(temp4)**4/(ell)**2 - 30*(temp4)**2/(ell) + 3) - 8*(temp4)*(5*(temp4)**2/(ell) - 3))*temp6/ell25
        elif order==8:
            kernel = 315*(-a3*(temp4)*(429*(temp4)**6/(ell)**3 - 693*(temp4)**4/(ell)**2 + 315*(temp4)**2/(ell) - 35)/(ell)**2 + 3*a2*(231*(temp4)**6/(ell)**3 - 315*(temp4)**4/(ell)**2 + 105*(temp4)**2/(ell) - 5)/(ell) - 6*a*(temp4)*(63*(temp4)**4/(ell)**2 - 70*(temp4)**2/(ell) + 15)/(ell) + 70*(temp4)**4/(ell)**2 - 60*(temp4)**2/(ell) + 6)*temp6/ell25
        elif order==9:
            kernel = 315*(a3*(6435*(temp4)**8/(ell)**4 - 12012*(temp4)**6/(ell)**3 + 6930*(temp4)**4/(ell)**2 - 1260*(temp4)**2/(ell) + 35)/(ell) - 24*a2*(temp4)*(429*(temp4)**6/(ell)**3 - 693*(temp4)**4/(ell)**2 + 315*(temp4)**2/(ell) - 35)/(ell) + 24*a*(231*(temp4)**6/(ell)**3 - 315*(temp4)**4/(ell)**2 + 105*(temp4)**2/(ell) - 5) - 16*(temp4)*(63*(temp4)**4/(ell)**2 - 70*(temp4)**2/(ell) + 15))*temp6/(ell)**(7/2)
        elif order==10:
            kernel = 2835*(-a3*(temp4)*(12155*(temp4)**8/(ell)**4 - 25740*(temp4)**6/(ell)**3 + 18018*(temp4)**4/(ell)**2 - 4620*(temp4)**2/(ell) + 315)/(ell)**2 + 3*a2*(6435*(temp4)**8/(ell)**4 - 12012*(temp4)**6/(ell)**3 + 6930*(temp4)**4/(ell)**2 - 1260*(temp4)**2/(ell) + 35)/(ell) - 24*a*(temp4)*(429*(temp4)**6/(ell)**3 - 693*(temp4)**4/(ell)**2 + 315*(temp4)**2/(ell) - 35)/(ell) + 1848*(temp4)**6/(ell)**3 - 2520*(temp4)**4/(ell)**2 + 840*(temp4)**2/(ell) - 40)*temp6/(ell)**(7/2)
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**3 * sy.cos(c) \
                / sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + \
                sy.cos(d) * sy.cos(c) * sy.cos(e - f)))
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    else:
        if order==0:
            if np.abs(temp3-1)<1e-15:
                kernel = -1/2 * temp6 * ( ( a )**( 2 ) + ( 2 * a * b + ( -3 * ( b )**( 2 ) + 2 * ( b )**( 2 ) * \
                    np.log( ( -1 * a + b ) ) ) ) )
            elif np.abs(temp3+1)<1e-15:
                kernel = temp6 * ( -2 * b * ( a + b ) + ( 1/2 * ( ( a + b ) )**( 2 ) + ( b )**( 2 ) * np.log( ( a + \
                    b ) ) ) )
            else:
                kernel = temp6 / 2 * (np.sqrt(ell) * (a + 3 * b * temp3) \
                    + b**2 * (3 * temp3**2 - 1) * np.log(np.sqrt(ell) + a - b * temp3))
        elif order==1:
            kernel = a2*temp6/np.sqrt(ell)
        elif order==2:
            kernel = a2*(-temp4)*temp6/(ell)**(3/2) + 2*a*temp6/np.sqrt(ell)
        elif order==3:
            kernel = (a2*(3*(temp4)**2/(ell) - 1)/(ell) - 4*a*(temp4)/(ell) + 2)*temp6/np.sqrt(ell)
        elif order==4:
            kernel = 3*(-a2*(temp4)*(5*(temp4)**2/(ell) - 3)/(ell) + 2*a*(3*(temp4)**2/(ell) - 1) - 2*a + 2*b*(temp3))*temp6/(ell)**(3/2)
        elif order==5:
            kernel = 3*(a2*(35*(temp4)**4/(ell)**2 - 30*(temp4)**2/(ell) + 3)/(ell) - 8*a*(temp4)*(5*(temp4)**2/(ell) - 3)/(ell) + 12*(temp4)**2/(ell) - 4)*temp6/(ell)**(3/2)
        elif order==6:
            kernel = 15*(-a2*(temp4)*(63*(temp4)**4/(ell)**2 - 70*(temp4)**2/(ell) + 15)/(ell) + 2*a*(35*(temp4)**4/(ell)**2 - 30*(temp4)**2/(ell) + 3) - 4*(temp4)*(5*(temp4)**2/(ell) - 3))*temp6/ell25
        elif order==7:
            kernel = 45*(a2*(231*(temp4)**6/(ell)**3 - 315*(temp4)**4/(ell)**2 + 105*(temp4)**2/(ell) - 5)/(ell) - 4*a*(temp4)*(63*(temp4)**4/(ell)**2 - 70*(temp4)**2/(ell) + 15)/(ell) + 70*(temp4)**4/(ell)**2 - 60*(temp4)**2/(ell) + 6)*temp6/ell25
        elif order==8:
            kernel = 315*(-a2*(temp4)*(429*(temp4)**6/(ell)**3 - 693*(temp4)**4/(ell)**2 + 315*(temp4)**2/(ell) - 35)/(ell) + 2*a*(231*(temp4)**6/(ell)**3 - 315*(temp4)**4/(ell)**2 + 105*(temp4)**2/(ell) - 5) - 2*(temp4)*(63*(temp4)**4/(ell)**2 - 70*(temp4)**2/(ell) + 15))*temp6/(ell)**(7/2)
        elif order==9:
            kernel = 315*(a2*(6435*(temp4)**8/(ell)**4 - 12012*(temp4)**6/(ell)**3 + 6930*(temp4)**4/(ell)**2 - 1260*(temp4)**2/(ell) + 35)/(ell) - 16*a*(temp4)*(429*(temp4)**6/(ell)**3 - 693*(temp4)**4/(ell)**2 + 315*(temp4)**2/(ell) - 35)/(ell) + 1848*(temp4)**6/(ell)**3 - 2520*(temp4)**4/(ell)**2 + 840*(temp4)**2/(ell) - 40)*temp6/(ell)**(7/2)
        elif order==10:
            kernel = 2835*(-a2*(temp4)*(12155*(temp4)**8/(ell)**4 - 25740*(temp4)**6/(ell)**3 + 18018*(temp4)**4/(ell)**2 - 4620*(temp4)**2/(ell) + 315)/(ell) + 2*a*(6435*(temp4)**8/(ell)**4 - 12012*(temp4)**6/(ell)**3 + 6930*(temp4)**4/(ell)**2 - 1260*(temp4)**2/(ell) + 35) - 8*(temp4)*(429*(temp4)**6/(ell)**3 - 693*(temp4)**4/(ell)**2 + 315*(temp4)**2/(ell) - 35))*temp6/(ell)**(9/2)
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**3 * sy.cos(c) \
                / sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + \
                sy.cos(d) * sy.cos(c) * sy.cos(e - f)))
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    
    return kernel


def cal_Vx_kernel(r_cal, phi_cal, lambda_cal, \
        r_tess, phi_tess, lambda_tess, \
        order, is_linear_density): 
    """
    Calculate the kernel of gravitational acceleration Vx.

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
    order: int
        Differentiation of kernel function.
    is_linear_density: bool
        If the tesseroid have linear varying density.
        
    Returns
    -------
    kernel: float
        Kernel of gravitational acceleration Vx.
    """
    a, b, c, d, e, f = r_tess, r_cal, phi_tess, phi_cal, lambda_tess, lambda_cal
    # Common part in taylor series.
    n = np.cos(phi_tess)
    g = np.cos(phi_cal) * np.sin(phi_tess) - np.sin(phi_cal) * n\
        * np.cos(lambda_tess - lambda_cal)
    # h = n * np.sin(lambda_tess - lambda_cal)
    m = np.sin(phi_cal) * np.sin(phi_tess) + n * np.cos(phi_cal)\
        * np.cos(lambda_tess - lambda_cal)
    ell = (a**2 + b**2 - 2 * a * b * m)**0.5
    tmp1 = 2 * a - 2 * b * m

    if is_linear_density:
        if order == 0:
            if np.abs(m - 1) < 1e-15:
                kernel = g * n * ( -1/2 * a * ( ( a + -1 * b ) )**( -2 ) * ( ( a )**( 3 ) + ( \
                    4 * ( a )**( 2 ) * b + ( -18 * a * ( b )**( 2 ) + 12 * ( b )**( 3 ) ) \
                    ) ) + -6 * ( b )**( 2 ) * np.log( ( -1 * a + b ) ) )
            elif np.abs(m + 1) < 1e-15:
                kernel = g * n * ( 1/2 * a * ( ( a + b ) )**( -2 ) * ( ( a )**( 3 ) + ( -4 * ( \
                    a )**( 2 ) * b + ( -18 * a * ( b )**( 2 ) + -12 * ( b )**( 3 ) ) ) ) \
                    + 6 * ( b )**( 2 ) * np.log( ( a + b ) ) )
            else:
                kernel = 1/2 * g * n * ( (ell)**(-1) * ( ( -1 + ( m )**( 2 ) ) )**( -1 ) * ( ( a )**( 3 ) * ( -1 + ( m )**( 2 ) ) + ( 5 * ( a )**( 2 ) * b * m * ( -1 + ( m )**( 2 ) ) + ( ( b )**( 3 ) * m * ( -13 + 15 * ( m )**( 2 ) ) + a * ( b )**( 2 ) * ( -3 + ( 31 * ( m )**( 2 ) + -30 * ( m )**( 4 ) ) ) ) ) ) + -3 * ( b )**( 2 ) * ( -1 + 5 * ( m )**( 2 ) ) * np.log( ( -1 * a + ( b * m + ( ( ( a )**( 2 ) + ( ( b )**( 2 ) + -2 * a * b * m ) ) )**( 1/2 ) ) ) ) )
        elif order == 1:
            kernel = ( a )**( 4 ) * g * (ell)**(-3) * n
        elif order == 2:
            kernel = ( -3/2 * ( a )**( 4 ) * g * tmp1 * (ell)**(-5) * n + 4 * ( a )**( 3 ) * g * (ell)**(-3) * n )
        elif order == 3:
            kernel = ( -12 * ( a )**( 3 ) * g * tmp1 * (ell)**(-5) * n + ( 12 * ( a )**( 2 ) * g * (ell)**(-3) * n + ( a )**( 4 ) * g * ( 15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + -3 * (ell)**(-5) ) * n ) )
        elif order == 4:
            kernel = ( -54 * ( a )**( 2 ) * g * tmp1 * (ell)**(-5) * n + ( 24 * a * g * (ell)**(-3) * n + ( ( a )**( 4 ) * g * ( -105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + 45/2 * tmp1 * (ell)**(-7) ) * n + 12 * ( a )**( 3 ) * g * ( 15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + -3 * (ell)**(-5) ) * n ) ) )
        elif order == 5:
            kernel = ( -144 * a * g * tmp1 * (ell)**(-5) * n + ( 24 * g * (ell)**(-3) * n + ( ( a )**( 4 ) * g * ( 945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( -315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + 45 * (ell)**(-7) ) ) * n + ( 16 * ( a )**( 3 ) * g * ( -105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + 45/2 * tmp1 * (ell)**(-7) ) * n + 72 * ( a )**( 2 ) * g * ( 15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + -3 * (ell)**(-5) ) * n ) ) ) )
        elif order == 6:
            kernel = ( -180 * g * tmp1 * (ell)**(-5) * n + ( ( a )**( 4 ) * g * ( -10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( 4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + -1575/2 * tmp1 * (ell)**(-9) ) ) * n + ( 20 * ( a )**( 3 ) * g * ( 945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( -315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + 45 * (ell)**(-7) ) ) * n + ( 120 * ( a )**( 2 ) * g * ( -105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + 45/2 * tmp1 * (ell)**(-7) ) * n + 240 * a * g * ( 15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + -3 * (ell)**(-5) ) * n ) ) ) )
        elif order == 7:
            kernel = ( ( a )**( 4 ) * g * ( 135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( -155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( 42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + -1575 * (ell)**(-9) ) ) ) * n + ( 24 * ( a )**( 3 ) * g * ( -10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( 4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + -1575/2 * tmp1 * (ell)**(-9) ) ) * n + ( 180 * ( a )**( 2 ) * g * ( 945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( -315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + 45 * (ell)**(-7) ) ) * n + ( 480 * a * g * ( -105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + 45/2 * tmp1 * (ell)**(-7) ) * n + 360 * g * ( 15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + -3 * (ell)**(-5) ) * n ) ) ) )
        elif order == 8:
            kernel = ( ( a )**( 4 ) * g * ( -2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( 2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( -1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + 99225/2 * tmp1 * (ell)**(-11) ) ) ) * n + ( 28 * ( a )**( 3 ) * g * ( 135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( -155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( 42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + -1575 * (ell)**(-9) ) ) ) * n + ( 252 * ( a )**( 2 ) * g * ( -10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( 4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + -1575/2 * tmp1 * (ell)**(-9) ) ) * n + ( 840 * a * g * ( 945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( -315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + 45 * (ell)**(-7) ) ) * n + 840 * g * ( -105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + 45/2 * tmp1 * (ell)**(-7) ) * n ) ) ) )
        elif order == 9:
            kernel = ( ( a )**( 4 ) * g * ( 34459425/256 * ( tmp1 )**( 8 ) * (ell)**(-19) + ( -14189175/16 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( 14189175/8 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( -1091475 * ( tmp1 )**( 2 ) * (ell)**(-13) + 99225 * (ell)**(-11) ) ) ) ) * n + ( 32 * ( a )**( 3 ) * g * ( -2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( 2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( -1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + 99225/2 * tmp1 * (ell)**(-11) ) ) ) * n + ( 336 * ( a )**( 2 ) * g * ( 135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( -155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( 42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + -1575 * (ell)**(-9) ) ) ) * n + ( 1344 * a * g * ( -10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( 4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + -1575/2 * tmp1 * (ell)**(-9) ) ) * n + 1680 * g * ( 945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( -315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + 45 * (ell)**(-7) ) ) * n ) ) ) )
        elif order == 10:
            kernel = ( ( a )**( 4 ) * g * ( -654729075/512 * ( tmp1 )**( 9 ) * (ell)**(-21) + ( 310134825/32 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( -383107725/16 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( 42567525/2 * ( tmp1 )**( 3 ) * (ell)**(-15) + -9823275/2 * tmp1 * (ell)**(-13) ) ) ) ) * n + ( 36 * ( a )**( 3 ) * g * ( 34459425/256 * ( tmp1 )**( 8 ) * (ell)**(-19) + ( -14189175/16 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( 14189175/8 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( -1091475 * ( tmp1 )**( 2 ) * (ell)**(-13) + 99225 * (ell)**(-11) ) ) ) ) * n + ( 432 * ( a )**( 2 ) * g * ( -2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( 2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( -1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + 99225/2 * tmp1 * (ell)**(-11) ) ) ) * n + ( 2016 * a * g * ( 135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( -155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( 42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + -1575 * (ell)**(-9) ) ) ) * n + 3024 * g * ( -10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( 4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + -1575/2 * tmp1 * (ell)**(-9) ) ) * n ) ) ) )
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**3 * sy.cos(c) \
                * (a*(sy.cos(d)*sy.sin(c)-sy.sin(d)*sy.cos(c)*sy.cos(e-f))) \
                / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**3
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    else:
        if order == 0:
            if np.abs(m - 1) < 1e-15:
                kernel = g * n * ( -1 * a + ( 1/2 * ( 6 * a + -5 * b ) * ( ( a + -1 * b ) )**( \
                    -2 ) * ( b )**( 2 ) + -3 * b * np.log( ( -1 * a + b ) ) ) )
            elif np.abs(m + 1) < 1e-15:
                kernel = g * n * ( a + ( -1/2 * ( b )**( 2 ) * ( ( a + b ) )**( -2 ) * ( 6 * \
                    a + 5 * b ) + -3 * b * np.log( ( a + b ) ) ) )
            else:
                kernel = g * n * ( (ell)**(-1) * ( ( -1 + ( m )**( 2 ) ) )**( -1 ) * ( a * b * m * ( 5 + -6 * ( m )**( 2 ) ) + ( ( a )**( 2 ) * ( -1 + ( m )**( 2 ) ) + ( b )**( 2 ) * ( -2 + 3 * ( m )**( 2 ) ) ) ) + -3 * b * m * np.log( ( -1 * a + ( b * m + ( ( ( a )**( 2 ) + ( ( b )**( 2 ) + -2 * a * b * m ) ) )**( 1/2 ) ) ) ) )
        elif order == 1:
            kernel = ( a )**( 3 ) * g * (ell)**(-3) * n
        elif order == 2:
            kernel = ( -3/2 * ( a )**( 3 ) * g * tmp1 * (ell)**(-5) * n + 3 * ( a )**( 2 ) * g * (ell)**(-3) * n )
        elif order == 3:
            kernel = ( -9 * ( a )**( 2 ) * g * tmp1 * (ell)**(-5) * n + ( 6 * a * g * (ell)**(-3) * n + ( a )**( 3 ) * g * ( 15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + -3 * (ell)**(-5) ) * n ) )
        elif order == 4:
            kernel = ( -27 * a * g * tmp1 * (ell)**(-5) * n + ( 6 * g * (ell)**(-3) * n + ( ( a )**( 3 ) * g * ( -105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + 45/2 * tmp1 * (ell)**(-7) ) * n + 9 * ( a )**( 2 ) * g * ( 15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + -3 * (ell)**(-5) ) * n ) ) )
        elif order == 5:
            kernel = ( -36 * g * tmp1 * (ell)**(-5) * n + ( ( a )**( 3 ) * g * ( 945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( -315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + 45 * (ell)**(-7) ) ) * n + ( 12 * ( a )**( 2 ) * g * ( -105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + 45/2 * tmp1 * (ell)**(-7) ) * n + 36 * a * g * ( 15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + -3 * (ell)**(-5) ) * n ) ) )
        elif order == 6:
            kernel = ( ( a )**( 3 ) * g * ( -10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( 4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + -1575/2 * tmp1 * (ell)**(-9) ) ) * n + ( 15 * ( a )**( 2 ) * g * ( 945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( -315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + 45 * (ell)**(-7) ) ) * n + ( 60 * a * g * ( -105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + 45/2 * tmp1 * (ell)**(-7) ) * n + 60 * g * ( 15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + -3 * (ell)**(-5) ) * n ) ) )
        elif order == 7:
            kernel = ( ( a )**( 3 ) * g * ( 135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( -155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( 42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + -1575 * (ell)**(-9) ) ) ) * n + ( 18 * ( a )**( 2 ) * g * ( -10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( 4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + -1575/2 * tmp1 * (ell)**(-9) ) ) * n + ( 90 * a * g * ( 945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( -315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + 45 * (ell)**(-7) ) ) * n + 120 * g * ( -105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + 45/2 * tmp1 * (ell)**(-7) ) * n ) ) )
        elif order == 8:
            kernel = ( ( a )**( 3 ) * g * ( -2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( 2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( -1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + 99225/2 * tmp1 * (ell)**(-11) ) ) ) * n + ( 21 * ( a )**( 2 ) * g * ( 135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( -155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( 42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + -1575 * (ell)**(-9) ) ) ) * n + ( 126 * a * g * ( -10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( 4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + -1575/2 * tmp1 * (ell)**(-9) ) ) * n + 210 * g * ( 945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( -315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + 45 * (ell)**(-7) ) ) * n ) ) )
        elif order == 9:
            kernel = ( ( a )**( 3 ) * g * ( 34459425/256 * ( tmp1 )**( 8 ) * (ell)**(-19) + ( -14189175/16 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( 14189175/8 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( -1091475 * ( tmp1 )**( 2 ) * (ell)**(-13) + 99225 * (ell)**(-11) ) ) ) ) * n + ( 24 * ( a )**( 2 ) * g * ( -2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( 2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( -1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + 99225/2 * tmp1 * (ell)**(-11) ) ) ) * n + ( 168 * a * g * ( 135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( -155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( 42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + -1575 * (ell)**(-9) ) ) ) * n + 336 * g * ( -10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( 4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + -1575/2 * tmp1 * (ell)**(-9) ) ) * n ) ) )
        elif order == 10:
            kernel = ( ( a )**( 3 ) * g * ( -654729075/512 * ( tmp1 )**( 9 ) * (ell)**(-21) + ( 310134825/32 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( -383107725/16 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( 42567525/2 * ( tmp1 )**( 3 ) * (ell)**(-15) + -9823275/2 * tmp1 * (ell)**(-13) ) ) ) ) * n + ( 27 * ( a )**( 2 ) * g * ( 34459425/256 * ( tmp1 )**( 8 ) * (ell)**(-19) + ( -14189175/16 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( 14189175/8 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( -1091475 * ( tmp1 )**( 2 ) * (ell)**(-13) + 99225 * (ell)**(-11) ) ) ) ) * n + ( 216 * a * g * ( -2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( 2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( -1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + 99225/2 * tmp1 * (ell)**(-11) ) ) ) * n + 504 * g * ( 135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( -155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( 42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + -1575 * (ell)**(-9) ) ) ) * n ) ) )
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**2 * sy.cos(c) \
                * (a*(sy.cos(d)*sy.sin(c)-sy.sin(d)*sy.cos(c)*sy.cos(e-f))) \
                / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**3
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    return kernel


def cal_Vy_kernel(r_cal, phi_cal, lambda_cal, \
        r_tess, phi_tess, lambda_tess, \
        order, is_linear_density): 
    """
    Calculate the kernel of gravitational acceleration Vy.

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
    order: int
        Differentiation of kernel function.
    is_linear_density: bool
        If the tesseroid have linear varying density.
        
    Returns
    -------
    kernel: float
        Kernel of gravitational acceleration Vy.
    """
    a, b, c, d, e, f = r_tess, r_cal, phi_tess, phi_cal, lambda_tess, lambda_cal
    # Common part of taylor series.
    n = np.cos(phi_tess)
    g = n * np.sin(lambda_tess - lambda_cal)
    m = np.sin(phi_cal) * np.sin(phi_tess) + n * np.cos(phi_cal)\
        * np.cos(lambda_tess - lambda_cal)
    ell = (a**2 + b**2 - 2 * a * b * m)**0.5
    tmp1 = 2 * a - 2 * b * m

    if is_linear_density:
        if order == 0:
            if np.abs(m - 1) < 1e-15:
                kernel = g * n * ( -1/2 * a * ( ( a + -1 * b ) )**( -2 ) * ( ( a )**( 3 ) + ( \
                    4 * ( a )**( 2 ) * b + ( -18 * a * ( b )**( 2 ) + 12 * ( b )**( 3 ) ) \
                    ) ) + -6 * ( b )**( 2 ) * np.log( ( -1 * a + b ) ) )
            elif np.abs(m + 1) < 1e-15:
                kernel = g * n * ( 1/2 * a * ( ( a + b ) )**( -2 ) * ( ( a )**( 3 ) + ( -4 * ( \
                    a )**( 2 ) * b + ( -18 * a * ( b )**( 2 ) + -12 * ( b )**( 3 ) ) ) ) \
                    + 6 * ( b )**( 2 ) * np.log( ( a + b ) ) )
            else:
                kernel = 1/2 * g * n * ( (ell)**(-1) * ( ( -1 + ( m )**( 2 ) ) )**( -1 ) * ( ( a )**( 3 ) * ( -1 + ( m )**( 2 ) ) + ( 5 * ( a )**( 2 ) * b * m * ( -1 + ( m )**( 2 ) ) + ( ( b )**( 3 ) * m * ( -13 + 15 * ( m )**( 2 ) ) + a * ( b )**( 2 ) * ( -3 + ( 31 * ( m )**( 2 ) + -30 * ( m )**( 4 ) ) ) ) ) ) + -3 * ( b )**( 2 ) * ( -1 + 5 * ( m )**( 2 ) ) * np.log( ( -1 * a + ( b * m + ( ( ( a )**( 2 ) + ( ( b )**( 2 ) + -2 * a * b * m ) ) )**( 1/2 ) ) ) ) )
        elif order == 1:
            kernel = ( a )**( 4 ) * g * (ell)**(-3) * n
        elif order == 2:
            kernel = ( -3/2 * ( a )**( 4 ) * g * tmp1 * (ell)**(-5) * n + 4 * ( a )**( 3 ) * g * (ell)**(-3) * n )
        elif order == 3:
            kernel = ( -12 * ( a )**( 3 ) * g * tmp1 * (ell)**(-5) * n + ( 12 * ( a )**( 2 ) * g * (ell)**(-3) * n + ( a )**( 4 ) * g * ( 15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + -3 * (ell)**(-5) ) * n ) )
        elif order == 4:
            kernel = ( -54 * ( a )**( 2 ) * g * tmp1 * (ell)**(-5) * n + ( 24 * a * g * (ell)**(-3) * n + ( ( a )**( 4 ) * g * ( -105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + 45/2 * tmp1 * (ell)**(-7) ) * n + 12 * ( a )**( 3 ) * g * ( 15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + -3 * (ell)**(-5) ) * n ) ) )
        elif order == 5:
            kernel = ( -144 * a * g * tmp1 * (ell)**(-5) * n + ( 24 * g * (ell)**(-3) * n + ( ( a )**( 4 ) * g * ( 945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( -315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + 45 * (ell)**(-7) ) ) * n + ( 16 * ( a )**( 3 ) * g * ( -105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + 45/2 * tmp1 * (ell)**(-7) ) * n + 72 * ( a )**( 2 ) * g * ( 15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + -3 * (ell)**(-5) ) * n ) ) ) )
        elif order == 6:
            kernel = ( -180 * g * tmp1 * (ell)**(-5) * n + ( ( a )**( 4 ) * g * ( -10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( 4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + -1575/2 * tmp1 * (ell)**(-9) ) ) * n + ( 20 * ( a )**( 3 ) * g * ( 945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( -315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + 45 * (ell)**(-7) ) ) * n + ( 120 * ( a )**( 2 ) * g * ( -105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + 45/2 * tmp1 * (ell)**(-7) ) * n + 240 * a * g * ( 15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + -3 * (ell)**(-5) ) * n ) ) ) )
        elif order == 7:
            kernel = ( ( a )**( 4 ) * g * ( 135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( -155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( 42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + -1575 * (ell)**(-9) ) ) ) * n + ( 24 * ( a )**( 3 ) * g * ( -10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( 4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + -1575/2 * tmp1 * (ell)**(-9) ) ) * n + ( 180 * ( a )**( 2 ) * g * ( 945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( -315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + 45 * (ell)**(-7) ) ) * n + ( 480 * a * g * ( -105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + 45/2 * tmp1 * (ell)**(-7) ) * n + 360 * g * ( 15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + -3 * (ell)**(-5) ) * n ) ) ) )
        elif order == 8:
            kernel = ( ( a )**( 4 ) * g * ( -2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( 2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( -1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + 99225/2 * tmp1 * (ell)**(-11) ) ) ) * n + ( 28 * ( a )**( 3 ) * g * ( 135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( -155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( 42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + -1575 * (ell)**(-9) ) ) ) * n + ( 252 * ( a )**( 2 ) * g * ( -10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( 4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + -1575/2 * tmp1 * (ell)**(-9) ) ) * n + ( 840 * a * g * ( 945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( -315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + 45 * (ell)**(-7) ) ) * n + 840 * g * ( -105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + 45/2 * tmp1 * (ell)**(-7) ) * n ) ) ) )
        elif order == 9:
            kernel = ( ( a )**( 4 ) * g * ( 34459425/256 * ( tmp1 )**( 8 ) * (ell)**(-19) + ( -14189175/16 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( 14189175/8 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( -1091475 * ( tmp1 )**( 2 ) * (ell)**(-13) + 99225 * (ell)**(-11) ) ) ) ) * n + ( 32 * ( a )**( 3 ) * g * ( -2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( 2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( -1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + 99225/2 * tmp1 * (ell)**(-11) ) ) ) * n + ( 336 * ( a )**( 2 ) * g * ( 135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( -155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( 42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + -1575 * (ell)**(-9) ) ) ) * n + ( 1344 * a * g * ( -10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( 4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + -1575/2 * tmp1 * (ell)**(-9) ) ) * n + 1680 * g * ( 945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( -315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + 45 * (ell)**(-7) ) ) * n ) ) ) )
        elif order == 10:
            kernel = ( ( a )**( 4 ) * g * ( -654729075/512 * ( tmp1 )**( 9 ) * (ell)**(-21) + ( 310134825/32 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( -383107725/16 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( 42567525/2 * ( tmp1 )**( 3 ) * (ell)**(-15) + -9823275/2 * tmp1 * (ell)**(-13) ) ) ) ) * n + ( 36 * ( a )**( 3 ) * g * ( 34459425/256 * ( tmp1 )**( 8 ) * (ell)**(-19) + ( -14189175/16 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( 14189175/8 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( -1091475 * ( tmp1 )**( 2 ) * (ell)**(-13) + 99225 * (ell)**(-11) ) ) ) ) * n + ( 432 * ( a )**( 2 ) * g * ( -2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( 2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( -1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + 99225/2 * tmp1 * (ell)**(-11) ) ) ) * n + ( 2016 * a * g * ( 135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( -155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( 42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + -1575 * (ell)**(-9) ) ) ) * n + 3024 * g * ( -10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( 4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + -1575/2 * tmp1 * (ell)**(-9) ) ) * n ) ) ) )
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**3 * sy.cos(c) \
                * (a*(sy.cos(d)*sy.sin(c)-sy.sin(d)*sy.cos(c)*sy.cos(e-f))) \
                / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**3
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    else:
        if order == 0:
            if np.abs(m - 1) < 1e-15:
                kernel = g * n * ( -1 * a + ( 1/2 * ( 6 * a + -5 * b ) * ( ( a + -1 * b ) )**( \
                    -2 ) * ( b )**( 2 ) + -3 * b * np.log( ( -1 * a + b ) ) ) )
            elif np.abs(m + 1) < 1e-15:
                kernel = g * n * ( a + ( -1/2 * ( b )**( 2 ) * ( ( a + b ) )**( -2 ) * ( 6 * \
                    a + 5 * b ) + -3 * b * np.log( ( a + b ) ) ) )
            else:
                kernel = g * n * ( (ell)**(-1) * ( ( -1 + ( m )**( 2 ) ) )**( -1 ) * ( a * b * m * ( 5 + -6 * ( m )**( 2 ) ) + ( ( a )**( 2 ) * ( -1 + ( m )**( 2 ) ) + ( b )**( 2 ) * ( -2 + 3 * ( m )**( 2 ) ) ) ) + -3 * b * m * np.log( ( -1 * a + ( b * m + ( ( ( a )**( 2 ) + ( ( b )**( 2 ) + -2 * a * b * m ) ) )**( 1/2 ) ) ) ) )
        elif order == 1:
            kernel = ( a )**( 3 ) * g * (ell)**(-3) * n
        elif order == 2:
            kernel = ( -3/2 * ( a )**( 3 ) * g * tmp1 * (ell)**(-5) * n + 3 * ( a )**( 2 ) * g * (ell)**(-3) * n )
        elif order == 3:
            kernel = ( -9 * ( a )**( 2 ) * g * tmp1 * (ell)**(-5) * n + ( 6 * a * g * (ell)**(-3) * n + ( a )**( 3 ) * g * ( 15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + -3 * (ell)**(-5) ) * n ) )
        elif order == 4:
            kernel = ( -27 * a * g * tmp1 * (ell)**(-5) * n + ( 6 * g * (ell)**(-3) * n + ( ( a )**( 3 ) * g * ( -105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + 45/2 * tmp1 * (ell)**(-7) ) * n + 9 * ( a )**( 2 ) * g * ( 15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + -3 * (ell)**(-5) ) * n ) ) )
        elif order == 5:
            kernel = ( -36 * g * tmp1 * (ell)**(-5) * n + ( ( a )**( 3 ) * g * ( 945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( -315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + 45 * (ell)**(-7) ) ) * n + ( 12 * ( a )**( 2 ) * g * ( -105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + 45/2 * tmp1 * (ell)**(-7) ) * n + 36 * a * g * ( 15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + -3 * (ell)**(-5) ) * n ) ) )
        elif order == 6:
            kernel = ( ( a )**( 3 ) * g * ( -10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( 4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + -1575/2 * tmp1 * (ell)**(-9) ) ) * n + ( 15 * ( a )**( 2 ) * g * ( 945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( -315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + 45 * (ell)**(-7) ) ) * n + ( 60 * a * g * ( -105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + 45/2 * tmp1 * (ell)**(-7) ) * n + 60 * g * ( 15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + -3 * (ell)**(-5) ) * n ) ) )
        elif order == 7:
            kernel = ( ( a )**( 3 ) * g * ( 135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( -155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( 42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + -1575 * (ell)**(-9) ) ) ) * n + ( 18 * ( a )**( 2 ) * g * ( -10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( 4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + -1575/2 * tmp1 * (ell)**(-9) ) ) * n + ( 90 * a * g * ( 945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( -315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + 45 * (ell)**(-7) ) ) * n + 120 * g * ( -105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + 45/2 * tmp1 * (ell)**(-7) ) * n ) ) )
        elif order == 8:
            kernel = ( ( a )**( 3 ) * g * ( -2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( 2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( -1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + 99225/2 * tmp1 * (ell)**(-11) ) ) ) * n + ( 21 * ( a )**( 2 ) * g * ( 135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( -155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( 42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + -1575 * (ell)**(-9) ) ) ) * n + ( 126 * a * g * ( -10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( 4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + -1575/2 * tmp1 * (ell)**(-9) ) ) * n + 210 * g * ( 945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( -315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + 45 * (ell)**(-7) ) ) * n ) ) )
        elif order == 9:
            kernel = ( ( a )**( 3 ) * g * ( 34459425/256 * ( tmp1 )**( 8 ) * (ell)**(-19) + ( -14189175/16 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( 14189175/8 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( -1091475 * ( tmp1 )**( 2 ) * (ell)**(-13) + 99225 * (ell)**(-11) ) ) ) ) * n + ( 24 * ( a )**( 2 ) * g * ( -2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( 2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( -1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + 99225/2 * tmp1 * (ell)**(-11) ) ) ) * n + ( 168 * a * g * ( 135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( -155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( 42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + -1575 * (ell)**(-9) ) ) ) * n + 336 * g * ( -10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( 4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + -1575/2 * tmp1 * (ell)**(-9) ) ) * n ) ) )
        elif order == 10:
            kernel = ( ( a )**( 3 ) * g * ( -654729075/512 * ( tmp1 )**( 9 ) * (ell)**(-21) + ( 310134825/32 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( -383107725/16 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( 42567525/2 * ( tmp1 )**( 3 ) * (ell)**(-15) + -9823275/2 * tmp1 * (ell)**(-13) ) ) ) ) * n + ( 27 * ( a )**( 2 ) * g * ( 34459425/256 * ( tmp1 )**( 8 ) * (ell)**(-19) + ( -14189175/16 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( 14189175/8 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( -1091475 * ( tmp1 )**( 2 ) * (ell)**(-13) + 99225 * (ell)**(-11) ) ) ) ) * n + ( 216 * a * g * ( -2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( 2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( -1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + 99225/2 * tmp1 * (ell)**(-11) ) ) ) * n + 504 * g * ( 135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( -155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( 42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + -1575 * (ell)**(-9) ) ) ) * n ) ) )
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**2 * sy.cos(c) \
                * (a*(sy.cos(d)*sy.sin(c)-sy.sin(d)*sy.cos(c)*sy.cos(e-f))) \
                / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**3
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    return kernel



def cal_Vz_kernel(r_cal, phi_cal, lambda_cal, \
        r_tess, phi_tess, lambda_tess, \
        order, is_linear_density): 
    """
    Calculate the kernel of gravitational acceleration Vz.

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
    order: int
        Differentiation of kernel function.
    is_linear_density: bool
        If the tesseroid have linear varying density.
        
    Returns
    -------
    kernel: float
        Kernel of gravitational acceleration Vz.
    """
    a, b, c, d, e, f = r_tess, r_cal, phi_tess, phi_cal, lambda_tess, lambda_cal
    # Common part of taylor series.
    temp1 = np.cos(c)*np.cos(d)*np.cos(e - f)
    temp2 = np.sin(c)*np.sin(d)
    temp3 = temp1 + temp2
    temp4 = a - b*(temp3)
    temp5 = a*(temp3) - b
    temp6 = np.cos(c)
    a2 = a*a
    a3 = a2 * a
    ell = a2 - 2*a*b*(temp3) + b*b
    ell25 = (ell)**(5/2)

    if is_linear_density:
        if order==0:
            m = temp3
            n = temp6
            if np.abs(temp3-1)<1e-15:
                kernel = temp6 * ( -1/2 * a2 + ( -2 * a * b + ( ( ( a -1 * b ) )**( -1 ) * ( b )**( 3 ) + \
                    ( b )**( 2 ) * ( 5/2 + -3 * np.log( ( b-a ) ) ) ) ) )
            elif np.abs(temp3+1)<1e-15:
                kernel = temp6 * ( -1 * ( b )**( 3 ) * ( ( a + b ) )**( -1 ) + ( -1/2 * ( a -5 * b ) * ( a + b ) + \
                    -3 * ( b )**( 2 ) * np.log( ( a + b ) ) ) )
            else:
                # kernel = 1/2 * ell**( -1/2 ) * temp6 * ( a3 * temp3 + ( a * ( b )**( 2 ) * temp3 * ( 13 + -30 * ( temp3 )**( 2 ) ) + ( a2 * b * ( \
                #     -2 + 5 * ( temp3 )**( 2 ) ) + ( ( b )**( 3 ) * ( -4 + 15 * ( temp3 )**( 2 ) ) + 3 * ( b )**( 2 ) \
                #     * temp3 * ell**( 1/2 ) * ( -3 + 5 * ( temp3 \
                #     )**( 2 ) ) * np.arctanh( ( a + -1 * b * temp3 ) * ell**( -1/2 ) ) ) ) ) )
                kernel = 1/2 * ell**( -1/2 ) * temp6 * ( a3 * temp3 + ( a * ( b )**( 2 ) * temp3 * ( 13 + -30 * ( temp3 )**( 2 ) ) + ( a2 * b * ( \
                    -2 + 5 * ( temp3 )**( 2 ) ) + ( ( b )**( 3 ) * ( -4 + 15 * ( temp3 )**( 2 ) ) + 3 * ( b )**( 2 ) \
                    * temp3 * ell**( 1/2 ) * ( -3 + 5 * ( temp3 \
                    )**( 2 ) ) * np.log(np.sqrt(ell) + a - b * temp3) ) ) ) )
        elif order==1:
            kernel = a3*(temp5)*temp6/(ell)**(3/2)
        elif order==2:
            kernel = a3*(-3*a + 3*b*(temp3))*(temp5)*temp6/ell25 + a3*(temp3)*temp6/(ell)**(3/2) + 3*a2*(temp5)*temp6/(ell)**(3/2)
        elif order==3:
            kernel = 3*a*(-2*a2*(temp4)*(temp3)/(ell) + a2*(temp5)*(5*(temp4)**2/(ell) - 1)/(ell) - 6*a*(temp4)*(temp5)/(ell) + 4*a*(temp3) - 2*b)*temp6/(ell)**(3/2)
        elif order==4:
            kernel = 3*(-5*a3*(temp4)*(temp5)*(7*(temp4)**2/(ell) - 3)/(ell)**2 + 3*a3*(5*(temp4)**2/(ell) - 1)*(temp3)/(ell) - 18*a2*(temp4)*(temp3)/(ell) + 9*a2*(temp5)*(5*(temp4)**2/(ell) - 1)/(ell) - 18*a*(temp4)*(temp5)/(ell) + 8*a*(temp3) - 2*b)*temp6/(ell)**(3/2)
        elif order==5:
            kernel = 3*(-20*a3*(temp4)*(7*(temp4)**2/(ell) - 3)*(temp3)/(ell)**2 + 15*a3*(temp5)*(21*(temp4)**4/(ell)**2 - 14*(temp4)**2/(ell) + 1)/(ell)**2 - 60*a2*(temp4)*(temp5)*(7*(temp4)**2/(ell) - 3)/(ell)**2 + 36*a2*(5*(temp4)**2/(ell) - 1)*(temp3)/(ell) - 72*a*(temp4)*(temp3)/(ell) + 36*a*(temp5)*(5*(temp4)**2/(ell) - 1)/(ell) - 24*(temp4)*(temp5)/(ell) + 8*np.sin(c)*np.sin(d) + 8*temp6*np.cos(d)*np.cos(e - f))*temp6/(ell)**(3/2)
        elif order==6:
            kernel = 45*(-7*a3*(temp4)*(temp5)*(33*(temp4)**4/(ell)**2 - 30*(temp4)**2/(ell) + 5)/(ell)**2 + 5*a3*(temp3)*(21*(temp4)**4/(ell)**2 - 14*(temp4)**2/(ell) + 1)/(ell) - 20*a2*(temp4)*(7*(temp4)**2/(ell) - 3)*(temp3)/(ell) + 15*a2*(temp5)*(21*(temp4)**4/(ell)**2 - 14*(temp4)**2/(ell) + 1)/(ell) - 20*a*(temp4)*(temp5)*(7*(temp4)**2/(ell) - 3)/(ell) + 12*a*(5*(temp4)**2/(ell) - 1)*(temp3) - 8*(temp4)*(temp3) + 4*(temp5)*(5*(temp4)**2/(ell) - 1))*temp6/ell25
        elif order==7:
            kernel = 45*(-42*a3*(temp4)*(temp3)*(33*(temp4)**4/(ell)**2 - 30*(temp4)**2/(ell) + 5)/(ell)**2 + 7*a3*(temp5)*(429*(temp4)**6/(ell)**3 - 495*(temp4)**4/(ell)**2 + 135*(temp4)**2/(ell) - 5)/(ell)**2 - 126*a2*(temp4)*(temp5)*(33*(temp4)**4/(ell)**2 - 30*(temp4)**2/(ell) + 5)/(ell)**2 + 90*a2*(temp3)*(21*(temp4)**4/(ell)**2 - 14*(temp4)**2/(ell) + 1)/(ell) - 120*a*(temp4)*(7*(temp4)**2/(ell) - 3)*(temp3)/(ell) + 90*a*(temp5)*(21*(temp4)**4/(ell)**2 - 14*(temp4)**2/(ell) + 1)/(ell) - 40*(temp4)*(temp5)*(7*(temp4)**2/(ell) - 3)/(ell) + 24*(5*(temp4)**2/(ell) - 1)*(temp3))*temp6/ell25
        elif order==8:
            kernel = 315*(-9*a3*(temp4)*(temp5)*(715*(temp4)**6/(ell)**3 - 1001*(temp4)**4/(ell)**2 + 385*(temp4)**2/(ell) - 35)/(ell)**2 + 7*a3*(temp3)*(429*(temp4)**6/(ell)**3 - 495*(temp4)**4/(ell)**2 + 135*(temp4)**2/(ell) - 5)/(ell) - 126*a2*(temp4)*(temp3)*(33*(temp4)**4/(ell)**2 - 30*(temp4)**2/(ell) + 5)/(ell) + 21*a2*(temp5)*(429*(temp4)**6/(ell)**3 - 495*(temp4)**4/(ell)**2 + 135*(temp4)**2/(ell) - 5)/(ell) - 126*a*(temp4)*(temp5)*(33*(temp4)**4/(ell)**2 - 30*(temp4)**2/(ell) + 5)/(ell) + 90*a*(temp3)*(21*(temp4)**4/(ell)**2 - 14*(temp4)**2/(ell) + 1) - 40*(temp4)*(7*(temp4)**2/(ell) - 3)*(temp3) + 30*(temp5)*(21*(temp4)**4/(ell)**2 - 14*(temp4)**2/(ell) + 1))*temp6/(ell)**(7/2)
        elif order==9:
            kernel = 945*(-24*a3*(temp4)*(temp3)*(715*(temp4)**6/(ell)**3 - 1001*(temp4)**4/(ell)**2 + 385*(temp4)**2/(ell) - 35)/(ell)**2 + 15*a3*(temp5)*(2431*(temp4)**8/(ell)**4 - 4004*(temp4)**6/(ell)**3 + 2002*(temp4)**4/(ell)**2 - 308*(temp4)**2/(ell) + 7)/(ell)**2 - 72*a2*(temp4)*(temp5)*(715*(temp4)**6/(ell)**3 - 1001*(temp4)**4/(ell)**2 + 385*(temp4)**2/(ell) - 35)/(ell)**2 + 56*a2*(temp3)*(429*(temp4)**6/(ell)**3 - 495*(temp4)**4/(ell)**2 + 135*(temp4)**2/(ell) - 5)/(ell) - 336*a*(temp4)*(temp3)*(33*(temp4)**4/(ell)**2 - 30*(temp4)**2/(ell) + 5)/(ell) + 56*a*(temp5)*(429*(temp4)**6/(ell)**3 - 495*(temp4)**4/(ell)**2 + 135*(temp4)**2/(ell) - 5)/(ell) - 112*(temp4)*(temp5)*(33*(temp4)**4/(ell)**2 - 30*(temp4)**2/(ell) + 5)/(ell) + 80*(temp3)*(21*(temp4)**4/(ell)**2 - 14*(temp4)**2/(ell) + 1))*temp6/(ell)**(7/2)
        elif order==10:
            kernel = 2835*(-55*a3*(temp4)*(temp5)*(4199*(temp4)**8/(ell)**4 - 7956*(temp4)**6/(ell)**3 + 4914*(temp4)**4/(ell)**2 - 1092*(temp4)**2/(ell) + 63)/(ell)**2 + 45*a3*(temp3)*(2431*(temp4)**8/(ell)**4 - 4004*(temp4)**6/(ell)**3 + 2002*(temp4)**4/(ell)**2 - 308*(temp4)**2/(ell) + 7)/(ell) - 216*a2*(temp4)*(temp3)*(715*(temp4)**6/(ell)**3 - 1001*(temp4)**4/(ell)**2 + 385*(temp4)**2/(ell) - 35)/(ell) + 135*a2*(temp5)*(2431*(temp4)**8/(ell)**4 - 4004*(temp4)**6/(ell)**3 + 2002*(temp4)**4/(ell)**2 - 308*(temp4)**2/(ell) + 7)/(ell) - 216*a*(temp4)*(temp5)*(715*(temp4)**6/(ell)**3 - 1001*(temp4)**4/(ell)**2 + 385*(temp4)**2/(ell) - 35)/(ell) + 168*a*(temp3)*(429*(temp4)**6/(ell)**3 - 495*(temp4)**4/(ell)**2 + 135*(temp4)**2/(ell) - 5) - 336*(temp4)*(temp3)*(33*(temp4)**4/(ell)**2 - 30*(temp4)**2/(ell) + 5) + 56*(temp5)*(429*(temp4)**6/(ell)**3 - 495*(temp4)**4/(ell)**2 + 135*(temp4)**2/(ell) - 5))*temp6/(ell)**(9/2)
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**3 * sy.cos(c) \
                * (a * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f)) - b) \
                / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**3
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    else:
        if order==0:
            if np.abs(temp3-1)<1e-15:
                kernel = - temp6 * (-b**2 / (a - b) + 2 * b * np.log(b - a) + a)
            elif np.abs(temp3+1)<1e-15:
                kernel = temp6 * (-a + b**2 / (a + b) + 2 * b * np.log(a + b))
            else:
                m = temp3
                n = temp6
                kernel = - temp6 / b * (a**3 / np.sqrt(ell) \
                    - np.sqrt(ell) * (a + 3 * b * temp3) \
                    - b**2 * (3 * temp3**2 - 1) * np.log(np.sqrt(ell) + a - b * temp3))
                # kernel = 1 / np.sqrt(ell) * temp6 * ( ( a )**( 2 ) \
                #     * temp3 + ( 3 * ( b )**( 2 ) * temp3 + ( a * ( b + -6 * b * ( temp3 )**( 2 ) )
                #     + b * np.sqrt(ell) * ( -1 + 3 * ( temp3 )**( 2 ) ) * \
                #     np.arctanh( ( a + -1 * b * temp3 ) / np.sqrt(ell) ) ) ) )
        elif order==1:
            kernel = a2*(temp5)*temp6/(ell)**(3/2)
        elif order==2:
            kernel = a2*(-3*a + 3*b*(temp3))*(temp5)*temp6/ell25 + a2*(temp3)*temp6/(ell)**(3/2) + 2*a*(temp5)*temp6/(ell)**(3/2)
        elif order==3:
            kernel = (-6*a2*(temp4)*(temp3)/(ell) + 3*a2*(temp5)*(5*(temp4)**2/(ell) - 1)/(ell) - 12*a*(temp4)*(temp5)/(ell) + 6*a*(temp3) - 2*b)*temp6/(ell)**(3/2)
        elif order==4:
            kernel = 3*(-5*a2*(temp4)*(temp5)*(7*(temp4)**2/(ell) - 3)/(ell)**2 + 3*a2*(5*(temp4)**2/(ell) - 1)*(temp3)/(ell) - 12*a*(temp4)*(temp3)/(ell) + 6*a*(temp5)*(5*(temp4)**2/(ell) - 1)/(ell) - 6*(temp4)*(temp5)/(ell) + 2*np.sin(c)*np.sin(d) + 2*temp6*np.cos(d)*np.cos(e - f))*temp6/(ell)**(3/2)
        elif order==5:
            kernel = 3*(-20*a2*(temp4)*(7*(temp4)**2/(ell) - 3)*(temp3)/(ell) + 15*a2*(temp5)*(21*(temp4)**4/(ell)**2 - 14*(temp4)**2/(ell) + 1)/(ell) - 40*a*(temp4)*(temp5)*(7*(temp4)**2/(ell) - 3)/(ell) + 24*a*(5*(temp4)**2/(ell) - 1)*(temp3) - 24*(temp4)*(temp3) + 12*(temp5)*(5*(temp4)**2/(ell) - 1))*temp6/ell25
        elif order==6:
            kernel = 15*(-21*a2*(temp4)*(temp5)*(33*(temp4)**4/(ell)**2 - 30*(temp4)**2/(ell) + 5)/(ell)**2 + 15*a2*(temp3)*(21*(temp4)**4/(ell)**2 - 14*(temp4)**2/(ell) + 1)/(ell) - 40*a*(temp4)*(7*(temp4)**2/(ell) - 3)*(temp3)/(ell) + 30*a*(temp5)*(21*(temp4)**4/(ell)**2 - 14*(temp4)**2/(ell) + 1)/(ell) - 20*(temp4)*(temp5)*(7*(temp4)**2/(ell) - 3)/(ell) + 12*(5*(temp4)**2/(ell) - 1)*(temp3))*temp6/ell25
        elif order==7:
            kernel = 45*(-42*a2*(temp4)*(temp3)*(33*(temp4)**4/(ell)**2 - 30*(temp4)**2/(ell) + 5)/(ell) + 7*a2*(temp5)*(429*(temp4)**6/(ell)**3 - 495*(temp4)**4/(ell)**2 + 135*(temp4)**2/(ell) - 5)/(ell) - 84*a*(temp4)*(temp5)*(33*(temp4)**4/(ell)**2 - 30*(temp4)**2/(ell) + 5)/(ell) + 60*a*(temp3)*(21*(temp4)**4/(ell)**2 - 14*(temp4)**2/(ell) + 1) - 40*(temp4)*(7*(temp4)**2/(ell) - 3)*(temp3) + 30*(temp5)*(21*(temp4)**4/(ell)**2 - 14*(temp4)**2/(ell) + 1))*temp6/(ell)**(7/2)
        elif order==8:
            kernel = 315*(-9*a2*(temp4)*(temp5)*(715*(temp4)**6/(ell)**3 - 1001*(temp4)**4/(ell)**2 + 385*(temp4)**2/(ell) - 35)/(ell)**2 + 7*a2*(temp3)*(429*(temp4)**6/(ell)**3 - 495*(temp4)**4/(ell)**2 + 135*(temp4)**2/(ell) - 5)/(ell) - 84*a*(temp4)*(temp3)*(33*(temp4)**4/(ell)**2 - 30*(temp4)**2/(ell) + 5)/(ell) + 14*a*(temp5)*(429*(temp4)**6/(ell)**3 - 495*(temp4)**4/(ell)**2 + 135*(temp4)**2/(ell) - 5)/(ell) - 42*(temp4)*(temp5)*(33*(temp4)**4/(ell)**2 - 30*(temp4)**2/(ell) + 5)/(ell) + 30*(temp3)*(21*(temp4)**4/(ell)**2 - 14*(temp4)**2/(ell) + 1))*temp6/(ell)**(7/2)
        elif order==9:
            kernel = 315*(-72*a2*(temp4)*(temp3)*(715*(temp4)**6/(ell)**3 - 1001*(temp4)**4/(ell)**2 + 385*(temp4)**2/(ell) - 35)/(ell) + 45*a2*(temp5)*(2431*(temp4)**8/(ell)**4 - 4004*(temp4)**6/(ell)**3 + 2002*(temp4)**4/(ell)**2 - 308*(temp4)**2/(ell) + 7)/(ell) - 144*a*(temp4)*(temp5)*(715*(temp4)**6/(ell)**3 - 1001*(temp4)**4/(ell)**2 + 385*(temp4)**2/(ell) - 35)/(ell) + 112*a*(temp3)*(429*(temp4)**6/(ell)**3 - 495*(temp4)**4/(ell)**2 + 135*(temp4)**2/(ell) - 5) - 336*(temp4)*(temp3)*(33*(temp4)**4/(ell)**2 - 30*(temp4)**2/(ell) + 5) + 56*(temp5)*(429*(temp4)**6/(ell)**3 - 495*(temp4)**4/(ell)**2 + 135*(temp4)**2/(ell) - 5))*temp6/(ell)**(9/2)
        elif order==10:
            kernel =2835*(-55*a2*(temp4)*(temp5)*(4199*(temp4)**8/(ell)**4 - 7956*(temp4)**6/(ell)**3 + 4914*(temp4)**4/(ell)**2 - 1092*(temp4)**2/(ell) + 63)/(ell)**2 + 45*a2*(temp3)*(2431*(temp4)**8/(ell)**4 - 4004*(temp4)**6/(ell)**3 + 2002*(temp4)**4/(ell)**2 - 308*(temp4)**2/(ell) + 7)/(ell) - 144*a*(temp4)*(temp3)*(715*(temp4)**6/(ell)**3 - 1001*(temp4)**4/(ell)**2 + 385*(temp4)**2/(ell) - 35)/(ell) + 90*a*(temp5)*(2431*(temp4)**8/(ell)**4 - 4004*(temp4)**6/(ell)**3 + 2002*(temp4)**4/(ell)**2 - 308*(temp4)**2/(ell) + 7)/(ell) - 72*(temp4)*(temp5)*(715*(temp4)**6/(ell)**3 - 1001*(temp4)**4/(ell)**2 + 385*(temp4)**2/(ell) - 35)/(ell) + 56*(temp3)*(429*(temp4)**6/(ell)**3 - 495*(temp4)**4/(ell)**2 + 135*(temp4)**2/(ell) - 5))*temp6/(ell)**(9/2)
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**2 * sy.cos(c) \
                * (a * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f)) - b) \
                / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**3
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    
    return kernel


def cal_Vxx_kernel(r_cal, phi_cal, lambda_cal, \
        r_tess, phi_tess, lambda_tess, \
        order, is_linear_density): 
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
    order: int
        Differentiation of kernel function.
    is_linear_density: bool
        If the tesseroid have linear varying density.
        
    Returns
    -------
    kernel: float
        Kernel of gravitational gradient Vxx.
    """
    a, b, c, d, e, f = r_tess, r_cal, phi_tess, phi_cal, lambda_tess, lambda_cal
    # Common part of taylor series.
    n = np.cos(phi_tess)
    g = np.cos(phi_cal) * np.sin(phi_tess) - np.sin(phi_cal) * n\
        * np.cos(lambda_tess - lambda_cal)
    h = n * np.sin(lambda_tess - lambda_cal)
    m = np.sin(phi_cal) * np.sin(phi_tess) + n * np.cos(phi_cal)\
        * np.cos(lambda_tess - lambda_cal)
    ell = (a**2 + b**2 - 2 * a * b * m)**0.5
    tmp1 = 2 * a - 2 * b * m

    if is_linear_density:
        if order == 0:
            if np.abs(m - 1) < 1e-15:
                kernel = n * ( 1/4 * ( ( a + -1 * b ) )**( -4 ) * ( 28 * ( a )**( 3 ) * ( b \
                    )**( 2 ) + ( ( b )**( 5 ) * ( 6 + -65 * ( g )**( 2 ) ) + ( ( a )**( 5 \
                    ) * ( 4 + -12 * ( g )**( 2 ) ) + ( 20 * ( a )**( 4 ) * b * ( -1 + 3 * \
                    ( g )**( 2 ) ) + ( -6 * ( a )**( 2 ) * ( b )**( 3 ) * ( 1 + 30 * ( g \
                    )**( 2 ) ) + 4 * a * ( b )**( 4 ) * ( -3 + 50 * ( g )**( 2 ) ) ) ) ) \
                    ) ) + 3 * b * ( 1 + -5 * ( g )**( 2 ) ) * np.log( ( a + -1 * b ) ) \
                    )
            elif np.abs(m + 1) < 1e-15:
                kernel = n * ( 3/4 * ( b )**( 5 ) * ( ( a + b ) )**( -4 ) * ( g )**( 2 ) + ( \
                    -5 * ( b )**( 4 ) * ( ( a + b ) )**( -3 ) * ( g )**( 2 ) + ( 3 * ( b \
                    )**( 2 ) * ( ( a + b ) )**( -1 ) * ( 1 + -10 * ( g )**( 2 ) ) + ( a * \
                    ( -1 + 3 * ( g )**( 2 ) ) + ( 1/2 * ( b )**( 3 ) * ( ( a + b ) )**( \
                    -2 ) * ( -1 + 30 * ( g )**( 2 ) ) + -3 * b * ( -1 + 5 * ( g )**( 2 ) \
                    ) * np.log( ( a + b ) ) ) ) ) ) )
            else:
                kernel = n * ( (ell)**(-3) * ( ( -1 + ( m )**( 2 ) ) )**( -2 ) * ( ( a )**( 4 ) * ( -1 + 3 * ( g )**( 2 ) ) * ( ( -1 + ( m )**( 2 ) ) )**( 2 ) + ( ( a )**( 3 ) * b * m * ( 7 + ( -15 * ( m )**( 2 ) + ( 8 * ( m )**( 4 ) + ( g )**( 2 ) * ( -32 + ( 74 * ( m )**( 2 ) + -40 * ( m )**( 4 ) ) ) ) ) ) + ( ( b )**( 4 ) * ( -2 + ( 5 * ( m )**( 2 ) + ( -3 * ( m )**( 4 ) + ( g )**( 2 ) * ( 8 + ( -25 * ( m )**( 2 ) + 15 * ( m )**( 4 ) ) ) ) ) ) + ( -3 * a * ( b )**( 3 ) * m * ( -3 + ( 7 * ( m )**( 2 ) + ( -4 * ( m )**( 4 ) + ( g )**( 2 ) * ( 13 + ( -35 * ( m )**( 2 ) + 20 * ( m )**( 4 ) ) ) ) ) ) + 3 * ( a )**( 2 ) * ( b )**( 2 ) * ( -1 + ( -1 * ( m )**( 2 ) + ( 6 * ( m )**( 4 ) + ( -4 * ( m )**( 6 ) + ( g )**( 2 ) * ( 4 + ( 4 * ( m )**( 2 ) + ( -30 * ( m )**( 4 ) + 20 * ( m )**( 6 ) ) ) ) ) ) ) ) ) ) ) ) + -3 * b * ( -1 + 5 * ( g )**( 2 ) ) * m * np.log( ( -1 * a + ( b * m + ( ( ( a )**( 2 ) + ( ( b )**( 2 ) + -2 * a * b * m ) ) )**( 1/2 ) ) ) ) )
        elif order == 1:
            kernel = ( a )**( 3 ) * ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * (ell)**(-5) + -1 * (ell)**(-3) ) * n
        elif order == 2:
            kernel = ( ( a )**( 3 ) * ( -15/2 * ( a )**( 2 ) * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 6 * a * ( g )**( 2 ) * (ell)**(-5) + 3/2 * tmp1 * (ell)**(-5) ) ) * n + 3 * ( a )**( 2 ) * ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * (ell)**(-5) + -1 * (ell)**(-3) ) * n )
        elif order == 3:
            kernel = ( 6 * ( a )**( 2 ) * ( -15/2 * ( a )**( 2 ) * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 6 * a * ( g )**( 2 ) * (ell)**(-5) + 3/2 * tmp1 * (ell)**(-5) ) ) + ( 6 * a * ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * (ell)**(-5) + -1 * (ell)**(-3) ) + ( a )**( 3 ) * ( -30 * a * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( -15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + ( 3 * (ell)**(-5) + ( 6 * ( g )**( 2 ) * (ell)**(-5) + 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) * n
        elif order == 4:
            kernel = ( 18 * a * ( -15/2 * ( a )**( 2 ) * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 6 * a * ( g )**( 2 ) * (ell)**(-5) + 3/2 * tmp1 * (ell)**(-5) ) ) + ( 6 * ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * (ell)**(-5) + -1 * (ell)**(-3) ) + ( ( a )**( 3 ) * ( 105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + ( -45/2 * tmp1 * (ell)**(-7) + ( -45 * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 18 * a * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) + 9 * ( a )**( 2 ) * ( -30 * a * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( -15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + ( 3 * (ell)**(-5) + ( 6 * ( g )**( 2 ) * (ell)**(-5) + 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) ) * n
        elif order == 5:
            kernel = ( 24 * ( -15/2 * ( a )**( 2 ) * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 6 * a * ( g )**( 2 ) * (ell)**(-5) + 3/2 * tmp1 * (ell)**(-5) ) ) + ( ( a )**( 3 ) * ( -945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( 315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + ( -45 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + ( 24 * a * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 36 * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) + ( 12 * ( a )**( 2 ) * ( 105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + ( -45/2 * tmp1 * (ell)**(-7) + ( -45 * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 18 * a * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) + 36 * a * ( -30 * a * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( -15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + ( 3 * (ell)**(-5) + ( 6 * ( g )**( 2 ) * (ell)**(-5) + 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) ) * n
        elif order == 6:
            kernel = ( ( a )**( 3 ) * ( 10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( -4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + ( 1575/2 * tmp1 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + ( 30 * a * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 60 * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) ) ) ) ) + ( 15 * ( a )**( 2 ) * ( -945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( 315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + ( -45 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + ( 24 * a * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 36 * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) + ( 60 * a * ( 105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + ( -45/2 * tmp1 * (ell)**(-7) + ( -45 * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 18 * a * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) + 60 * ( -30 * a * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( -15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + ( 3 * (ell)**(-5) + ( 6 * ( g )**( 2 ) * (ell)**(-5) + 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) ) * n
        elif order == 7:
            kernel = ( ( a )**( 3 ) * ( -135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( 155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + ( 1575 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + ( 36 * a * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 90 * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) ) ) ) ) ) + ( 18 * ( a )**( 2 ) * ( 10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( -4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + ( 1575/2 * tmp1 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + ( 30 * a * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 60 * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) ) ) ) ) + ( 90 * a * ( -945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( 315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + ( -45 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + ( 24 * a * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 36 * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) + 120 * ( 105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + ( -45/2 * tmp1 * (ell)**(-7) + ( -45 * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 18 * a * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) ) * n
        elif order == 8:
            kernel = ( ( a )**( 3 ) * ( 2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( -2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + ( -99225/2 * tmp1 * (ell)**(-11) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + ( 42 * a * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 126 * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) ) ) ) ) ) + ( 21 * ( a )**( 2 ) * ( -135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( 155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + ( 1575 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + ( 36 * a * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 90 * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) ) ) ) ) ) + ( 126 * a * ( 10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( -4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + ( 1575/2 * tmp1 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + ( 30 * a * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 60 * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) ) ) ) ) + 210 * ( -945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( 315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + ( -45 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + ( 24 * a * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 36 * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) ) ) * n
        elif order == 9:
            kernel = ( ( a )**( 3 ) * ( -34459425/256 * ( tmp1 )**( 8 ) * (ell)**(-19) + ( 14189175/16 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -14189175/8 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 1091475 * ( tmp1 )**( 2 ) * (ell)**(-13) + ( -99225 * (ell)**(-11) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) + ( 48 * a * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + 168 * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) ) ) ) ) ) ) ) + ( 24 * ( a )**( 2 ) * ( 2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( -2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + ( -99225/2 * tmp1 * (ell)**(-11) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + ( 42 * a * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 126 * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) ) ) ) ) ) + ( 168 * a * ( -135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( 155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + ( 1575 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + ( 36 * a * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 90 * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) ) ) ) ) ) + 336 * ( 10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( -4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + ( 1575/2 * tmp1 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + ( 30 * a * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 60 * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) ) ) ) ) ) ) ) * n
        elif order == 10:
            kernel = ( ( a )**( 3 ) * ( 654729075/512 * ( tmp1 )**( 9 ) * (ell)**(-21) + ( -310134825/32 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 383107725/16 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -42567525/2 * ( tmp1 )**( 3 ) * (ell)**(-15) + ( 9823275/2 * tmp1 * (ell)**(-13) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -4583103525/512 * ( tmp1 )**( 9 ) * (ell)**(-23) + ( 1964187225/32 * ( tmp1 )**( 7 ) * (ell)**(-21) + ( -2170943775/16 * ( tmp1 )**( 5 ) * (ell)**(-19) + ( 212837625/2 * ( tmp1 )**( 3 ) * (ell)**(-17) + -42567525/2 * tmp1 * (ell)**(-15) ) ) ) ) + ( 54 * a * ( g )**( 2 ) * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) + 216 * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) ) ) ) ) ) ) ) + ( 27 * ( a )**( 2 ) * ( -34459425/256 * ( tmp1 )**( 8 ) * (ell)**(-19) + ( 14189175/16 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -14189175/8 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 1091475 * ( tmp1 )**( 2 ) * (ell)**(-13) + ( -99225 * (ell)**(-11) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) + ( 48 * a * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + 168 * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) ) ) ) ) ) ) ) + ( 216 * a * ( 2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( -2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + ( -99225/2 * tmp1 * (ell)**(-11) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + ( 42 * a * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 126 * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) ) ) ) ) ) + 504 * ( -135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( 155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + ( 1575 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + ( 36 * a * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 90 * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) ) ) ) ) ) ) ) ) * n
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**3 * sy.cos(c) \
                * (3 * (a*(sy.cos(d)*sy.sin(c)-sy.sin(d)*sy.cos(c)*sy.cos(e-f)))**2 \
                / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**5 \
                - 1 / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**3)
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    else:
        if order == 0:
            if np.abs(m - 1) < 1e-15:
                kernel = n * ( 1/4 * ( ( a + -1 * b ) )**( -4 ) * b * ( 2 * ( a )**( 2 ) * b * \
                    ( 11 + -54 * ( g )**( 2 ) ) + ( ( b )**( 3 ) * ( 6 + -25 * ( g )**( 2 \
                    ) ) + ( 8 * ( a )**( 3 ) * ( -1 + 6 * ( g )**( 2 ) ) + 4 * a * ( b \
                    )**( 2 ) * ( -5 + 22 * ( g )**( 2 ) ) ) ) ) + ( 1 + -3 * ( g )**( 2 ) \
                    ) * np.log( ( a + -1 * b ) ) )
            elif np.abs(m + 1) < 1e-15:
                kernel = n * ( 1/4 * b * ( ( a + b ) )**( -4 ) * ( 8 * ( a )**( 3 ) * ( -1 + 6 \
                    * ( g )**( 2 ) ) + ( 4 * a * ( b )**( 2 ) * ( -5 + 22 * ( g )**( 2 ) \
                    ) + ( ( b )**( 3 ) * ( -6 + 25 * ( g )**( 2 ) ) + 2 * ( a )**( 2 ) * \
                    b * ( -11 + 54 * ( g )**( 2 ) ) ) ) ) + ( -1 + 3 * ( g )**( 2 ) ) * \
                    np.log( ( a + b ) ) )
            else:
                kernel = n * ( (ell)**(-3) * ( ( -1 + ( m )**( 2 ) ) )**( -2 ) * ( ( a )**( 2 ) * b * m * ( -1 + ( ( 5 + -18 * ( g )**( 2 ) ) * ( m )**( 2 ) + 4 * ( -1 + 3 * ( g )**( 2 ) ) * ( m )**( 4 ) ) ) + ( ( b )**( 3 ) * m * ( 1 + ( -1 * ( m )**( 2 ) + ( g )**( 2 ) * ( -5 + 3 * ( m )**( 2 ) ) ) ) + ( a * ( b )**( 2 ) * ( 1 + ( -5 * ( m )**( 2 ) + ( 4 * ( m )**( 4 ) + -3 * ( g )**( 2 ) * ( 1 + ( -7 * ( m )**( 2 ) + 4 * ( m )**( 4 ) ) ) ) ) ) + ( a )**( 3 ) * ( 1 + ( -3 * ( m )**( 2 ) + ( 2 * ( m )**( 4 ) + -2 * ( g )**( 2 ) * ( 2 + ( -7 * ( m )**( 2 ) + 4 * ( m )**( 4 ) ) ) ) ) ) ) ) ) + ( 1 + -3 * ( g )**( 2 ) ) * np.log( ( -1 * a + ( b * m + ( ( ( a )**( 2 ) + ( ( b )**( 2 ) + -2 * a * b * m ) ) )**( 1/2 ) ) ) ) )
        elif order == 1:
            kernel = ( a )**( 2 ) * ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * (ell)**(-5) + -1 * (ell)**(-3) ) * n
        elif order == 2:
            kernel = ( ( a )**( 2 ) * ( -15/2 * ( a )**( 2 ) * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 6 * a * ( g )**( 2 ) * (ell)**(-5) + 3/2 * tmp1 * (ell)**(-5) ) ) * n + 2 * a * ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * (ell)**(-5) + -1 * (ell)**(-3) ) * n )
        elif order == 3:
            kernel = ( 4 * a * ( -15/2 * ( a )**( 2 ) * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 6 * a * ( g )**( 2 ) * (ell)**(-5) + 3/2 * tmp1 * (ell)**(-5) ) ) + ( 2 * ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * (ell)**(-5) + -1 * (ell)**(-3) ) + ( a )**( 2 ) * ( -30 * a * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( -15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + ( 3 * (ell)**(-5) + ( 6 * ( g )**( 2 ) * (ell)**(-5) + 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) * n
        elif order == 4:
            kernel = ( 6 * ( -15/2 * ( a )**( 2 ) * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 6 * a * ( g )**( 2 ) * (ell)**(-5) + 3/2 * tmp1 * (ell)**(-5) ) ) + ( ( a )**( 2 ) * ( 105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + ( -45/2 * tmp1 * (ell)**(-7) + ( -45 * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 18 * a * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) + 6 * a * ( -30 * a * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( -15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + ( 3 * (ell)**(-5) + ( 6 * ( g )**( 2 ) * (ell)**(-5) + 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) * n
        elif order == 5:
            kernel = ( ( a )**( 2 ) * ( -945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( 315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + ( -45 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + ( 24 * a * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 36 * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) + ( 8 * a * ( 105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + ( -45/2 * tmp1 * (ell)**(-7) + ( -45 * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 18 * a * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) + 12 * ( -30 * a * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( -15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + ( 3 * (ell)**(-5) + ( 6 * ( g )**( 2 ) * (ell)**(-5) + 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) * n
        elif order == 6:
            kernel = ( ( a )**( 2 ) * ( 10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( -4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + ( 1575/2 * tmp1 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + ( 30 * a * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 60 * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) ) ) ) ) + ( 10 * a * ( -945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( 315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + ( -45 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + ( 24 * a * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 36 * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) + 20 * ( 105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + ( -45/2 * tmp1 * (ell)**(-7) + ( -45 * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 18 * a * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) * n
        elif order == 7:
            kernel = ( ( a )**( 2 ) * ( -135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( 155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + ( 1575 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + ( 36 * a * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 90 * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) ) ) ) ) ) + ( 12 * a * ( 10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( -4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + ( 1575/2 * tmp1 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + ( 30 * a * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 60 * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) ) ) ) ) + 30 * ( -945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( 315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + ( -45 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + ( 24 * a * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 36 * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) ) * n
        elif order == 8:
            kernel = ( ( a )**( 2 ) * ( 2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( -2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + ( -99225/2 * tmp1 * (ell)**(-11) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + ( 42 * a * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 126 * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) ) ) ) ) ) + ( 14 * a * ( -135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( 155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + ( 1575 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + ( 36 * a * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 90 * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) ) ) ) ) ) + 42 * ( 10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( -4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + ( 1575/2 * tmp1 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + ( 30 * a * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 60 * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) ) ) ) ) ) ) * n
        elif order == 9:
            kernel = ( ( a )**( 2 ) * ( -34459425/256 * ( tmp1 )**( 8 ) * (ell)**(-19) + ( 14189175/16 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -14189175/8 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 1091475 * ( tmp1 )**( 2 ) * (ell)**(-13) + ( -99225 * (ell)**(-11) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) + ( 48 * a * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + 168 * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) ) ) ) ) ) ) ) + ( 16 * a * ( 2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( -2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + ( -99225/2 * tmp1 * (ell)**(-11) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + ( 42 * a * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 126 * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) ) ) ) ) ) + 56 * ( -135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( 155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + ( 1575 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + ( 36 * a * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 90 * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) ) ) ) ) ) ) ) * n
        elif order == 10:
            kernel = ( ( a )**( 2 ) * ( 654729075/512 * ( tmp1 )**( 9 ) * (ell)**(-21) + ( -310134825/32 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 383107725/16 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -42567525/2 * ( tmp1 )**( 3 ) * (ell)**(-15) + ( 9823275/2 * tmp1 * (ell)**(-13) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -4583103525/512 * ( tmp1 )**( 9 ) * (ell)**(-23) + ( 1964187225/32 * ( tmp1 )**( 7 ) * (ell)**(-21) + ( -2170943775/16 * ( tmp1 )**( 5 ) * (ell)**(-19) + ( 212837625/2 * ( tmp1 )**( 3 ) * (ell)**(-17) + -42567525/2 * tmp1 * (ell)**(-15) ) ) ) ) + ( 54 * a * ( g )**( 2 ) * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) + 216 * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) ) ) ) ) ) ) ) + ( 18 * a * ( -34459425/256 * ( tmp1 )**( 8 ) * (ell)**(-19) + ( 14189175/16 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -14189175/8 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 1091475 * ( tmp1 )**( 2 ) * (ell)**(-13) + ( -99225 * (ell)**(-11) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) + ( 48 * a * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + 168 * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) ) ) ) ) ) ) ) + 72 * ( 2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( -2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + ( -99225/2 * tmp1 * (ell)**(-11) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + ( 42 * a * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 126 * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) ) ) ) ) ) ) ) * n
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**2 * sy.cos(c) \
                * (3 * (a*(sy.cos(d)*sy.sin(c)-sy.sin(d)*sy.cos(c)*sy.cos(e-f)))**2 \
                / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**5 \
                - 1 / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**3)
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    return kernel


def cal_Vxy_kernel(r_cal, phi_cal, lambda_cal, \
        r_tess, phi_tess, lambda_tess, \
        order, is_linear_density):
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
    order: int
        Differentiation of kernel function.
    is_linear_density: bool
        If the tesseroid have linear varying density.
        
    Returns
    -------
    kernel: float
        Kernel of gravitational gradient Vxy.
    """
    a, b, c, d, e, f = r_tess, r_cal, phi_tess, phi_cal, lambda_tess, lambda_cal
    # Common part of taylor series.
    n = np.cos(phi_tess)
    g = np.cos(phi_cal) * np.sin(phi_tess) - np.sin(phi_cal) * n\
        * np.cos(lambda_tess - lambda_cal)
    h = n * np.sin(lambda_tess - lambda_cal)
    p = g * h
    m = np.sin(phi_cal) * np.sin(phi_tess) + n * np.cos(phi_cal)\
        * np.cos(lambda_tess - lambda_cal)
    ell = (a**2 + b**2 - 2 * a * b * m)**0.5
    tmp1 = 2 * a - 2 * b * m

    if is_linear_density:
        if order == 0:
            if np.abs(m - 1) < 1e-15:
                kernel = 3 * g * h * n * ( 1/12 * ( ( a + -1 * b ) )**( -4 ) * ( -12 * ( a \
                    )**( 5 ) + ( 48 * ( a )**( 4 ) * b + ( 48 * ( a )**( 3 ) * ( b )**( 2 \
                    ) + ( -252 * ( a )**( 2 ) * ( b )**( 3 ) + ( 248 * a * ( b )**( 4 ) + \
                    -77 * ( b )**( 5 ) ) ) ) ) ) + -5 * b * np.log( ( a + -1 * b ) ) )
            elif np.abs(m + 1) < 1e-15:
                kernel = 3 * g * h * n * ( a + ( -1/12 * ( b )**( 2 ) * ( ( a + b ) )**( -4 ) \
                    * ( 120 * ( a )**( 3 ) + ( 300 * ( a )**( 2 ) * b + ( 260 * a * ( b \
                    )**( 2 ) + 77 * ( b )**( 3 ) ) ) ) + -5 * b * np.log( ( a + b ) ) \
                    ) )
            else:
                kernel = 3 * g * h * n * ( 1/3 * (ell)**(-3) * ( ( -1 + ( m )**( 2 ) ) )**( -2 ) * ( 3 * ( a )**( 4 ) * ( ( -1 + ( m )**( 2 ) ) )**( 2 ) + ( ( b )**( 4 ) * ( 8 + ( -25 * ( m )**( 2 ) + 15 * ( m )**( 4 ) ) ) + ( -2 * ( a )**( 3 ) * b * m * ( 16 + ( -37 * ( m )**( 2 ) + 20 * ( m )**( 4 ) ) ) + ( -3 * a * ( b )**( 3 ) * m * ( 13 + ( -35 * ( m )**( 2 ) + 20 * ( m )**( 4 ) ) ) + 6 * ( a )**( 2 ) * ( b )**( 2 ) * ( 2 + ( 2 * ( m )**( 2 ) + ( -15 * ( m )**( 4 ) + 10 * ( m )**( 6 ) ) ) ) ) ) ) ) + -5 * b * m * np.log( ( -1 * a + ( b * m + ( ( ( a )**( 2 ) + ( ( b )**( 2 ) + -2 * a * b * m ) ) )**( 1/2 ) ) ) ) )
        elif order == 1:
            kernel = 3 * ( a )**( 5 ) * g * h * (ell)**(-5) * n
        elif order == 2:
            kernel = ( -15/2 * ( a )**( 5 ) * g * h * tmp1 * (ell)**(-7) * n + 15 * ( a )**( 4 ) * g * h * (ell)**(-5) * n )
        elif order == 3:
            kernel = ( -75 * ( a )**( 4 ) * g * h * tmp1 * (ell)**(-7) * n + ( 60 * ( a )**( 3 ) * g * h * (ell)**(-5) * n + 3 * ( a )**( 5 ) * g * h * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) * n ) )
        elif order == 4:
            kernel = ( -450 * ( a )**( 3 ) * g * h * tmp1 * (ell)**(-7) * n + ( 180 * ( a )**( 2 ) * g * h * (ell)**(-5) * n + ( 3 * ( a )**( 5 ) * g * h * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) * n + 45 * ( a )**( 4 ) * g * h * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) * n ) ) )
        elif order == 5:
            kernel = ( -1800 * ( a )**( 2 ) * g * h * tmp1 * (ell)**(-7) * n + ( 360 * a * g * h * (ell)**(-5) * n + ( 3 * ( a )**( 5 ) * g * h * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) * n + ( 60 * ( a )**( 4 ) * g * h * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) * n + 360 * ( a )**( 3 ) * g * h * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) * n ) ) ) )
        elif order == 6:
            kernel = ( -4500 * a * g * h * tmp1 * (ell)**(-7) * n + ( 360 * g * h * (ell)**(-5) * n + ( 3 * ( a )**( 5 ) * g * h * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) * n + ( 75 * ( a )**( 4 ) * g * h * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) * n + ( 600 * ( a )**( 3 ) * g * h * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) * n + 1800 * ( a )**( 2 ) * g * h * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) * n ) ) ) ) )
        elif order == 7:
            kernel = ( -5400 * g * h * tmp1 * (ell)**(-7) * n + ( 3 * ( a )**( 5 ) * g * h * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) * n + ( 90 * ( a )**( 4 ) * g * h * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) * n + ( 900 * ( a )**( 3 ) * g * h * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) * n + ( 3600 * ( a )**( 2 ) * g * h * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) * n + 5400 * a * g * h * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) * n ) ) ) ) )
        elif order == 8:
            kernel = ( 3 * ( a )**( 5 ) * g * h * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) * n + ( 105 * ( a )**( 4 ) * g * h * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) * n + ( 1260 * ( a )**( 3 ) * g * h * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) * n + ( 6300 * ( a )**( 2 ) * g * h * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) * n + ( 12600 * a * g * h * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) * n + 7560 * g * h * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) * n ) ) ) ) )
        elif order == 9:
            kernel = ( 3 * ( a )**( 5 ) * g * h * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) * n + ( 120 * ( a )**( 4 ) * g * h * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) * n + ( 1680 * ( a )**( 3 ) * g * h * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) * n + ( 10080 * ( a )**( 2 ) * g * h * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) * n + ( 25200 * a * g * h * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) * n + 20160 * g * h * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) * n ) ) ) ) )
        elif order == 10:
            kernel = ( 3 * ( a )**( 5 ) * g * h * ( -4583103525/512 * ( tmp1 )**( 9 ) * (ell)**(-23) + ( 1964187225/32 * ( tmp1 )**( 7 ) * (ell)**(-21) + ( -2170943775/16 * ( tmp1 )**( 5 ) * (ell)**(-19) + ( 212837625/2 * ( tmp1 )**( 3 ) * (ell)**(-17) + -42567525/2 * tmp1 * (ell)**(-15) ) ) ) ) * n + ( 135 * ( a )**( 4 ) * g * h * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) * n + ( 2160 * ( a )**( 3 ) * g * h * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) * n + ( 15120 * ( a )**( 2 ) * g * h * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) * n + ( 45360 * a * g * h * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) * n + 45360 * g * h * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) * n ) ) ) ) )
        else:
                a, b, c, d, e, f = sy.symbols('a b c d e f')
                kernel = a**3 * sy.cos(c) \
                    * (3*(a*(sy.cos(d)*sy.sin(c)-sy.sin(d)*sy.cos(c)*sy.cos(e-f))) \
                    * (a*sy.cos(c)*sy.sin(e-f))
                    / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                    * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**5)
                d_kernel = sy.diff(kernel, a, order-1)
                kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                    d: phi_cal, e: lambda_tess, f: lambda_cal}))
    else:
        if order == 0:
            if np.abs(m - 1) < 1e-15:
                kernel = 3 * g * h * n * ( 1/12 * ( ( a + -1 * b ) )**( -4 ) * b * ( 48 * ( a \
                    )**( 3 ) + ( -108 * ( a )**( 2 ) * b + ( 88 * a * ( b )**( 2 ) + -25 \
                    * ( b )**( 3 ) ) ) ) + -1 * np.log( ( -1 * a + b ) ) )
            elif np.abs(m + 1) < 1e-15:
                kernel = 3 * g * h * n * ( 1/12 * b * ( ( a + b ) )**( -4 ) * ( 48 * ( a )**( \
                    3 ) + ( 108 * ( a )**( 2 ) * b + ( 88 * a * ( b )**( 2 ) + 25 * ( b \
                    )**( 3 ) ) ) ) + np.log( ( a + b ) ) )
            else:
                kernel = 3 * g * h * n * ( 1/3 * (ell)**(-3) * ( ( -1 + ( m )**( 2 ) ) )**( -2 ) * ( 6 * ( a )**( 2 ) * b * ( m )**( 3 ) * ( -3 + 2 * ( m )**( 2 ) ) + ( ( b )**( 3 ) * m * ( -5 + 3 * ( m )**( 2 ) ) + ( -3 * a * ( b )**( 2 ) * ( 1 + ( -7 * ( m )**( 2 ) + 4 * ( m )**( 4 ) ) ) + -2 * ( a )**( 3 ) * ( 2 + ( -7 * ( m )**( 2 ) + 4 * ( m )**( 4 ) ) ) ) ) ) + -1 * np.log( ( -1 * a + ( b * m + ( ( ( a )**( 2 ) + ( ( b )**( 2 ) + -2 * a * b * m ) ) )**( 1/2 ) ) ) ) )
        elif order == 1:
            kernel = 3 * ( a )**( 4 ) * g * h * (ell)**(-5) * n
        elif order == 2:
            kernel = ( -15/2 * ( a )**( 4 ) * g * h * tmp1 * (ell)**(-7) * n + 12 * ( a )**( 3 ) * g * h * (ell)**(-5) * n )
        elif order == 3:
            kernel = ( -60 * ( a )**( 3 ) * g * h * tmp1 * (ell)**(-7) * n + ( 36 * ( a )**( 2 ) * g * h * (ell)**(-5) * n + 3 * ( a )**( 4 ) * g * h * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) * n ) )
        elif order == 4:
            kernel = ( -270 * ( a )**( 2 ) * g * h * tmp1 * (ell)**(-7) * n + ( 72 * a * g * h * (ell)**(-5) * n + ( 3 * ( a )**( 4 ) * g * h * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) * n + 36 * ( a )**( 3 ) * g * h * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) * n ) ) )
        elif order == 5:
            kernel = ( -720 * a * g * h * tmp1 * (ell)**(-7) * n + ( 72 * g * h * (ell)**(-5) * n + ( 3 * ( a )**( 4 ) * g * h * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) * n + ( 48 * ( a )**( 3 ) * g * h * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) * n + 216 * ( a )**( 2 ) * g * h * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) * n ) ) ) )
        elif order == 6:
            kernel = ( -900 * g * h * tmp1 * (ell)**(-7) * n + ( 3 * ( a )**( 4 ) * g * h * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) * n + ( 60 * ( a )**( 3 ) * g * h * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) * n + ( 360 * ( a )**( 2 ) * g * h * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) * n + 720 * a * g * h * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) * n ) ) ) )
        elif order == 7:
            kernel = ( 3 * ( a )**( 4 ) * g * h * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) * n + ( 72 * ( a )**( 3 ) * g * h * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) * n + ( 540 * ( a )**( 2 ) * g * h * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) * n + ( 1440 * a * g * h * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) * n + 1080 * g * h * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) * n ) ) ) )
        elif order == 8:
            kernel = ( 3 * ( a )**( 4 ) * g * h * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) * n + ( 84 * ( a )**( 3 ) * g * h * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) * n + ( 756 * ( a )**( 2 ) * g * h * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) * n + ( 2520 * a * g * h * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) * n + 2520 * g * h * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) * n ) ) ) )
        elif order == 9:
            kernel = ( 3 * ( a )**( 4 ) * g * h * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) * n + ( 96 * ( a )**( 3 ) * g * h * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) * n + ( 1008 * ( a )**( 2 ) * g * h * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) * n + ( 4032 * a * g * h * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) * n + 5040 * g * h * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) * n ) ) ) )
        elif order == 10:
            kernel = ( 3 * ( a )**( 4 ) * g * h * ( -4583103525/512 * ( tmp1 )**( 9 ) * (ell)**(-23) + ( 1964187225/32 * ( tmp1 )**( 7 ) * (ell)**(-21) + ( -2170943775/16 * ( tmp1 )**( 5 ) * (ell)**(-19) + ( 212837625/2 * ( tmp1 )**( 3 ) * (ell)**(-17) + -42567525/2 * tmp1 * (ell)**(-15) ) ) ) ) * n + ( 108 * ( a )**( 3 ) * g * h * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) * n + ( 1296 * ( a )**( 2 ) * g * h * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) * n + ( 6048 * a * g * h * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) * n + 9072 * g * h * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) * n ) ) ) )
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**2 * sy.cos(c) \
                * (3*(a*(sy.cos(d)*sy.sin(c)-sy.sin(d)*sy.cos(c)*sy.cos(e-f))) \
                * (a*sy.cos(c)*sy.sin(e-f))
                / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**5)
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    return kernel


def cal_Vxz_kernel(r_cal, phi_cal, lambda_cal, \
        r_tess, phi_tess, lambda_tess, \
        order, is_linear_density):
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
    order: int
        Differentiation of kernel function.
    is_linear_density: bool
        If the tesseroid have linear varying density.
        
    Returns
    -------
    kernel: float
        Kernel of gravitational gradient Vxz.
    """
    a, b, c, d, e, f = r_tess, r_cal, phi_tess, phi_cal, lambda_tess, lambda_cal
    # Common part in taylor series.
    n = np.cos(phi_tess)
    g = np.cos(phi_cal) * np.sin(phi_tess) - np.sin(phi_cal) * n\
        * np.cos(lambda_tess - lambda_cal)
    h = n * np.sin(lambda_tess - lambda_cal)
    m = np.sin(phi_cal) * np.sin(phi_tess) + n * np.cos(phi_cal)\
        * np.cos(lambda_tess - lambda_cal)
    ell = (a**2 + b**2 - 2 * a * b * m)**0.5
    tmp1 = 2 * a - 2 * b * m
    tmp2 = a * m - b
    
    if is_linear_density:
        if order == 0:
            if np.abs(m - 1) < 1e-15:
                kernel = 3 * g * n * ( 1/12 * ( ( a + -1 * b ) )**( -4 ) * ( ( b )**( 5 ) * ( \
                    25 + -77 * m ) + ( 48 * ( a )**( 3 ) * ( b )**( 2 ) * ( -1 + m ) + ( \
                    -12 * ( a )**( 5 ) * m + ( 48 * ( a )**( 4 ) * b * m + ( -36 * ( a \
                    )**( 2 ) * ( b )**( 3 ) * ( -3 + 7 * m ) + 8 * a * ( b )**( 4 ) * ( \
                    -11 + 31 * m ) ) ) ) ) ) + ( b + -5 * b * m ) * np.log( ( -1 * a + \
                    b ) ) )
            elif np.abs(m + 1) < 1e-15:
                kernel = 3 * g * n * ( a * m + ( 1/4 * ( b )**( 5 ) * ( ( a + b ) )**( -4 ) * \
                    ( 1 + m ) + ( -2 * ( b )**( 2 ) * ( ( a + b ) )**( -1 ) * ( 2 + 5 * m \
                    ) + ( ( b )**( 3 ) * ( ( a + b ) )**( -2 ) * ( 3 + 5 * m ) + ( -1/3 * \
                    ( b )**( 4 ) * ( ( a + b ) )**( -3 ) * ( 4 + 5 * m ) + -1 * ( b + 5 * \
                    b * m ) * np.log( ( a + b ) ) ) ) ) ) )
            else:
                kernel = 3 * g * n * ( 1/3 * (ell)**(-3) * ( ( -1 + ( m )**( 2 ) ) )**( -1 ) * ( 3 * ( a )**( 4 ) * m * ( -1 + ( m )**( 2 ) ) + ( ( b )**( 4 ) * m * ( -13 + 15 * ( m )**( 2 ) ) + ( 6 * ( a )**( 2 ) * ( b )**( 2 ) * m * ( -2 + ( -7 * ( m )**( 2 ) + 10 * ( m )**( 4 ) ) ) + ( -2 * ( a )**( 3 ) * b * ( 2 + ( -21 * ( m )**( 2 ) + 20 * ( m )**( 4 ) ) ) + -3 * a * ( b )**( 3 ) * ( 1 + ( -19 * ( m )**( 2 ) + 20 * ( m )**( 4 ) ) ) ) ) ) ) + ( b + -5 * b * ( m )**( 2 ) ) * np.log( ( -1 * a + ( b * m + ( ( ( a )**( 2 ) + ( ( b )**( 2 ) + -2 * a * b * m ) ) )**( 1/2 ) ) ) ) )
        elif order == 1:
            kernel = 3 * ( a )**( 4 ) * g * tmp2 * (ell)**(-5) * n
        elif order == 2:
            kernel = ( -15/2 * ( a )**( 4 ) * g * tmp2 * tmp1 * (ell)**(-7) * n + ( 3 * ( a )**( 4 ) * g * m * (ell)**(-5) * n + 12 * ( a )**( 3 ) * g * tmp2 * (ell)**(-5) * n ) )
        elif order == 3:
            kernel = ( 36 * ( a )**( 2 ) * g * tmp2 * (ell)**(-5) * n + ( 3 * ( a )**( 4 ) * g * ( -5 * m * tmp1 * (ell)**(-7) + tmp2 * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + 24 * ( a )**( 3 ) * g * ( -5/2 * tmp2 * tmp1 * (ell)**(-7) * n + m * (ell)**(-5) * n ) ) )
        elif order == 4:
            kernel = ( 72 * a * g * tmp2 * (ell)**(-5) * n + ( 3 * ( a )**( 4 ) * g * ( tmp2 * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 3 * m * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + ( 36 * ( a )**( 3 ) * g * ( -5 * m * tmp1 * (ell)**(-7) + tmp2 * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + 108 * ( a )**( 2 ) * g * ( -5/2 * tmp2 * tmp1 * (ell)**(-7) * n + m * (ell)**(-5) * n ) ) ) )
        elif order == 5:
            kernel = ( 72 * g * tmp2 * (ell)**(-5) * n + ( 3 * ( a )**( 4 ) * g * ( tmp2 * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 4 * m * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) * n + ( 48 * ( a )**( 3 ) * g * ( tmp2 * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 3 * m * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + ( 216 * ( a )**( 2 ) * g * ( -5 * m * tmp1 * (ell)**(-7) + tmp2 * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + 288 * a * g * ( -5/2 * tmp2 * tmp1 * (ell)**(-7) * n + m * (ell)**(-5) * n ) ) ) ) )
        elif order == 6:
            kernel = ( 3 * ( a )**( 4 ) * g * ( tmp2 * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 5 * m * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) * n + ( 60 * ( a )**( 3 ) * g * ( tmp2 * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 4 * m * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) * n + ( 360 * ( a )**( 2 ) * g * ( tmp2 * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 3 * m * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + ( 720 * a * g * ( -5 * m * tmp1 * (ell)**(-7) + tmp2 * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + 360 * g * ( -5/2 * tmp2 * tmp1 * (ell)**(-7) * n + m * (ell)**(-5) * n ) ) ) ) )
        elif order == 7:
            kernel = ( 3 * ( a )**( 4 ) * g * ( tmp2 * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 6 * m * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) * n + ( 72 * ( a )**( 3 ) * g * ( tmp2 * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 5 * m * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) * n + ( 540 * ( a )**( 2 ) * g * ( tmp2 * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 4 * m * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) * n + ( 1440 * a * g * ( tmp2 * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 3 * m * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + 1080 * g * ( -5 * m * tmp1 * (ell)**(-7) + tmp2 * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n ) ) ) )
        elif order == 8:
            kernel = ( 3 * ( a )**( 4 ) * g * ( tmp2 * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + 7 * m * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) ) * n + ( 84 * ( a )**( 3 ) * g * ( tmp2 * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 6 * m * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) * n + ( 756 * ( a )**( 2 ) * g * ( tmp2 * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 5 * m * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) * n + ( 2520 * a * g * ( tmp2 * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 4 * m * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) * n + 2520 * g * ( tmp2 * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 3 * m * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n ) ) ) )
        elif order == 9:
            kernel = ( 3 * ( a )**( 4 ) * g * ( tmp2 * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) + 8 * m * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) ) * n + ( 96 * ( a )**( 3 ) * g * ( tmp2 * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + 7 * m * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) ) * n + ( 1008 * ( a )**( 2 ) * g * ( tmp2 * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 6 * m * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) * n + ( 4032 * a * g * ( tmp2 * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 5 * m * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) * n + 5040 * g * ( tmp2 * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 4 * m * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) * n ) ) ) )
        elif order == 10:
            kernel = ( 3 * ( a )**( 4 ) * g * ( tmp2 * ( -4583103525/512 * ( tmp1 )**( 9 ) * (ell)**(-23) + ( 1964187225/32 * ( tmp1 )**( 7 ) * (ell)**(-21) + ( -2170943775/16 * ( tmp1 )**( 5 ) * (ell)**(-19) + ( 212837625/2 * ( tmp1 )**( 3 ) * (ell)**(-17) + -42567525/2 * tmp1 * (ell)**(-15) ) ) ) ) + 9 * m * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) ) * n + ( 108 * ( a )**( 3 ) * g * ( tmp2 * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) + 8 * m * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) ) * n + ( 1296 * ( a )**( 2 ) * g * ( tmp2 * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + 7 * m * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) ) * n + ( 6048 * a * g * ( tmp2 * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 6 * m * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) * n + 9072 * g * ( tmp2 * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 5 * m * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) * n ) ) ) )
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**3 * sy.cos(c) \
                * (3*(a*(sy.cos(d)*sy.sin(c)-sy.sin(d)*sy.cos(c)*sy.cos(e-f))) \
                * (a * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f)) - b)
                / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**5)
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    else: 
        if order == 0:
            if np.abs(m - 1) < 1e-15:
                kernel = 3 * g * n * ( 1/12 * ( ( a + -1 * b ) )**( -4 ) * b * ( ( b )**( 3 ) \
                    * ( 3 + -25 * m ) + ( 12 * ( a )**( 3 ) * ( -1 + 4 * m ) + ( -18 * ( \
                    a )**( 2 ) * b * ( -1 + 6 * m ) + 4 * a * ( b )**( 2 ) * ( -3 + 22 * \
                    m ) ) ) ) + -1 * m * np.log( ( a + -1 * b ) ) )
            elif np.abs(m + 1) < 1e-15:
                kernel = 1/4 * ( ( a + b ) )**( -4 ) * g * n * ( 12 * ( a )**( 3 ) * b * ( 1 + \
                    4 * m ) + ( 4 * a * ( b )**( 3 ) * ( 3 + 22 * m ) + ( ( b )**( 4 ) * \
                    ( 3 + 25 * m ) + ( 18 * ( a )**( 2 ) * b * ( b + 6 * b * m ) + 12 * ( \
                    ( a + b ) )**( 4 ) * m * np.log( ( a + b ) ) ) ) ) )
            else:
                kernel = 3 * g * n * ( 1/3 * (ell)**(-3) * ( ( -1 + ( m )**( 2 ) ) )**( -1 ) * ( 3 * a * ( b )**( 2 ) * m * ( 3 + -4 * ( m )**( 2 ) ) + ( ( b )**( 3 ) * ( -2 + 3 * ( m )**( 2 ) ) + ( ( a )**( 3 ) * ( 7 * m + -8 * ( m )**( 3 ) ) + 3 * ( a )**( 2 ) * b * ( -1 + ( -2 * ( m )**( 2 ) + 4 * ( m )**( 4 ) ) ) ) ) ) + -1 * m * np.log( ( -1 * a + ( b * m + ( ( ( a )**( 2 ) + ( ( b )**( 2 ) + -2 * a * b * m ) ) )**( 1/2 ) ) ) ) )
        elif order == 1:
            kernel = 3 * ( a )**( 3 ) * g * tmp2 * (ell)**(-5) * n
        elif order == 2:
            kernel = ( -15/2 * ( a )**( 3 ) * g * tmp2 * tmp1 * (ell)**(-7) * n + ( 3 * ( a )**( 3 ) * g * m * (ell)**(-5) * n + 9 * ( a )**( 2 ) * g * tmp2 * (ell)**(-5) * n ) )
        elif order == 3:
            kernel = ( 18 * a * g * tmp2 * (ell)**(-5) * n + ( 3 * ( a )**( 3 ) * g * ( -5 * m * tmp1 * (ell)**(-7) + tmp2 * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + 18 * ( a )**( 2 ) * g * ( -5/2 * tmp2 * tmp1 * (ell)**(-7) * n + m * (ell)**(-5) * n ) ) )
        elif order == 4:
            kernel = ( 18 * g * tmp2 * (ell)**(-5) * n + ( 3 * ( a )**( 3 ) * g * ( tmp2 * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 3 * m * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + ( 27 * ( a )**( 2 ) * g * ( -5 * m * tmp1 * (ell)**(-7) + tmp2 * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + 54 * a * g * ( -5/2 * tmp2 * tmp1 * (ell)**(-7) * n + m * (ell)**(-5) * n ) ) ) )
        elif order == 5:
            kernel = ( 3 * ( a )**( 3 ) * g * ( tmp2 * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 4 * m * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) * n + ( 36 * ( a )**( 2 ) * g * ( tmp2 * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 3 * m * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + ( 108 * a * g * ( -5 * m * tmp1 * (ell)**(-7) + tmp2 * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + 72 * g * ( -5/2 * tmp2 * tmp1 * (ell)**(-7) * n + m * (ell)**(-5) * n ) ) ) )
        elif order == 6:
            kernel = ( 3 * ( a )**( 3 ) * g * ( tmp2 * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 5 * m * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) * n + ( 45 * ( a )**( 2 ) * g * ( tmp2 * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 4 * m * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) * n + ( 180 * a * g * ( tmp2 * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 3 * m * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + 180 * g * ( -5 * m * tmp1 * (ell)**(-7) + tmp2 * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n ) ) )
        elif order == 7:
            kernel = ( 3 * ( a )**( 3 ) * g * ( tmp2 * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 6 * m * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) * n + ( 54 * ( a )**( 2 ) * g * ( tmp2 * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 5 * m * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) * n + ( 270 * a * g * ( tmp2 * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 4 * m * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) * n + 360 * g * ( tmp2 * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 3 * m * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n ) ) )
        elif order == 8:
            kernel = ( 3 * ( a )**( 3 ) * g * ( tmp2 * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + 7 * m * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) ) * n + ( 63 * ( a )**( 2 ) * g * ( tmp2 * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 6 * m * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) * n + ( 378 * a * g * ( tmp2 * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 5 * m * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) * n + 630 * g * ( tmp2 * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 4 * m * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) * n ) ) )
        elif order == 9:
            kernel = ( 3 * ( a )**( 3 ) * g * ( tmp2 * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) + 8 * m * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) ) * n + ( 72 * ( a )**( 2 ) * g * ( tmp2 * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + 7 * m * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) ) * n + ( 504 * a * g * ( tmp2 * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 6 * m * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) * n + 1008 * g * ( tmp2 * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 5 * m * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) * n ) ) )
        elif order == 10:
            kernel = ( 3 * ( a )**( 3 ) * g * ( tmp2 * ( -4583103525/512 * ( tmp1 )**( 9 ) * (ell)**(-23) + ( 1964187225/32 * ( tmp1 )**( 7 ) * (ell)**(-21) + ( -2170943775/16 * ( tmp1 )**( 5 ) * (ell)**(-19) + ( 212837625/2 * ( tmp1 )**( 3 ) * (ell)**(-17) + -42567525/2 * tmp1 * (ell)**(-15) ) ) ) ) + 9 * m * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) ) * n + ( 81 * ( a )**( 2 ) * g * ( tmp2 * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) + 8 * m * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) ) * n + ( 648 * a * g * ( tmp2 * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + 7 * m * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) ) * n + 1512 * g * ( tmp2 * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 6 * m * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) * n ) ) )
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**2 * sy.cos(c) \
                * (3*(a*(sy.cos(d)*sy.sin(c)-sy.sin(d)*sy.cos(c)*sy.cos(e-f))) \
                * (a * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f)) - b)
                / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**5)
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    return kernel


def cal_Vyy_kernel(r_cal, phi_cal, lambda_cal, \
        r_tess, phi_tess, lambda_tess, \
        order, is_linear_density):
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
    order: int
        Differentiation of kernel function.
    is_linear_density: bool
        If the tesseroid have linear varying density.
        
    Returns
    -------
    kernel: float
        Kernel of gravitational gradient Vyy.
    """
    a, b, c, d, e, f = r_tess, r_cal, phi_tess, phi_cal, lambda_tess, lambda_cal
    # Common part of taylor series.
    n = np.cos(phi_tess)
    g = n * np.sin(lambda_tess - lambda_cal)
    m = np.sin(phi_cal) * np.sin(phi_tess) + n * np.cos(phi_cal)\
        * np.cos(lambda_tess - lambda_cal)
    ell = (a**2 + b**2 - 2 * a * b * m)**0.5
    tmp1 = 2 * a - 2 * b * m

    if is_linear_density:
        if order == 0:
            if np.abs(m - 1) < 1e-15:
                kernel = n * ( 1/4 * ( ( a + -1 * b ) )**( -4 ) * ( 28 * ( a )**( 3 ) * ( b \
                    )**( 2 ) + ( ( b )**( 5 ) * ( 6 + -65 * ( g )**( 2 ) ) + ( ( a )**( 5 \
                    ) * ( 4 + -12 * ( g )**( 2 ) ) + ( 20 * ( a )**( 4 ) * b * ( -1 + 3 * \
                    ( g )**( 2 ) ) + ( -6 * ( a )**( 2 ) * ( b )**( 3 ) * ( 1 + 30 * ( g \
                    )**( 2 ) ) + 4 * a * ( b )**( 4 ) * ( -3 + 50 * ( g )**( 2 ) ) ) ) ) \
                    ) ) + 3 * b * ( 1 + -5 * ( g )**( 2 ) ) * np.log( ( a + -1 * b ) ) \
                    )
            elif np.abs(m + 1) < 1e-15:
                kernel = n * ( 3/4 * ( b )**( 5 ) * ( ( a + b ) )**( -4 ) * ( g )**( 2 ) + ( \
                    -5 * ( b )**( 4 ) * ( ( a + b ) )**( -3 ) * ( g )**( 2 ) + ( 3 * ( b \
                    )**( 2 ) * ( ( a + b ) )**( -1 ) * ( 1 + -10 * ( g )**( 2 ) ) + ( a * \
                    ( -1 + 3 * ( g )**( 2 ) ) + ( 1/2 * ( b )**( 3 ) * ( ( a + b ) )**( \
                    -2 ) * ( -1 + 30 * ( g )**( 2 ) ) + -3 * b * ( -1 + 5 * ( g )**( 2 ) \
                    ) * np.log( ( a + b ) ) ) ) ) ) )
            else:
                kernel = n * ( (ell)**(-3) * ( ( -1 + ( m )**( 2 ) ) )**( -2 ) * ( ( a )**( 4 ) * ( -1 + 3 * ( g )**( 2 ) ) * ( ( -1 + ( m )**( 2 ) ) )**( 2 ) + ( ( a )**( 3 ) * b * m * ( 7 + ( -15 * ( m )**( 2 ) + ( 8 * ( m )**( 4 ) + ( g )**( 2 ) * ( -32 + ( 74 * ( m )**( 2 ) + -40 * ( m )**( 4 ) ) ) ) ) ) + ( ( b )**( 4 ) * ( -2 + ( 5 * ( m )**( 2 ) + ( -3 * ( m )**( 4 ) + ( g )**( 2 ) * ( 8 + ( -25 * ( m )**( 2 ) + 15 * ( m )**( 4 ) ) ) ) ) ) + ( -3 * a * ( b )**( 3 ) * m * ( -3 + ( 7 * ( m )**( 2 ) + ( -4 * ( m )**( 4 ) + ( g )**( 2 ) * ( 13 + ( -35 * ( m )**( 2 ) + 20 * ( m )**( 4 ) ) ) ) ) ) + 3 * ( a )**( 2 ) * ( b )**( 2 ) * ( -1 + ( -1 * ( m )**( 2 ) + ( 6 * ( m )**( 4 ) + ( -4 * ( m )**( 6 ) + ( g )**( 2 ) * ( 4 + ( 4 * ( m )**( 2 ) + ( -30 * ( m )**( 4 ) + 20 * ( m )**( 6 ) ) ) ) ) ) ) ) ) ) ) ) + -3 * b * ( -1 + 5 * ( g )**( 2 ) ) * m * np.log( ( -1 * a + ( b * m + ( ( ( a )**( 2 ) + ( ( b )**( 2 ) + -2 * a * b * m ) ) )**( 1/2 ) ) ) ) )
        elif order == 1:
            kernel = ( a )**( 3 ) * ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * (ell)**(-5) + -1 * (ell)**(-3) ) * n
        elif order == 2:
            kernel = ( ( a )**( 3 ) * ( -15/2 * ( a )**( 2 ) * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 6 * a * ( g )**( 2 ) * (ell)**(-5) + 3/2 * tmp1 * (ell)**(-5) ) ) * n + 3 * ( a )**( 2 ) * ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * (ell)**(-5) + -1 * (ell)**(-3) ) * n )
        elif order == 3:
            kernel = ( 6 * ( a )**( 2 ) * ( -15/2 * ( a )**( 2 ) * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 6 * a * ( g )**( 2 ) * (ell)**(-5) + 3/2 * tmp1 * (ell)**(-5) ) ) + ( 6 * a * ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * (ell)**(-5) + -1 * (ell)**(-3) ) + ( a )**( 3 ) * ( -30 * a * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( -15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + ( 3 * (ell)**(-5) + ( 6 * ( g )**( 2 ) * (ell)**(-5) + 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) * n
        elif order == 4:
            kernel = ( 18 * a * ( -15/2 * ( a )**( 2 ) * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 6 * a * ( g )**( 2 ) * (ell)**(-5) + 3/2 * tmp1 * (ell)**(-5) ) ) + ( 6 * ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * (ell)**(-5) + -1 * (ell)**(-3) ) + ( ( a )**( 3 ) * ( 105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + ( -45/2 * tmp1 * (ell)**(-7) + ( -45 * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 18 * a * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) + 9 * ( a )**( 2 ) * ( -30 * a * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( -15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + ( 3 * (ell)**(-5) + ( 6 * ( g )**( 2 ) * (ell)**(-5) + 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) ) * n
        elif order == 5:
            kernel = ( 24 * ( -15/2 * ( a )**( 2 ) * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 6 * a * ( g )**( 2 ) * (ell)**(-5) + 3/2 * tmp1 * (ell)**(-5) ) ) + ( ( a )**( 3 ) * ( -945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( 315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + ( -45 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + ( 24 * a * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 36 * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) + ( 12 * ( a )**( 2 ) * ( 105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + ( -45/2 * tmp1 * (ell)**(-7) + ( -45 * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 18 * a * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) + 36 * a * ( -30 * a * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( -15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + ( 3 * (ell)**(-5) + ( 6 * ( g )**( 2 ) * (ell)**(-5) + 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) ) * n
        elif order == 6:
            kernel = ( ( a )**( 3 ) * ( 10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( -4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + ( 1575/2 * tmp1 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + ( 30 * a * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 60 * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) ) ) ) ) + ( 15 * ( a )**( 2 ) * ( -945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( 315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + ( -45 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + ( 24 * a * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 36 * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) + ( 60 * a * ( 105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + ( -45/2 * tmp1 * (ell)**(-7) + ( -45 * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 18 * a * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) + 60 * ( -30 * a * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( -15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + ( 3 * (ell)**(-5) + ( 6 * ( g )**( 2 ) * (ell)**(-5) + 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) ) * n
        elif order == 7:
            kernel = ( ( a )**( 3 ) * ( -135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( 155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + ( 1575 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + ( 36 * a * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 90 * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) ) ) ) ) ) + ( 18 * ( a )**( 2 ) * ( 10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( -4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + ( 1575/2 * tmp1 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + ( 30 * a * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 60 * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) ) ) ) ) + ( 90 * a * ( -945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( 315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + ( -45 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + ( 24 * a * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 36 * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) + 120 * ( 105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + ( -45/2 * tmp1 * (ell)**(-7) + ( -45 * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 18 * a * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) ) * n
        elif order == 8:
            kernel = ( ( a )**( 3 ) * ( 2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( -2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + ( -99225/2 * tmp1 * (ell)**(-11) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + ( 42 * a * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 126 * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) ) ) ) ) ) + ( 21 * ( a )**( 2 ) * ( -135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( 155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + ( 1575 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + ( 36 * a * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 90 * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) ) ) ) ) ) + ( 126 * a * ( 10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( -4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + ( 1575/2 * tmp1 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + ( 30 * a * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 60 * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) ) ) ) ) + 210 * ( -945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( 315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + ( -45 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + ( 24 * a * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 36 * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) ) ) * n
        elif order == 9:
            kernel = ( ( a )**( 3 ) * ( -34459425/256 * ( tmp1 )**( 8 ) * (ell)**(-19) + ( 14189175/16 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -14189175/8 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 1091475 * ( tmp1 )**( 2 ) * (ell)**(-13) + ( -99225 * (ell)**(-11) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) + ( 48 * a * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + 168 * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) ) ) ) ) ) ) ) + ( 24 * ( a )**( 2 ) * ( 2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( -2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + ( -99225/2 * tmp1 * (ell)**(-11) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + ( 42 * a * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 126 * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) ) ) ) ) ) + ( 168 * a * ( -135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( 155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + ( 1575 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + ( 36 * a * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 90 * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) ) ) ) ) ) + 336 * ( 10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( -4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + ( 1575/2 * tmp1 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + ( 30 * a * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 60 * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) ) ) ) ) ) ) ) * n
        elif order == 10:
            kernel = ( ( a )**( 3 ) * ( 654729075/512 * ( tmp1 )**( 9 ) * (ell)**(-21) + ( -310134825/32 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 383107725/16 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -42567525/2 * ( tmp1 )**( 3 ) * (ell)**(-15) + ( 9823275/2 * tmp1 * (ell)**(-13) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -4583103525/512 * ( tmp1 )**( 9 ) * (ell)**(-23) + ( 1964187225/32 * ( tmp1 )**( 7 ) * (ell)**(-21) + ( -2170943775/16 * ( tmp1 )**( 5 ) * (ell)**(-19) + ( 212837625/2 * ( tmp1 )**( 3 ) * (ell)**(-17) + -42567525/2 * tmp1 * (ell)**(-15) ) ) ) ) + ( 54 * a * ( g )**( 2 ) * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) + 216 * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) ) ) ) ) ) ) ) + ( 27 * ( a )**( 2 ) * ( -34459425/256 * ( tmp1 )**( 8 ) * (ell)**(-19) + ( 14189175/16 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -14189175/8 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 1091475 * ( tmp1 )**( 2 ) * (ell)**(-13) + ( -99225 * (ell)**(-11) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) + ( 48 * a * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + 168 * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) ) ) ) ) ) ) ) + ( 216 * a * ( 2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( -2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + ( -99225/2 * tmp1 * (ell)**(-11) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + ( 42 * a * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 126 * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) ) ) ) ) ) + 504 * ( -135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( 155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + ( 1575 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + ( 36 * a * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 90 * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) ) ) ) ) ) ) ) ) * n
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**3 * sy.cos(c) \
                * (3 * (a*(sy.cos(d)*sy.sin(c)-sy.sin(d)*sy.cos(c)*sy.cos(e-f)))**2 \
                / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**5 \
                - 1 / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**3)
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    else:
        if order == 0:
            if np.abs(m - 1) < 1e-15:
                kernel = n * ( 1/4 * ( ( a + -1 * b ) )**( -4 ) * b * ( 2 * ( a )**( 2 ) * b * \
                    ( 11 + -54 * ( g )**( 2 ) ) + ( ( b )**( 3 ) * ( 6 + -25 * ( g )**( 2 \
                    ) ) + ( 8 * ( a )**( 3 ) * ( -1 + 6 * ( g )**( 2 ) ) + 4 * a * ( b \
                    )**( 2 ) * ( -5 + 22 * ( g )**( 2 ) ) ) ) ) + ( 1 + -3 * ( g )**( 2 ) \
                    ) * np.log( ( a + -1 * b ) ) )
            elif np.abs(m + 1) < 1e-15:
                kernel = n * ( 1/4 * b * ( ( a + b ) )**( -4 ) * ( 8 * ( a )**( 3 ) * ( -1 + 6 \
                    * ( g )**( 2 ) ) + ( 4 * a * ( b )**( 2 ) * ( -5 + 22 * ( g )**( 2 ) \
                    ) + ( ( b )**( 3 ) * ( -6 + 25 * ( g )**( 2 ) ) + 2 * ( a )**( 2 ) * \
                    b * ( -11 + 54 * ( g )**( 2 ) ) ) ) ) + ( -1 + 3 * ( g )**( 2 ) ) * \
                    np.log( ( a + b ) ) )
            else:
                kernel = n * ( (ell)**(-3) * ( ( -1 + ( m )**( 2 ) ) )**( -2 ) * ( ( a )**( 2 ) * b * m * ( -1 + ( ( 5 + -18 * ( g )**( 2 ) ) * ( m )**( 2 ) + 4 * ( -1 + 3 * ( g )**( 2 ) ) * ( m )**( 4 ) ) ) + ( ( b )**( 3 ) * m * ( 1 + ( -1 * ( m )**( 2 ) + ( g )**( 2 ) * ( -5 + 3 * ( m )**( 2 ) ) ) ) + ( a * ( b )**( 2 ) * ( 1 + ( -5 * ( m )**( 2 ) + ( 4 * ( m )**( 4 ) + -3 * ( g )**( 2 ) * ( 1 + ( -7 * ( m )**( 2 ) + 4 * ( m )**( 4 ) ) ) ) ) ) + ( a )**( 3 ) * ( 1 + ( -3 * ( m )**( 2 ) + ( 2 * ( m )**( 4 ) + -2 * ( g )**( 2 ) * ( 2 + ( -7 * ( m )**( 2 ) + 4 * ( m )**( 4 ) ) ) ) ) ) ) ) ) + ( 1 + -3 * ( g )**( 2 ) ) * np.log( ( -1 * a + ( b * m + ( ( ( a )**( 2 ) + ( ( b )**( 2 ) + -2 * a * b * m ) ) )**( 1/2 ) ) ) ) )
        elif order == 1:
            kernel = ( a )**( 2 ) * ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * (ell)**(-5) + -1 * (ell)**(-3) ) * n
        elif order == 2:
            kernel = ( ( a )**( 2 ) * ( -15/2 * ( a )**( 2 ) * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 6 * a * ( g )**( 2 ) * (ell)**(-5) + 3/2 * tmp1 * (ell)**(-5) ) ) * n + 2 * a * ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * (ell)**(-5) + -1 * (ell)**(-3) ) * n )
        elif order == 3:
            kernel = ( 4 * a * ( -15/2 * ( a )**( 2 ) * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 6 * a * ( g )**( 2 ) * (ell)**(-5) + 3/2 * tmp1 * (ell)**(-5) ) ) + ( 2 * ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * (ell)**(-5) + -1 * (ell)**(-3) ) + ( a )**( 2 ) * ( -30 * a * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( -15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + ( 3 * (ell)**(-5) + ( 6 * ( g )**( 2 ) * (ell)**(-5) + 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) * n
        elif order == 4:
            kernel = ( 6 * ( -15/2 * ( a )**( 2 ) * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 6 * a * ( g )**( 2 ) * (ell)**(-5) + 3/2 * tmp1 * (ell)**(-5) ) ) + ( ( a )**( 2 ) * ( 105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + ( -45/2 * tmp1 * (ell)**(-7) + ( -45 * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 18 * a * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) + 6 * a * ( -30 * a * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( -15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + ( 3 * (ell)**(-5) + ( 6 * ( g )**( 2 ) * (ell)**(-5) + 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) * n
        elif order == 5:
            kernel = ( ( a )**( 2 ) * ( -945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( 315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + ( -45 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + ( 24 * a * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 36 * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) + ( 8 * a * ( 105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + ( -45/2 * tmp1 * (ell)**(-7) + ( -45 * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 18 * a * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) + 12 * ( -30 * a * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( -15/4 * ( tmp1 )**( 2 ) * (ell)**(-7) + ( 3 * (ell)**(-5) + ( 6 * ( g )**( 2 ) * (ell)**(-5) + 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) * n
        elif order == 6:
            kernel = ( ( a )**( 2 ) * ( 10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( -4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + ( 1575/2 * tmp1 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + ( 30 * a * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 60 * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) ) ) ) ) + ( 10 * a * ( -945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( 315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + ( -45 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + ( 24 * a * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 36 * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) + 20 * ( 105/8 * ( tmp1 )**( 3 ) * (ell)**(-9) + ( -45/2 * tmp1 * (ell)**(-7) + ( -45 * ( g )**( 2 ) * tmp1 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 18 * a * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) * n
        elif order == 7:
            kernel = ( ( a )**( 2 ) * ( -135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( 155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + ( 1575 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + ( 36 * a * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 90 * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) ) ) ) ) ) + ( 12 * a * ( 10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( -4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + ( 1575/2 * tmp1 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + ( 30 * a * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 60 * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) ) ) ) ) + 30 * ( -945/16 * ( tmp1 )**( 4 ) * (ell)**(-11) + ( 315/2 * ( tmp1 )**( 2 ) * (ell)**(-9) + ( -45 * (ell)**(-7) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + ( 24 * a * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 36 * ( g )**( 2 ) * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) ) ) ) ) ) ) * n
        elif order == 8:
            kernel = ( ( a )**( 2 ) * ( 2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( -2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + ( -99225/2 * tmp1 * (ell)**(-11) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + ( 42 * a * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 126 * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) ) ) ) ) ) + ( 14 * a * ( -135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( 155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + ( 1575 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + ( 36 * a * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 90 * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) ) ) ) ) ) + 42 * ( 10395/32 * ( tmp1 )**( 5 ) * (ell)**(-13) + ( -4725/4 * ( tmp1 )**( 3 ) * (ell)**(-11) + ( 1575/2 * tmp1 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + ( 30 * a * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 60 * ( g )**( 2 ) * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) ) ) ) ) ) ) * n
        elif order == 9:
            kernel = ( ( a )**( 2 ) * ( -34459425/256 * ( tmp1 )**( 8 ) * (ell)**(-19) + ( 14189175/16 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -14189175/8 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 1091475 * ( tmp1 )**( 2 ) * (ell)**(-13) + ( -99225 * (ell)**(-11) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) + ( 48 * a * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + 168 * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) ) ) ) ) ) ) ) + ( 16 * a * ( 2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( -2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + ( -99225/2 * tmp1 * (ell)**(-11) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + ( 42 * a * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 126 * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) ) ) ) ) ) + 56 * ( -135135/64 * ( tmp1 )**( 6 ) * (ell)**(-15) + ( 155925/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -42525/4 * ( tmp1 )**( 2 ) * (ell)**(-11) + ( 1575 * (ell)**(-9) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + ( 36 * a * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 90 * ( g )**( 2 ) * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) ) ) ) ) ) ) ) * n
        elif order == 10:
            kernel = ( ( a )**( 2 ) * ( 654729075/512 * ( tmp1 )**( 9 ) * (ell)**(-21) + ( -310134825/32 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 383107725/16 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -42567525/2 * ( tmp1 )**( 3 ) * (ell)**(-15) + ( 9823275/2 * tmp1 * (ell)**(-13) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -4583103525/512 * ( tmp1 )**( 9 ) * (ell)**(-23) + ( 1964187225/32 * ( tmp1 )**( 7 ) * (ell)**(-21) + ( -2170943775/16 * ( tmp1 )**( 5 ) * (ell)**(-19) + ( 212837625/2 * ( tmp1 )**( 3 ) * (ell)**(-17) + -42567525/2 * tmp1 * (ell)**(-15) ) ) ) ) + ( 54 * a * ( g )**( 2 ) * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) + 216 * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) ) ) ) ) ) ) ) + ( 18 * a * ( -34459425/256 * ( tmp1 )**( 8 ) * (ell)**(-19) + ( 14189175/16 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -14189175/8 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 1091475 * ( tmp1 )**( 2 ) * (ell)**(-13) + ( -99225 * (ell)**(-11) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) + ( 48 * a * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + 168 * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) ) ) ) ) ) ) ) + 72 * ( 2027025/128 * ( tmp1 )**( 7 ) * (ell)**(-17) + ( -2837835/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 1091475/8 * ( tmp1 )**( 3 ) * (ell)**(-13) + ( -99225/2 * tmp1 * (ell)**(-11) + ( 3 * ( a )**( 2 ) * ( g )**( 2 ) * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + ( 42 * a * ( g )**( 2 ) * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 126 * ( g )**( 2 ) * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) ) ) ) ) ) ) ) * n
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**2 * sy.cos(c) \
                * (3 * (a*(sy.cos(d)*sy.sin(c)-sy.sin(d)*sy.cos(c)*sy.cos(e-f)))**2 \
                / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**5 \
                - 1 / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**3)
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    return kernel


def cal_Vyz_kernel(r_cal, phi_cal, lambda_cal, \
        r_tess, phi_tess, lambda_tess, \
        order, is_linear_density):
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
    order: int
        Differentiation of kernel function.
    is_linear_density: bool
        If the tesseroid have linear varying density.
        
    Returns
    -------
    kernel: float
        Kernel of gravitational gradient Vyz.
    """
    a, b, c, d, e, f = r_tess, r_cal, phi_tess, phi_cal, lambda_tess, lambda_cal
    # Common part in taylor series
    n = np.cos(phi_tess)
    g = n * np.sin(lambda_tess - lambda_cal)
    m = np.sin(phi_cal) * np.sin(phi_tess) + n * np.cos(phi_cal)\
        * np.cos(lambda_tess - lambda_cal)
    ell = (a**2 + b**2 - 2 * a * b * m)**0.5
    tmp1 = 2 * a - 2 * b * m
    tmp2 = a * m - b

    if is_linear_density:
        if order == 0:
            if np.abs(m - 1) < 1e-15:
                kernel = 3 * g * n * ( 1/12 * ( ( a + -1 * b ) )**( -4 ) * ( ( b )**( 5 ) * ( \
                    25 + -77 * m ) + ( 48 * ( a )**( 3 ) * ( b )**( 2 ) * ( -1 + m ) + ( \
                    -12 * ( a )**( 5 ) * m + ( 48 * ( a )**( 4 ) * b * m + ( -36 * ( a \
                    )**( 2 ) * ( b )**( 3 ) * ( -3 + 7 * m ) + 8 * a * ( b )**( 4 ) * ( \
                    -11 + 31 * m ) ) ) ) ) ) + ( b + -5 * b * m ) * np.log( ( -1 * a + \
                    b ) ) )
            elif np.abs(m + 1) < 1e-15:
                kernel = 3 * g * n * ( a * m + ( 1/4 * ( b )**( 5 ) * ( ( a + b ) )**( -4 ) * \
                    ( 1 + m ) + ( -2 * ( b )**( 2 ) * ( ( a + b ) )**( -1 ) * ( 2 + 5 * m \
                    ) + ( ( b )**( 3 ) * ( ( a + b ) )**( -2 ) * ( 3 + 5 * m ) + ( -1/3 * \
                    ( b )**( 4 ) * ( ( a + b ) )**( -3 ) * ( 4 + 5 * m ) + -1 * ( b + 5 * \
                    b * m ) * np.log( ( a + b ) ) ) ) ) ) )
            else:
                kernel = 3 * g * n * ( 1/3 * (ell)**(-3) * ( ( -1 + ( m )**( 2 ) ) )**( -1 ) * ( 3 * ( a )**( 4 ) * m * ( -1 + ( m )**( 2 ) ) + ( ( b )**( 4 ) * m * ( -13 + 15 * ( m )**( 2 ) ) + ( 6 * ( a )**( 2 ) * ( b )**( 2 ) * m * ( -2 + ( -7 * ( m )**( 2 ) + 10 * ( m )**( 4 ) ) ) + ( -2 * ( a )**( 3 ) * b * ( 2 + ( -21 * ( m )**( 2 ) + 20 * ( m )**( 4 ) ) ) + -3 * a * ( b )**( 3 ) * ( 1 + ( -19 * ( m )**( 2 ) + 20 * ( m )**( 4 ) ) ) ) ) ) ) + ( b + -5 * b * ( m )**( 2 ) ) * np.log( ( -1 * a + ( b * m + ( ( ( a )**( 2 ) + ( ( b )**( 2 ) + -2 * a * b * m ) ) )**( 1/2 ) ) ) ) )
        elif order == 1:
            kernel = 3 * ( a )**( 4 ) * g * tmp2 * (ell)**(-5) * n
        elif order == 2:
            kernel = ( -15/2 * ( a )**( 4 ) * g * tmp2 * tmp1 * (ell)**(-7) * n + ( 3 * ( a )**( 4 ) * g * m * (ell)**(-5) * n + 12 * ( a )**( 3 ) * g * tmp2 * (ell)**(-5) * n ) )
        elif order == 3:
            kernel = ( 36 * ( a )**( 2 ) * g * tmp2 * (ell)**(-5) * n + ( 3 * ( a )**( 4 ) * g * ( -5 * m * tmp1 * (ell)**(-7) + tmp2 * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + 24 * ( a )**( 3 ) * g * ( -5/2 * tmp2 * tmp1 * (ell)**(-7) * n + m * (ell)**(-5) * n ) ) )
        elif order == 4:
            kernel = ( 72 * a * g * tmp2 * (ell)**(-5) * n + ( 3 * ( a )**( 4 ) * g * ( tmp2 * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 3 * m * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + ( 36 * ( a )**( 3 ) * g * ( -5 * m * tmp1 * (ell)**(-7) + tmp2 * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + 108 * ( a )**( 2 ) * g * ( -5/2 * tmp2 * tmp1 * (ell)**(-7) * n + m * (ell)**(-5) * n ) ) ) )
        elif order == 5:
            kernel = ( 72 * g * tmp2 * (ell)**(-5) * n + ( 3 * ( a )**( 4 ) * g * ( tmp2 * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 4 * m * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) * n + ( 48 * ( a )**( 3 ) * g * ( tmp2 * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 3 * m * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + ( 216 * ( a )**( 2 ) * g * ( -5 * m * tmp1 * (ell)**(-7) + tmp2 * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + 288 * a * g * ( -5/2 * tmp2 * tmp1 * (ell)**(-7) * n + m * (ell)**(-5) * n ) ) ) ) )
        elif order == 6:
            kernel = ( 3 * ( a )**( 4 ) * g * ( tmp2 * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 5 * m * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) * n + ( 60 * ( a )**( 3 ) * g * ( tmp2 * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 4 * m * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) * n + ( 360 * ( a )**( 2 ) * g * ( tmp2 * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 3 * m * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + ( 720 * a * g * ( -5 * m * tmp1 * (ell)**(-7) + tmp2 * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + 360 * g * ( -5/2 * tmp2 * tmp1 * (ell)**(-7) * n + m * (ell)**(-5) * n ) ) ) ) )
        elif order == 7:
            kernel = ( 3 * ( a )**( 4 ) * g * ( tmp2 * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 6 * m * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) * n + ( 72 * ( a )**( 3 ) * g * ( tmp2 * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 5 * m * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) * n + ( 540 * ( a )**( 2 ) * g * ( tmp2 * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 4 * m * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) * n + ( 1440 * a * g * ( tmp2 * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 3 * m * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + 1080 * g * ( -5 * m * tmp1 * (ell)**(-7) + tmp2 * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n ) ) ) )
        elif order == 8:
            kernel = ( 3 * ( a )**( 4 ) * g * ( tmp2 * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + 7 * m * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) ) * n + ( 84 * ( a )**( 3 ) * g * ( tmp2 * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 6 * m * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) * n + ( 756 * ( a )**( 2 ) * g * ( tmp2 * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 5 * m * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) * n + ( 2520 * a * g * ( tmp2 * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 4 * m * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) * n + 2520 * g * ( tmp2 * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 3 * m * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n ) ) ) )
        elif order == 9:
            kernel = ( 3 * ( a )**( 4 ) * g * ( tmp2 * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) + 8 * m * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) ) * n + ( 96 * ( a )**( 3 ) * g * ( tmp2 * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + 7 * m * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) ) * n + ( 1008 * ( a )**( 2 ) * g * ( tmp2 * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 6 * m * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) * n + ( 4032 * a * g * ( tmp2 * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 5 * m * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) * n + 5040 * g * ( tmp2 * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 4 * m * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) * n ) ) ) )
        elif order == 10:
            kernel = ( 3 * ( a )**( 4 ) * g * ( tmp2 * ( -4583103525/512 * ( tmp1 )**( 9 ) * (ell)**(-23) + ( 1964187225/32 * ( tmp1 )**( 7 ) * (ell)**(-21) + ( -2170943775/16 * ( tmp1 )**( 5 ) * (ell)**(-19) + ( 212837625/2 * ( tmp1 )**( 3 ) * (ell)**(-17) + -42567525/2 * tmp1 * (ell)**(-15) ) ) ) ) + 9 * m * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) ) * n + ( 108 * ( a )**( 3 ) * g * ( tmp2 * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) + 8 * m * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) ) * n + ( 1296 * ( a )**( 2 ) * g * ( tmp2 * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + 7 * m * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) ) * n + ( 6048 * a * g * ( tmp2 * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 6 * m * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) * n + 9072 * g * ( tmp2 * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 5 * m * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) * n ) ) ) )
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**3 * sy.cos(c) \
                * (3*(a*(sy.cos(d)*sy.sin(c)-sy.sin(d)*sy.cos(c)*sy.cos(e-f))) \
                * (a * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f)) - b)
                / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**5)
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    else: 
        if order == 0:
            if np.abs(m - 1) < 1e-15:
                kernel = 3 * g * n * ( 1/12 * ( ( a + -1 * b ) )**( -4 ) * b * ( ( b )**( 3 ) \
                    * ( 3 + -25 * m ) + ( 12 * ( a )**( 3 ) * ( -1 + 4 * m ) + ( -18 * ( \
                    a )**( 2 ) * b * ( -1 + 6 * m ) + 4 * a * ( b )**( 2 ) * ( -3 + 22 * \
                    m ) ) ) ) + -1 * m * np.log( ( a + -1 * b ) ) )
            elif np.abs(m + 1) < 1e-15:
                kernel = 1/4 * ( ( a + b ) )**( -4 ) * g * n * ( 12 * ( a )**( 3 ) * b * ( 1 + \
                    4 * m ) + ( 4 * a * ( b )**( 3 ) * ( 3 + 22 * m ) + ( ( b )**( 4 ) * \
                    ( 3 + 25 * m ) + ( 18 * ( a )**( 2 ) * b * ( b + 6 * b * m ) + 12 * ( \
                    ( a + b ) )**( 4 ) * m * np.log( ( a + b ) ) ) ) ) )
            else:
                kernel = 3 * g * n * ( 1/3 * (ell)**(-3) * ( ( -1 + ( m )**( 2 ) ) )**( -1 ) * ( 3 * a * ( b )**( 2 ) * m * ( 3 + -4 * ( m )**( 2 ) ) + ( ( b )**( 3 ) * ( -2 + 3 * ( m )**( 2 ) ) + ( ( a )**( 3 ) * ( 7 * m + -8 * ( m )**( 3 ) ) + 3 * ( a )**( 2 ) * b * ( -1 + ( -2 * ( m )**( 2 ) + 4 * ( m )**( 4 ) ) ) ) ) ) + -1 * m * np.log( ( -1 * a + ( b * m + ( ( ( a )**( 2 ) + ( ( b )**( 2 ) + -2 * a * b * m ) ) )**( 1/2 ) ) ) ) )
        elif order == 1:
            kernel = 3 * ( a )**( 3 ) * g * tmp2 * (ell)**(-5) * n
        elif order == 2:
            kernel = ( -15/2 * ( a )**( 3 ) * g * tmp2 * tmp1 * (ell)**(-7) * n + ( 3 * ( a )**( 3 ) * g * m * (ell)**(-5) * n + 9 * ( a )**( 2 ) * g * tmp2 * (ell)**(-5) * n ) )
        elif order == 3:
            kernel = ( 18 * a * g * tmp2 * (ell)**(-5) * n + ( 3 * ( a )**( 3 ) * g * ( -5 * m * tmp1 * (ell)**(-7) + tmp2 * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + 18 * ( a )**( 2 ) * g * ( -5/2 * tmp2 * tmp1 * (ell)**(-7) * n + m * (ell)**(-5) * n ) ) )
        elif order == 4:
            kernel = ( 18 * g * tmp2 * (ell)**(-5) * n + ( 3 * ( a )**( 3 ) * g * ( tmp2 * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 3 * m * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + ( 27 * ( a )**( 2 ) * g * ( -5 * m * tmp1 * (ell)**(-7) + tmp2 * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + 54 * a * g * ( -5/2 * tmp2 * tmp1 * (ell)**(-7) * n + m * (ell)**(-5) * n ) ) ) )
        elif order == 5:
            kernel = ( 3 * ( a )**( 3 ) * g * ( tmp2 * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 4 * m * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) * n + ( 36 * ( a )**( 2 ) * g * ( tmp2 * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 3 * m * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + ( 108 * a * g * ( -5 * m * tmp1 * (ell)**(-7) + tmp2 * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + 72 * g * ( -5/2 * tmp2 * tmp1 * (ell)**(-7) * n + m * (ell)**(-5) * n ) ) ) )
        elif order == 6:
            kernel = ( 3 * ( a )**( 3 ) * g * ( tmp2 * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 5 * m * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) * n + ( 45 * ( a )**( 2 ) * g * ( tmp2 * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 4 * m * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) * n + ( 180 * a * g * ( tmp2 * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 3 * m * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n + 180 * g * ( -5 * m * tmp1 * (ell)**(-7) + tmp2 * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n ) ) )
        elif order == 7:
            kernel = ( 3 * ( a )**( 3 ) * g * ( tmp2 * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 6 * m * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) * n + ( 54 * ( a )**( 2 ) * g * ( tmp2 * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 5 * m * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) * n + ( 270 * a * g * ( tmp2 * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 4 * m * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) * n + 360 * g * ( tmp2 * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) + 3 * m * ( 35/4 * ( tmp1 )**( 2 ) * (ell)**(-9) + -5 * (ell)**(-7) ) ) * n ) ) )
        elif order == 8:
            kernel = ( 3 * ( a )**( 3 ) * g * ( tmp2 * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + 7 * m * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) ) * n + ( 63 * ( a )**( 2 ) * g * ( tmp2 * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 6 * m * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) * n + ( 378 * a * g * ( tmp2 * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 5 * m * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) * n + 630 * g * ( tmp2 * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) + 4 * m * ( -315/8 * ( tmp1 )**( 3 ) * (ell)**(-11) + 105/2 * tmp1 * (ell)**(-9) ) ) * n ) ) )
        elif order == 9:
            kernel = ( 3 * ( a )**( 3 ) * g * ( tmp2 * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) + 8 * m * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) ) * n + ( 72 * ( a )**( 2 ) * g * ( tmp2 * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + 7 * m * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) ) * n + ( 504 * a * g * ( tmp2 * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 6 * m * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) * n + 1008 * g * ( tmp2 * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) + 5 * m * ( 3465/16 * ( tmp1 )**( 4 ) * (ell)**(-13) + ( -945/2 * ( tmp1 )**( 2 ) * (ell)**(-11) + 105 * (ell)**(-9) ) ) ) * n ) ) )
        elif order == 10:
            kernel = ( 3 * ( a )**( 3 ) * g * ( tmp2 * ( -4583103525/512 * ( tmp1 )**( 9 ) * (ell)**(-23) + ( 1964187225/32 * ( tmp1 )**( 7 ) * (ell)**(-21) + ( -2170943775/16 * ( tmp1 )**( 5 ) * (ell)**(-19) + ( 212837625/2 * ( tmp1 )**( 3 ) * (ell)**(-17) + -42567525/2 * tmp1 * (ell)**(-15) ) ) ) ) + 9 * m * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) ) * n + ( 81 * ( a )**( 2 ) * g * ( tmp2 * ( 218243025/256 * ( tmp1 )**( 8 ) * (ell)**(-21) + ( -80405325/16 * ( tmp1 )**( 6 ) * (ell)**(-19) + ( 70945875/8 * ( tmp1 )**( 4 ) * (ell)**(-17) + ( -4729725 * ( tmp1 )**( 2 ) * (ell)**(-15) + 363825 * (ell)**(-13) ) ) ) ) + 8 * m * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) ) * n + ( 648 * a * g * ( tmp2 * ( -11486475/128 * ( tmp1 )**( 7 ) * (ell)**(-19) + ( 14189175/32 * ( tmp1 )**( 5 ) * (ell)**(-17) + ( -4729725/8 * ( tmp1 )**( 3 ) * (ell)**(-15) + 363825/2 * tmp1 * (ell)**(-13) ) ) ) + 7 * m * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) ) * n + 1512 * g * ( tmp2 * ( 675675/64 * ( tmp1 )**( 6 ) * (ell)**(-17) + ( -675675/16 * ( tmp1 )**( 4 ) * (ell)**(-15) + ( 155925/4 * ( tmp1 )**( 2 ) * (ell)**(-13) + -4725 * (ell)**(-11) ) ) ) + 6 * m * ( -45045/32 * ( tmp1 )**( 5 ) * (ell)**(-15) + ( 17325/4 * ( tmp1 )**( 3 ) * (ell)**(-13) + -4725/2 * tmp1 * (ell)**(-11) ) ) ) * n ) ) )
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**2 * sy.cos(c) \
                * (3*(a*(sy.cos(d)*sy.sin(c)-sy.sin(d)*sy.cos(c)*sy.cos(e-f))) \
                * (a * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f)) - b)
                / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**5)
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    return kernel
    

def cal_Vzz_kernel(r_cal, phi_cal, lambda_cal, \
        r_tess, phi_tess, lambda_tess, \
        order, is_linear_density):
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
    order: int
        Differentiation of kernel function.
    is_linear_density: bool
        If the tesseroid have linear varying density.
        
    Returns
    -------
    kernel: float
        Kernel of gravitational gradient Vzz.
    """
    a, b, c, d, e, f = r_tess, r_cal, phi_tess, phi_cal, lambda_tess, lambda_cal
    # Common part of taylor series.
    temp6 = np.cos(c)
    temp7 = np.cos(d)*np.cos(e - f)
    temp1 = temp6 * temp7
    temp2 = np.sin(c)*np.sin(d)
    temp3 = temp1 + temp2
    temp32 = temp3 * temp3
    temp4 = a - b*(temp3)
    temp42 = temp4 * temp4
    temp43 = temp42 * temp4
    # temp44 = temp43 * temp4
    # temp45 = temp44 * temp4
    # temp46 = temp45 * temp4
    # temp47 = temp46 * temp4
    # temp48 = temp47 * temp4
    # temp49 = temp48 * temp4
    temp5 = a*(temp3) - b
    temp52 = temp5 * temp5
    a2 = a*a
    a3 = a2 * a
    ell = a2 - 2*a*b*(temp3) + b*b
    ell2 = ell * ell
    ell23 = (ell)**(3/2)
    ell25 = (ell) * ell23
    temp8 = (temp5) * (temp3)

    if is_linear_density:
        if order==0:
            
            if np.abs(temp3-1)<1e-15:
                kernel = -2 * temp6 * ( a + ( 1/2 * ( ( a -1 * b ) )**( -2 ) * ( b )**( 2 ) * ( -6 * a + 5 * b ) + 3 \
                    * b * np.log( ( b-a) ) ) )
            elif np.abs(temp3+1)<1e-15:
                kernel = 2 * temp6 * ( a + ( -1/2 * ( b )**( 2 ) * ( ( a + b ) )**( -2 ) * ( 6 * a + 5 * b ) + -3 * b \
                    * np.log( ( a + b ) ) ) )
            else:
                kernel =  ell**( -3/2 ) * temp6 * ( 3 * a * ( b \
                    )**( 3 ) * temp3 * ( 7 + -20 * ( temp3 )**( 2 ) ) + ( 2 * a3 * b * temp3 * ( 9 + -20 * ( temp3 \
                    )**( 2 ) ) + ( ( a )**( 4 ) * ( -1 + 3 * temp32 ) + ( ( b )**( 4 ) * ( -4 + 15 * ( \
                    temp3 )**( 2 ) ) + ( 6 * a2 * ( b )**( 2 ) * ( -1 + ( -1 * temp32 + 10 * ( temp3 \
                    )**( 4 ) ) ) + 3 * b * temp3 * ( ( ell ) )**( \
                    3/2 ) * ( -3 + 5 * temp32 ) * np.arctanh( temp4  *  ell**( -1/2 ) ) ) ) ) ) )
        elif order==1:
            ell25 = ell23 * ell
            kernel = a3*(3*temp52/(ell25) - 1/(ell23))*temp6
        elif order==2:
            ell25 = ell23 * ell
            ell27 = ell25 * ell
            kernel = a3*(3*(-5*a + 5*b*(temp3))*temp52/ell27 - (-3*a + 3*b*(temp3))/(ell25) + 3*(temp5)*(2*temp2 + 2*temp6*temp7)/(ell25))*temp6 + 3*a2*(3*temp52/(ell25) - 1/(ell23))*temp6
        elif order==3:
            kernel = 3*a*(a2*(35*temp42*temp52/ell2 - 5*temp42/(ell) - 20*(temp4)*temp8/(ell) - 5*temp52/(ell) + 2*temp32 + 1)/(ell) + 6*a*(a - b*(temp3) - 5*(temp4)*temp52/(ell) + 2*temp8)/(ell) + 6*temp52/(ell) - 2)*temp6/(ell23)
        elif order==4:
            kernel = 3*(-5*a3*(3*a - 3*b*(temp3) + 63*temp43*temp52/ell2 - 7*temp43/(ell) - 42*temp42*temp8/(ell) - 21*(temp4)*temp52/(ell) + 6*(temp4)*temp32 + 6*temp8)/ell2 + 9*a2*(35*temp42*temp52/ell2 - 5*temp42/(ell) - 20*(temp4)*temp8/(ell) - 5*temp52/(ell) + 2*temp32 + 1)/(ell) + 18*a*(a - b*(temp3) - 5*(temp4)*temp52/(ell) + 2*temp8)/(ell) + 6*temp52/(ell) - 2)*temp6/(ell23)
        elif order==5:
            ell3 = ell2 * ell
            temp44 = temp43 * temp4
            kernel = 9*(5*a3*(231*temp44*temp52/ell3 - 21*temp44/ell2 - 168*temp43*temp8/ell2 - 126*temp42*temp52/ell2 + 28*temp42*temp32/(ell) + 14*temp42/(ell) + 56*(temp4)*temp8/(ell) + 7*temp52/(ell) - 4*temp32 - 1)/(ell) - 20*a2*(3*a - 3*b*(temp3) + 63*temp43*temp52/ell2 - 7*temp43/(ell) - 42*temp42*temp8/(ell) - 21*(temp4)*temp52/(ell) + 6*(temp4)*temp32 + 6*temp8)/(ell) + 12*a*(35*temp42*temp52/ell2 - 5*temp42/(ell) - 20*(temp4)*temp8/(ell) - 5*temp52/(ell) + 2*temp32 + 1) + 8*a - 8*b*(temp3) - 40*(temp4)*temp52/(ell) + 16*temp8)*temp6/(ell25)
        elif order==6:
            ell3 = ell2 * ell
            temp44 = temp43 * temp4
            temp45 = temp44 * temp4
            kernel = 45*(7*a3*(5*a - 5*b*(temp3) - 429*temp45*temp52/ell3 + 33*temp45/ell2 + 330*temp44*temp8/ell2 + 330*temp43*temp52/ell2 - 60*temp43*temp32/(ell) - 30*temp43/(ell) - 180*temp42*temp8/(ell) - 45*(temp4)*temp52/(ell) + 20*(temp4)*temp32 + 10*temp8)/ell2 + 15*a2*(231*temp44*temp52/ell3 - 21*temp44/ell2 - 168*temp43*temp8/ell2 - 126*temp42*temp52/ell2 + 28*temp42*temp32/(ell) + 14*temp42/(ell) + 56*(temp4)*temp8/(ell) + 7*temp52/(ell) - 4*temp32 - 1)/(ell) - 20*a*(3*a - 3*b*(temp3) + 63*temp43*temp52/ell2 - 7*temp43/(ell) - 42*temp42*temp8/(ell) - 21*(temp4)*temp52/(ell) + 6*(temp4)*temp32 + 6*temp8)/(ell) + 140*temp42*temp52/ell2 - 20*temp42/(ell) - 80*(temp4)*temp8/(ell) - 20*temp52/(ell) + 8*temp32 + 4)*temp6/(ell25)
        elif order==7:
            ell3 = ell2 * ell
            ell4 = ell3 * ell
            temp44 = temp43 * temp4
            temp45 = temp44 * temp4
            temp46 = temp45 * temp4
            kernel = 45*(7*a3*(6435*temp46*temp52/ell4 - 429*temp46/ell3 - 5148*temp45*temp8/ell3 - 6435*temp44*temp52/ell3 + 990*temp44*temp32/ell2 + 495*temp44/ell2 + 3960*temp43*temp8/ell2 + 1485*temp42*temp52/ell2 - 540*temp42*temp32/(ell) - 135*temp42/(ell) - 540*(temp4)*temp8/(ell) - 45*temp52/(ell) + 30*temp32 + 5)/(ell) + 126*a2*(5*a - 5*b*(temp3) - 429*temp45*temp52/ell3 + 33*temp45/ell2 + 330*temp44*temp8/ell2 + 330*temp43*temp52/ell2 - 60*temp43*temp32/(ell) - 30*temp43/(ell) - 180*temp42*temp8/(ell) - 45*(temp4)*temp52/(ell) + 20*(temp4)*temp32 + 10*temp8)/(ell) + 90*a*(231*temp44*temp52/ell3 - 21*temp44/ell2 - 168*temp43*temp8/ell2 - 126*temp42*temp52/ell2 + 28*temp42*temp32/(ell) + 14*temp42/(ell) + 56*(temp4)*temp8/(ell) + 7*temp52/(ell) - 4*temp32 - 1) - 120*a + 120*b*(temp3) - 2520*temp43*temp52/ell2 + 280*temp43/(ell) + 1680*temp42*temp8/(ell) + 840*(temp4)*temp52/(ell) - 240*(temp4)*temp32 - 240*temp8)*temp6/ell27
        elif order==8:
            ell3 = ell2 * ell
            ell4 = ell3 * ell
            temp44 = temp43 * temp4
            temp45 = temp44 * temp4
            temp46 = temp45 * temp4
            temp47 = temp46 * temp4
            kernel = 945*(-3*a3*(35*a - 35*b*(temp3) + 12155*temp47*temp52/ell4 - 715*temp47/ell3 - 10010*temp46*temp8/ell3 - 15015*temp45*temp52/ell3 + 2002*temp45*temp32/ell2 + 1001*temp45/ell2 + 10010*temp44*temp8/ell2 + 5005*temp43*temp52/ell2 - 1540*temp43*temp32/(ell) - 385*temp43/(ell) - 2310*temp42*temp8/(ell) - 385*(temp4)*temp52/(ell) + 210*(temp4)*temp32 + 70*temp8)/ell2 + 7*a2*(6435*temp46*temp52/ell4 - 429*temp46/ell3 - 5148*temp45*temp8/ell3 - 6435*temp44*temp52/ell3 + 990*temp44*temp32/ell2 + 495*temp44/ell2 + 3960*temp43*temp8/ell2 + 1485*temp42*temp52/ell2 - 540*temp42*temp32/(ell) - 135*temp42/(ell) - 540*(temp4)*temp8/(ell) - 45*temp52/(ell) + 30*temp32 + 5)/(ell) + 42*a*(5*a - 5*b*(temp3) - 429*temp45*temp52/ell3 + 33*temp45/ell2 + 330*temp44*temp8/ell2 + 330*temp43*temp52/ell2 - 60*temp43*temp32/(ell) - 30*temp43/(ell) - 180*temp42*temp8/(ell) - 45*(temp4)*temp52/(ell) + 20*(temp4)*temp32 + 10*temp8)/(ell) + 2310*temp44*temp52/ell3 - 210*temp44/ell2 - 1680*temp43*temp8/ell2 - 1260*temp42*temp52/ell2 + 280*temp42*temp32/(ell) + 140*temp42/(ell) + 560*(temp4)*temp8/(ell) + 70*temp52/(ell) - 40*temp32 - 10)*temp6/ell27
        elif order==9:
            ell3 = ell2 * ell
            ell4 = ell3 * ell
            ell5 = ell4 * ell
            ell29 = ell3 * ell23
            temp44 = temp43 * temp4
            temp45 = temp44 * temp4
            temp46 = temp45 * temp4
            temp47 = temp46 * temp4
            temp48 = temp47 * temp4
            kernel = 945*(15*a3*(46189*temp48*temp52/ell5 - 2431*temp48/ell4 - 38896*temp47*temp8/ell4 - 68068*temp46*temp52/ell4 + 8008*temp46*temp32/ell3 + 4004*temp46/ell3 + 48048*temp45*temp8/ell3 + 30030*temp44*temp52/ell3 - 8008*temp44*temp32/ell2 - 2002*temp44/ell2 - 16016*temp43*temp8/ell2 - 4004*temp42*temp52/ell2 + 1848*temp42*temp32/(ell) + 308*temp42/(ell) + 1232*(temp4)*temp8/(ell) + 77*temp52/(ell) - 56*temp32 - 7)/(ell) - 72*a2*(35*a - 35*b*(temp3) + 12155*temp47*temp52/ell4 - 715*temp47/ell3 - 10010*temp46*temp8/ell3 - 15015*temp45*temp52/ell3 + 2002*temp45*temp32/ell2 + 1001*temp45/ell2 + 10010*temp44*temp8/ell2 + 5005*temp43*temp52/ell2 - 1540*temp43*temp32/(ell) - 385*temp43/(ell) - 2310*temp42*temp8/(ell) - 385*(temp4)*temp52/(ell) + 210*(temp4)*temp32 + 70*temp8)/(ell) + 56*a*(6435*temp46*temp52/ell4 - 429*temp46/ell3 - 5148*temp45*temp8/ell3 - 6435*temp44*temp52/ell3 + 990*temp44*temp32/ell2 + 495*temp44/ell2 + 3960*temp43*temp8/ell2 + 1485*temp42*temp52/ell2 - 540*temp42*temp32/(ell) - 135*temp42/(ell) - 540*(temp4)*temp8/(ell) - 45*temp52/(ell) + 30*temp32 + 5) + 560*a - 560*b*(temp3) - 48048*temp45*temp52/ell3 + 3696*temp45/ell2 + 36960*temp44*temp8/ell2 + 36960*temp43*temp52/ell2 - 6720*temp43*temp32/(ell) - 3360*temp43/(ell) - 20160*temp42*temp8/(ell) - 5040*(temp4)*temp52/(ell) + 2240*(temp4)*temp32 + 1120*temp8)*temp6/ell29
        elif order==10:
            ell3 = ell2 * ell
            ell5 = ell4 * ell
            ell29 = ell3 * ell23
            temp44 = temp43 * temp4
            temp45 = temp44 * temp4
            temp46 = temp45 * temp4
            temp47 = temp46 * temp4
            temp48 = temp47 * temp4
            temp49 = temp48 * temp4
            kernel = 2835*(55*a3*(63*a - 63*b*(temp3) - 88179*temp49*temp52/ell5 + 4199*temp49/ell4 + 75582*temp48*temp8/ell4 + 151164*temp47*temp52/ell4 - 15912*temp47*temp32/ell3 - 7956*temp47/ell3 - 111384*temp46*temp8/ell3 - 83538*temp45*temp52/ell3 + 19656*temp45*temp32/ell2 + 4914*temp45/ell2 + 49140*temp44*temp8/ell2 + 16380*temp43*temp52/ell2 - 6552*temp43*temp32/(ell) - 1092*temp43/(ell) - 6552*temp42*temp8/(ell) - 819*(temp4)*temp52/(ell) + 504*(temp4)*temp32 + 126*temp8)/ell2 + 135*a2*(46189*temp48*temp52/ell5 - 2431*temp48/ell4 - 38896*temp47*temp8/ell4 - 68068*temp46*temp52/ell4 + 8008*temp46*temp32/ell3 + 4004*temp46/ell3 + 48048*temp45*temp8/ell3 + 30030*temp44*temp52/ell3 - 8008*temp44*temp32/ell2 - 2002*temp44/ell2 - 16016*temp43*temp8/ell2 - 4004*temp42*temp52/ell2 + 1848*temp42*temp32/(ell) + 308*temp42/(ell) + 1232*(temp4)*temp8/(ell) + 77*temp52/(ell) - 56*temp32 - 7)/(ell) - 216*a*(35*a - 35*b*(temp3) + 12155*temp47*temp52/ell4 - 715*temp47/ell3 - 10010*temp46*temp8/ell3 - 15015*temp45*temp52/ell3 + 2002*temp45*temp32/ell2 + 1001*temp45/ell2 + 10010*temp44*temp8/ell2 + 5005*temp43*temp52/ell2 - 1540*temp43*temp32/(ell) - 385*temp43/(ell) - 2310*temp42*temp8/(ell) - 385*(temp4)*temp52/(ell) + 210*(temp4)*temp32 + 70*temp8)/(ell) + 360360*temp46*temp52/ell4 - 24024*temp46/ell3 - 288288*temp45*temp8/ell3 - 360360*temp44*temp52/ell3 + 55440*temp44*temp32/ell2 + 27720*temp44/ell2 + 221760*temp43*temp8/ell2 + 83160*temp42*temp52/ell2 - 30240*temp42*temp32/(ell) - 7560*temp42/(ell) - 30240*(temp4)*temp8/(ell) - 2520*temp52/(ell) + 1680*temp32 + 280)*temp6/ell29
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**3 * sy.cos(c) \
                * (3 * ((a * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f)) - b))**2 \
                / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**5 \
                - 1 / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**3)
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    else:
        if order==0:
            
            if np.abs(temp3-1)<1e-15:
                kernel = -2 * temp6 * ( 1/2 * ( ( a - b ) )**( -2 ) * b * ( -4 * a + 3 * b ) + np.log( ( -a + b ) ) )
            elif np.abs(temp3+1)<1e-15:
                kernel = ( b * ( ( a + b ) )**( -2 ) * ( 4 * a + 3 * b ) * temp6 + 2 * temp6 * np.log( ( a + b ) ) )
            else:
                kernel = ell**( -3/2 ) * temp6 * ( 3 * ( b )**( 3 \
                    ) * temp3 + ( a * ( b )**( 2 ) * ( 1 -12 * temp32 ) + ( a3 \
                    * ( 2 -8 * temp32 ) + ( 2 * a2 * b * ( temp3 + 6 * temp32*temp3 ) + ell**( 3/2 ) \
                    * ( -1 + 3 * temp32 ) * np.arctanh(temp4 * ( ( ell) )**( -1/2 ) ) ) ) \
                    ) )
        elif order==1:
            ell25 = ell23 * ell
            kernel = a2*(3*temp52/(ell25) - 1/(ell23))*temp6
        elif order==2:
            ell25 = ell23 * ell
            ell27 = ell25 * ell
            kernel = a2*(3*(-5*a + 5*b*(temp3))*temp52/ell27 - (-3*a + 3*b*(temp3))/(ell25) + 3*(temp5)*(2*temp2 + 2*temp6*temp7)/(ell25))*temp6 + 2*a*(3*temp52/(ell25) - 1/(ell23))*temp6
        elif order==3:
            kernel = (3*a2*(35*temp42*temp52/ell2 - 5*temp42/(ell) - 20*(temp4)*temp8/(ell) - 5*temp52/(ell) + 2*temp32 + 1)/(ell) + 12*a*(a - b*(temp3) - 5*(temp4)*temp52/(ell) + 2*temp8)/(ell) + 6*temp52/(ell) - 2)*temp6/(ell23)
        elif order==4:
            kernel = 3*(-5*a2*(3*a - 3*b*(temp3) + 63*temp43*temp52/ell2 - 7*temp43/(ell) - 42*temp42*temp8/(ell) - 21*(temp4)*temp52/(ell) + 6*(temp4)*temp32 + 6*temp8)/(ell) + 6*a*(35*temp42*temp52/ell2 - 5*temp42/(ell) - 20*(temp4)*temp8/(ell) - 5*temp52/(ell) + 2*temp32 + 1) + 6*a - 6*b*(temp3) - 30*(temp4)*temp52/(ell) + 12*temp8)*temp6/(ell25)
        elif order==5:
            ell3 = ell2 * ell
            temp44 = temp43 * temp4
            kernel = 3*(15*a2*(231*temp44*temp52/ell3 - 21*temp44/ell2 - 168*temp43*temp8/ell2 - 126*temp42*temp52/ell2 + 28*temp42*temp32/(ell) + 14*temp42/(ell) + 56*(temp4)*temp8/(ell) + 7*temp52/(ell) - 4*temp32 - 1)/(ell) - 40*a*(3*a - 3*b*(temp3) + 63*temp43*temp52/ell2 - 7*temp43/(ell) - 42*temp42*temp8/(ell) - 21*(temp4)*temp52/(ell) + 6*(temp4)*temp32 + 6*temp8)/(ell) + 420*temp42*temp52/ell2 - 60*temp42/(ell) - 240*(temp4)*temp8/(ell) - 60*temp52/(ell) + 24*temp32 + 12)*temp6/(ell25)
        elif order==6:
            ell3 = ell2 * ell
            ell27 = ell2 * ell23
            temp44 = temp43 * temp4
            temp45 = temp44 * temp4
            kernel = 15*(21*a2*(5*a - 5*b*(temp3) - 429*temp45*temp52/ell3 + 33*temp45/ell2 + 330*temp44*temp8/ell2 + 330*temp43*temp52/ell2 - 60*temp43*temp32/(ell) - 30*temp43/(ell) - 180*temp42*temp8/(ell) - 45*(temp4)*temp52/(ell) + 20*(temp4)*temp32 + 10*temp8)/(ell) + 30*a*(231*temp44*temp52/ell3 - 21*temp44/ell2 - 168*temp43*temp8/ell2 - 126*temp42*temp52/ell2 + 28*temp42*temp32/(ell) + 14*temp42/(ell) + 56*(temp4)*temp8/(ell) + 7*temp52/(ell) - 4*temp32 - 1) - 60*a + 60*b*(temp3) - 1260*temp43*temp52/ell2 + 140*temp43/(ell) + 840*temp42*temp8/(ell) + 420*(temp4)*temp52/(ell) - 120*(temp4)*temp32 - 120*temp8)*temp6/ell27
        elif order==7:
            ell3 = ell2 * ell
            ell4 = ell3 * ell
            temp44 = temp43 * temp4
            temp45 = temp44 * temp4
            temp46 = temp45 * temp4
            kernel = 45*(7*a2*(6435*temp46*temp52/ell4 - 429*temp46/ell3 - 5148*temp45*temp8/ell3 - 6435*temp44*temp52/ell3 + 990*temp44*temp32/ell2 + 495*temp44/ell2 + 3960*temp43*temp8/ell2 + 1485*temp42*temp52/ell2 - 540*temp42*temp32/(ell) - 135*temp42/(ell) - 540*(temp4)*temp8/(ell) - 45*temp52/(ell) + 30*temp32 + 5)/(ell) + 84*a*(5*a - 5*b*(temp3) - 429*temp45*temp52/ell3 + 33*temp45/ell2 + 330*temp44*temp8/ell2 + 330*temp43*temp52/ell2 - 60*temp43*temp32/(ell) - 30*temp43/(ell) - 180*temp42*temp8/(ell) - 45*(temp4)*temp52/(ell) + 20*(temp4)*temp32 + 10*temp8)/(ell) + 6930*temp44*temp52/ell3 - 630*temp44/ell2 - 5040*temp43*temp8/ell2 - 3780*temp42*temp52/ell2 + 840*temp42*temp32/(ell) + 420*temp42/(ell) + 1680*(temp4)*temp8/(ell) + 210*temp52/(ell) - 120*temp32 - 30)*temp6/ell27
        elif order==8:
            ell3 = ell2 * ell
            ell4 = ell3 * ell
            ell29 = ell3 * ell23
            temp44 = temp43 * temp4
            temp45 = temp44 * temp4
            temp46 = temp45 * temp4
            temp47 = temp46 * temp4
            kernel = 315*(-9*a2*(35*a - 35*b*(temp3) + 12155*temp47*temp52/ell4 - 715*temp47/ell3 - 10010*temp46*temp8/ell3 - 15015*temp45*temp52/ell3 + 2002*temp45*temp32/ell2 + 1001*temp45/ell2 + 10010*temp44*temp8/ell2 + 5005*temp43*temp52/ell2 - 1540*temp43*temp32/(ell) - 385*temp43/(ell) - 2310*temp42*temp8/(ell) - 385*(temp4)*temp52/(ell) + 210*(temp4)*temp32 + 70*temp8)/(ell) + 14*a*(6435*temp46*temp52/ell4 - 429*temp46/ell3 - 5148*temp45*temp8/ell3 - 6435*temp44*temp52/ell3 + 990*temp44*temp32/ell2 + 495*temp44/ell2 + 3960*temp43*temp8/ell2 + 1485*temp42*temp52/ell2 - 540*temp42*temp32/(ell) - 135*temp42/(ell) - 540*(temp4)*temp8/(ell) - 45*temp52/(ell) + 30*temp32 + 5) + 210*a - 210*b*(temp3) - 18018*temp45*temp52/ell3 + 1386*temp45/ell2 + 13860*temp44*temp8/ell2 + 13860*temp43*temp52/ell2 - 2520*temp43*temp32/(ell) - 1260*temp43/(ell) - 7560*temp42*temp8/(ell) - 1890*(temp4)*temp52/(ell) + 840*(temp4)*temp32 + 420*temp8)*temp6/ell29
        elif order==9:
            ell3 = ell2 * ell
            ell4 = ell3 * ell
            ell5 = ell4 * ell
            ell29 = ell3 * ell23
            temp44 = temp43 * temp4
            temp45 = temp44 * temp4
            temp46 = temp45 * temp4
            temp47 = temp46 * temp4
            temp48 = temp47 * temp4
            kernel = 315*(45*a2*(46189*temp48*temp52/ell5 - 2431*temp48/ell4 - 38896*temp47*temp8/ell4 - 68068*temp46*temp52/ell4 + 8008*temp46*temp32/ell3 + 4004*temp46/ell3 + 48048*temp45*temp8/ell3 + 30030*temp44*temp52/ell3 - 8008*temp44*temp32/ell2 - 2002*temp44/ell2 - 16016*temp43*temp8/ell2 - 4004*temp42*temp52/ell2 + 1848*temp42*temp32/(ell) + 308*temp42/(ell) + 1232*(temp4)*temp8/(ell) + 77*temp52/(ell) - 56*temp32 - 7)/(ell) - 144*a*(35*a - 35*b*(temp3) + 12155*temp47*temp52/ell4 - 715*temp47/ell3 - 10010*temp46*temp8/ell3 - 15015*temp45*temp52/ell3 + 2002*temp45*temp32/ell2 + 1001*temp45/ell2 + 10010*temp44*temp8/ell2 + 5005*temp43*temp52/ell2 - 1540*temp43*temp32/(ell) - 385*temp43/(ell) - 2310*temp42*temp8/(ell) - 385*(temp4)*temp52/(ell) + 210*(temp4)*temp32 + 70*temp8)/(ell) + 360360*temp46*temp52/ell4 - 24024*temp46/ell3 - 288288*temp45*temp8/ell3 - 360360*temp44*temp52/ell3 + 55440*temp44*temp32/ell2 + 27720*temp44/ell2 + 221760*temp43*temp8/ell2 + 83160*temp42*temp52/ell2 - 30240*temp42*temp32/(ell) - 7560*temp42/(ell) - 30240*(temp4)*temp8/(ell) - 2520*temp52/(ell) + 1680*temp32 + 280)*temp6/ell29
        elif order==10:
            ell3 = ell2 * ell
            ell4 = ell3 * ell
            ell5 = ell4 * ell
            ell211 = ell4 * ell23
            temp44 = temp43 * temp4
            temp45 = temp44 * temp4
            temp46 = temp45 * temp4
            temp47 = temp46 * temp4
            temp48 = temp47 * temp4
            temp49 = temp48 * temp4
            kernel = 2835*(55*a2*(63*a - 63*b*(temp3) - 88179*temp49*temp52/ell5 + 4199*temp49/ell4 + 75582*temp48*temp8/ell4 + 151164*temp47*temp52/ell4 - 15912*temp47*temp32/ell3 - 7956*temp47/ell3 - 111384*temp46*temp8/ell3 - 83538*temp45*temp52/ell3 + 19656*temp45*temp32/ell2 + 4914*temp45/ell2 + 49140*temp44*temp8/ell2 + 16380*temp43*temp52/ell2 - 6552*temp43*temp32/(ell) - 1092*temp43/(ell) - 6552*temp42*temp8/(ell) - 819*(temp4)*temp52/(ell) + 504*(temp4)*temp32 + 126*temp8)/(ell) + 90*a*(46189*temp48*temp52/ell5 - 2431*temp48/ell4 - 38896*temp47*temp8/ell4 - 68068*temp46*temp52/ell4 + 8008*temp46*temp32/ell3 + 4004*temp46/ell3 + 48048*temp45*temp8/ell3 + 30030*temp44*temp52/ell3 - 8008*temp44*temp32/ell2 - 2002*temp44/ell2 - 16016*temp43*temp8/ell2 - 4004*temp42*temp52/ell2 + 1848*temp42*temp32/(ell) + 308*temp42/(ell) + 1232*(temp4)*temp8/(ell) + 77*temp52/(ell) - 56*temp32 - 7) - 2520*a + 2520*b*(temp3) - 875160*temp47*temp52/ell4 + 51480*temp47/ell3 + 720720*temp46*temp8/ell3 + 1081080*temp45*temp52/ell3 - 144144*temp45*temp32/ell2 - 72072*temp45/ell2 - 720720*temp44*temp8/ell2 - 360360*temp43*temp52/ell2 + 110880*temp43*temp32/(ell) + 27720*temp43/(ell) + 166320*temp42*temp8/(ell) + 27720*(temp4)*temp52/(ell) - 15120*(temp4)*temp32 - 5040*temp8)*temp6/ell211
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**2 * sy.cos(c) \
                * (3 * ((a * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f)) - b))**2 \
                / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**5 \
                - 1 / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**3)
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    
    return kernel


def cal_kernel(r_cal, phi_cal, lambda_cal, \
        r_tess, phi_tess, lambda_tess, order, \
        tag, is_linear_density):
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
    order: int
        Differentiation of kernel function.
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
            r_tess, phi_tess, lambda_tess, order, is_linear_density)             
    elif tag=='Vx':
        kernel = cal_Vx_kernel(r_cal, phi_cal, lambda_cal, \
            r_tess, phi_tess, lambda_tess, order, is_linear_density)                   
    elif tag=='Vy':
        kernel = cal_Vy_kernel(r_cal, phi_cal, lambda_cal, \
            r_tess, phi_tess, lambda_tess, order, is_linear_density)                     
    elif tag=='Vz':
        kernel = cal_Vz_kernel(r_cal, phi_cal, lambda_cal, \
            r_tess, phi_tess, lambda_tess, order, is_linear_density)                   
    elif tag=='Vxx':
        kernel = cal_Vxx_kernel(r_cal, phi_cal, lambda_cal, \
            r_tess, phi_tess, lambda_tess, order, is_linear_density)                   
    elif tag=='Vxy':
        kernel = cal_Vxy_kernel(r_cal, phi_cal, lambda_cal, \
            r_tess, phi_tess, lambda_tess, order, is_linear_density)                   
    elif tag=='Vxz':
        kernel = cal_Vxz_kernel(r_cal, phi_cal, lambda_cal, \
            r_tess, phi_tess, lambda_tess, order, is_linear_density)                    
    elif tag=='Vyy':
        kernel = cal_Vyy_kernel(r_cal, phi_cal, lambda_cal, \
            r_tess, phi_tess, lambda_tess, order, is_linear_density)                   
    elif tag=='Vyz':
        kernel = cal_Vyz_kernel(r_cal, phi_cal, lambda_cal, \
            r_tess, phi_tess, lambda_tess, order, is_linear_density) 
    else:
        kernel = cal_Vzz_kernel(r_cal, phi_cal, lambda_cal, \
            r_tess, phi_tess, lambda_tess, order, is_linear_density) 
    return kernel


def subdivision(r_cal, phi_cal, lambda_cal, \
        r0, r_max, phi_min, phi_max, lambda_min, lambda_max, \
        ratio, order, roots, weights, \
        tag, is_linear_density):
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
    r0: float
        Expansion point of Taylor series in meter.
    r_max: float
        Radius of tesseroid in meter.
    phi_min: float
        Min latitude of tesseroid in radian.
    phi_max: float
        Max latitude of tesseroid in radian.
    lambda_min: float
        Max longitude of tesseroid in radian.
    lambda_max: float
        Max longitude of tesseroid in radian.
    ratio: float
        Distance-size ratio, which is specified by the user. 
        The larger ratio is, the smaller tesseroid is divided, 
        and the higher accuracy of calculation is.
    order: int
        Differentiation of kernel function.
    roots: numpy.ndarray, float
        Roots of legendre polynomial.
    weights: numpy.ndarray, float
        Weights of quadrature.
    tag: string
        Kernel function to be calculated. 
        tag \in {'V', 'Vx', 'Vy', 'Vz', 
        'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz'}
    is_linear_density: bool
        If the tesseroid have linear varying density.
        
    Returns
    -------
    result: float
        The gravitational field generated by tesseroid after subdivision.
    """
    if is_linear_density:
        result_constant = 0
        result_linear = 0

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

                constant_temp, linear_temp = cal_single_tesseroid_gravitational_field(
                    r_cal, phi_cal, lambda_cal, 
                    r0, r_max, phi_min_temp, phi_max_temp, 
                    lambda_min_temp, lambda_max_temp, 
                    roots, weights, order, tag, ratio, is_linear_density)
                
                result_constant = result_constant + constant_temp
                result_linear = result_linear + linear_temp
                
        return result_constant, result_linear
    else:
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

                temp = cal_single_tesseroid_gravitational_field(
                    r_cal, phi_cal, lambda_cal, 
                    r0, r_max, phi_min_temp, phi_max_temp, 
                    lambda_min_temp, lambda_max_temp, 
                    roots, weights, order, tag, ratio, is_linear_density)
                result += temp 
        return result


def direct_cal_gravitational_field(r_cal, phi_cal, lambda_cal, 
        r0, phi_min, phi_max, lambda_min, lambda_max, 
        roots, weights, order, tag, is_linear_density):
    """
    Directly calculate the kernel of gravitational field.

    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: float
        Latitude of computation point in radian.
    lambda_cal: float
        Longitude of compuattion point in radian.
    r0: float
        Expansion point of Taylor series in meter.
    phi_min: float
        Min latitude of tesseroid in radian.
    phi_max: float
        Max latitude of tesseroid in radian.
    lambda_min: float
        Min longitude of tesseroid in radian.
    lambda_max: float
        Max longitude of tesseroid in radian.
    roots: numpy.ndarray, float
        Roots of legendre polynomial.
    weights: numpy.ndarray, float
        Weights of quadrature.
    order: int
        Differentiation of kernel function.
    tag: string
        Kernel function to be calculated. 
        tag \in {'V', 'Vx', 'Vy', 'Vz', 
        'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz'}
    is_linear_density: bool
        If the tesseroid have linear varying density.
        
    Returns
    -------
    result: float
        Gravitational field generated by tesseroid.
    """
    result_constant = 0
    result_linear = 0
    delta_lambda = lambda_max - lambda_min
    delta_phi = phi_max - phi_min

    if is_linear_density:
        for index_phi in range(len(roots)):
            for index_lambda in range(len(roots)):
                phi_tess_temp = (roots[index_phi] * delta_phi \
                    + phi_max + phi_min) / 2
                lambda_tess_temp = (roots[index_lambda] \
                    * delta_lambda + lambda_max + lambda_min) / 2
                result_constant += cal_kernel(r_cal, phi_cal, lambda_cal, \
                    r0, phi_tess_temp, lambda_tess_temp, order, tag, False) \
                    * weights[index_phi] * weights[index_lambda]
                result_linear += cal_kernel(r_cal, phi_cal, lambda_cal, \
                    r0, phi_tess_temp, lambda_tess_temp, order, tag, True) \
                    * weights[index_phi] * weights[index_lambda]
                    
        result_constant *= delta_phi * delta_lambda / 4
        result_linear *= delta_phi * delta_lambda / 4

        return result_constant, result_linear
    else:
        for index_phi in range(len(roots)):
            for index_lambda in range(len(roots)):
                phi_tess_temp = (roots[index_phi] * delta_phi \
                    + phi_max + phi_min) / 2
                lambda_tess_temp = (roots[index_lambda] \
                    * delta_lambda + lambda_max + lambda_min) / 2
                result_constant += cal_kernel(r_cal, phi_cal, lambda_cal, \
                    r0, phi_tess_temp, lambda_tess_temp, order, tag, False) \
                    * weights[index_phi] * weights[index_lambda]
                    
        result_constant *= delta_phi * delta_lambda / 4

        return result_constant


def cal_single_tesseroid_gravitational_field(
        r_cal, phi_cal, lambda_cal, 
        r0, r_max, phi_min, phi_max, lambda_min, lambda_max, 
        roots, weights, order, tag, ratio, 
        is_linear_density):
    """
    Calculate the gravitational field of a tesseroid.

    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: float
        Latitude of computation point in radian.
    lambda_cal: float
        Longitude of compuattion point in radian.
    r0: float
        Expansion point of Taylor series in meter.
    r_max: float
        Radius of tesseroid in meter.
    phi_min: float
        Min latitude of tesseroid in radian.
    phi_max: float
        Max latitude of tesseroid in radian.
    lambda_min: float
        Min longitude of tesseroid in radian.
    lambda_max: float
        Max longitude of tesseroid in radian.
    roots: numpy.ndarray, float
        Roots of legendre polynomial.
    weights: numpy.ndarray, float
        Weights of quadrature.
    order: int
        Differentiation of kernel function.
    tag: string
        Kernel function to be calculated. 
        tag \in {'V', 'Vx', 'Vy', 'Vz', 
        'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz'}
    ratio: float
        Distance-size ratio, which is specified by the user. 
        The larger ratio is, the smaller tesseroid is divided, 
        and the higher accuracy of calculation is.
    is_linear_density: bool
        If the tesseroid have linear varying density.
        
    Returns
    -------
    result: numpy.ndarray, float
        The gravitational field of a tesseroid.
    """
        
    phi0 = (phi_max + phi_min) / 2
    lambda0 = (lambda_max + lambda_min) / 2
    ell0 = cal_distance(r_cal, phi_cal, lambda_cal, r_max, phi0, lambda0)
    
    L_phi = r_max * (phi_max - phi_min)
    L_lambda = r_max * np.cos(phi_min) * (lambda_max - lambda_min)
    if is_linear_density:
        if (ell0/L_phi>ratio) and (ell0/L_lambda>ratio):
            result_constant, result_linear = direct_cal_gravitational_field(r_cal, phi_cal, lambda_cal, 
                r0, phi_min, phi_max, lambda_min, lambda_max, 
                roots, weights, order, tag, is_linear_density)
        else:
            result_constant, result_linear = subdivision(r_cal, phi_cal, lambda_cal, 
                r0, r_max, phi_min, phi_max, lambda_min, lambda_max,
                ratio, order, roots, weights, tag, is_linear_density)
        return result_constant, result_linear
    else:
        if (ell0/L_phi>ratio) and (ell0/L_lambda>ratio):
            result = direct_cal_gravitational_field(r_cal, phi_cal, lambda_cal, 
                r0, phi_min, phi_max, lambda_min, lambda_max, 
                roots, weights, order, tag, is_linear_density)
        else:
            result = subdivision(r_cal, phi_cal, lambda_cal, 
                r0, r_max, phi_min, phi_max, lambda_min, lambda_max,
                ratio, order, roots, weights, tag, is_linear_density)
        return result


def double_data(data, lambda0, delta_lambda, shape, tag):
    """
    
    """
    data2 = np.fliplr(data[:, 1:-1])
    if tag in ['Vy', 'Vxy', 'Vyz']:
        data2 = - data2
    idx_lambda0 = int((lambda0 - (-np.pi + delta_lambda / 2)) / delta_lambda + 1e-10) + 1
    if lambda0 < 0:
        data2 = np.hstack((data2, data))
        tmp_idx = int(shape[1]/2+1e-15)-idx_lambda0
        data3 = np.hstack((data2[:, tmp_idx:], data2[:, :tmp_idx]))
        return data3
    else:
        data2 = np.hstack((data, data2))
        tmp_idx = int(shape[1]/2+1e-15)+shape[1]-idx_lambda0+1
        data3 = np.hstack((data2[:, tmp_idx:], data2[:, :tmp_idx]))
        return data3


def cal_order(r_cal, phi_cal, lambda_cal,
    r0, r_max, phi_min, phi_max, lambda_min, lambda_max,
    shape, roots, weights, order, ratio, 
    tag, is_linear_density):
    """
    Calculate the order-differentiation of kernel function.

    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: numpy.ndarray, float
        Latitude of computation point in radian.
    lambda_cal: numpy.ndarray, float
        Longitude of computation point in radian.
    r0: float
        Expansion point of Taylor series in meter.
    r_max: float
        Radius of tesseroid in meter.
    phi_min: float
        Min latitude of tesseroid in radian.
    phi_max: float
        Max latitude of tesseroid in radian.
    lambda_min: float
        Min longitude of tesseroid in radian.
    lambda_max: float
        Max longitude of tesseroid in radian.
    shape: tuple
        Number of rows and columns of computation points.
    roots: numpy.ndarray, float
        Roots of legendre polynomial.
    weights: numpy.ndarray, float
        Weights of quadrature.
    order: int
        Differentiation of kernel function.
    ratio: float
        Distance-size ratio, which is specified by the user. 
        The larger ratio is, the smaller tesseroid is divided, 
        and the higher accuracy of calculation is.
    tag: string
        Kernel function to be calculated. 
        tag \in {'V', 'Vx', 'Vy', 'Vz', 
        'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz'}
    is_linear_density: bool
        If the tesseroid have linear varying density. 
        
    Returns
    -------
    data: numpy.ndarray, float
        The order-differentiation of kernel function.
    """
    lambda0 = (lambda_max + lambda_min) / 2
    delta_lambda = lambda_max - lambda_min
    if is_linear_density:
        data_constant = np.zeros((shape[0], int(shape[1]/2+1e-15)+1))
        data_linear = np.zeros((shape[0], int(shape[1]/2+1e-15)+1))

        for index_latitude in range(shape[0]):
            for index_longitude in range(int(shape[1]/2+1e-15)+1):
                constant_temp, linear_temp \
                    = cal_single_tesseroid_gravitational_field \
                        (r_cal, phi_cal[index_latitude], lambda_cal[index_longitude], \
                        r0, r_max, phi_min, phi_max, lambda_min, lambda_max, \
                        roots, weights, order, tag, ratio, \
                        is_linear_density)
                data_constant[index_latitude, index_longitude] = constant_temp
                data_linear[index_latitude, index_longitude] = linear_temp
        data_constant2 = double_data(data_constant, lambda0, delta_lambda, shape, tag)
        data_linear2 = double_data(data_linear, lambda0, delta_lambda, shape, tag)
        return data_constant2, data_linear2
    else:
        data = np.zeros((shape[0], int(shape[1]/2+1e-15)+1))

        for index_latitude in range(shape[0]):
            for index_longitude in range(int(shape[1]/2+1e-15)+1):
                if lambda0 < 0:
                    tmp_lambda_cal = lambda0 + index_longitude * delta_lambda
                else:
                    tmp_lambda_cal = lambda0 + index_longitude * delta_lambda - np.pi
                data[index_latitude, index_longitude] \
                    = cal_single_tesseroid_gravitational_field \
                        (r_cal, phi_cal[index_latitude], tmp_lambda_cal, \
                        r0, r_max, phi_min, phi_max, lambda_min, lambda_max, \
                        roots, weights, order, tag, ratio, \
                        is_linear_density)
        return double_data(data, lambda0, delta_lambda, shape, tag)


def cal_order2(r_cal, phi_cal, lambda_cal,
    r0, r_max, phi_min, phi_max, lambda_min, lambda_max,
    shape, roots, weights, order, ratio, 
    tag, is_linear_density):
    """
    Calculate the order-differentiation of kernel function.

    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: numpy.ndarray, float
        Latitude of computation point in radian.
    lambda_cal: numpy.ndarray, float
        Longitude of computation point in radian.
    r0: float
        Expansion point of Taylor series in meter.
    r_max: float
        Radius of tesseroid in meter.
    phi_min: float
        Min latitude of tesseroid in radian.
    phi_max: float
        Max latitude of tesseroid in radian.
    lambda_min: float
        Min longitude of tesseroid in radian.
    lambda_max: float
        Max longitude of tesseroid in radian.
    shape: tuple
        Number of rows and columns of computation points.
    roots: numpy.ndarray, float
        Roots of legendre polynomial.
    weights: numpy.ndarray, float
        Weights of quadrature.
    order: int
        Differentiation of kernel function.
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
        is_linear_density = true: tesseroid have linear varying density.
        is_linear_density = false: tesseroid have constant density.

    Returns
    -------
    data: numpy.ndarray, float
        The order-differentiation of kernel function.
    """
    if is_linear_density:
        data_constant = np.zeros(shape)
        data_linear = np.zeros(shape)

        for index_latitude in range(shape[0]):
            for index_longitude in range(shape[1]):
                constant_temp, linear_temp \
                    = cal_single_tesseroid_gravitational_field \
                        (r_cal, phi_cal[index_latitude], lambda_cal[index_longitude], \
                        r0, r_max, phi_min, phi_max, lambda_min, lambda_max, \
                        roots, weights, order, tag, ratio, \
                        is_linear_density)
                data_constant[index_latitude, index_longitude] = constant_temp
                data_linear[index_latitude, index_longitude] = linear_temp

        return data_constant, data_linear
    else:
        data = np.zeros(shape)

        for index_latitude in range(shape[0]):
            for index_longitude in range(shape[1]):
                data[index_latitude, index_longitude] \
                    = cal_single_tesseroid_gravitational_field \
                        (r_cal, phi_cal[index_latitude], lambda_cal[index_longitude], \
                        r0, r_max, phi_min, phi_max, lambda_min, lambda_max, \
                        roots, weights, order, tag, ratio, \
                        is_linear_density)

        return data


def shift_single_tesseroid(index_source, index_target, data, shape, tag):
    """
    Translate the data in the first column
        to other columns by using equivalence.

    Parameters
    ----------
    index: int
        Index of tesseroid. 
    data: numpy.ndarray, float
        Data in first column.
    shape: tuple
        Number of rows and columns of computation points.

    Returns
    -------
    data: numpy.ndarray, float
        Data of index-column obtained by equivalence.
    """
    
    if index_source==index_target:
        return data
    else:
        temp1 = data[:, shape[1]-(index_target-index_source):]
        temp2 = data[:, 0:shape[1]-(index_target-index_source)]
        
        return np.hstack((temp1, temp2))
    
        # if tag in ['Vy', 'Vxy', 'Vyz']:
        #     return np.hstack((-temp1, temp2))
        # else:
        #     return np.hstack((temp1, temp2))


def shift_single_tesseroid2(index_source, index_target, data, shape, tag):
    """
    Translate the data in the first column
        to other columns by using equivalence.

    Parameters
    ----------
    index: int
        Index of tesseroid. 
    data: numpy.ndarray, float
        Data in first column.
    shape: tuple
        Number of rows and columns of computation points.

    Returns
    -------
    data: numpy.ndarray, float
        Data of index-column obtained by equivalence.
    """
    
    if index_source==index_target:
        return data
    else:
        a = int(1e-15+shape[1]/2)
        if index_source<shape[1]/2:
            part1 = data[:, index_source:1+index_source+a]
            part2 = np.fliplr(part1[:, 1:a])
            temp1 = np.hstack((part2, part1))
            if index_target<shape[1]/2:
                temp2 = temp1[:, a-index_target:]
                temp3 = temp1[:, :a-index_target]

                data = np.hstack((temp2, temp3))
            else:
                temp2 = temp1[:, a-(shape[1]-index_target):]
                temp3 = temp1[:, :a-(shape[1]-index_target)]

                data = np.hstack((temp2, temp3))
        else:
            part1 = data[:, index_source-a:index_source+1]
            part2 = np.fliplr(part1[:, 1:shape[1]-1])
            temp1 = np.hstack((part2, part1))
            
            temp2 = temp1[:, a-(shape[1]-index_target):]
            temp3 = temp1[:, :a-(shape[1]-index_target)]

            data = np.hstack((temp2, temp3))
            
        return data

    
        # if tag in ['Vy', 'Vxy', 'Vyz']:
        #     return np.hstack((-temp1, temp2))
        # else:
        #     return np.hstack((temp1, temp2))


def shift_equivalent(r0, rt, shape,
        data, order, density, index_parts, tag):
    """
    
    !!! This function can only work on global data, and not include Vy, Vxy, Vyz.
        This issue will be solved later.

    Get the gravitational field generated by all tesseroid 
        in the same row through equivalence.

    Parameters
    ----------
    r0: numpy.ndarray, float
        Min radius(or max radius) of tesseroid in meter.
    rt: float
        Expansion point of Taylor series in meter.
    shape: tuple
        Number of rows and columns of computation points.
    data: numpy.ndarray, float
        Data in the first column.
    order: int
        Differentiation of kernel function.
    density: numpy.ndarray, float
        Density of tesseroid in kg/m^3.
    index_source: int 
        Index of the first column of data. 
    index_parts: 
        Index of all data.
    tag: string
        Gravitational field to be calculated. 
        tag \in {'V', 'Vx', 'Vy', 'Vz', 
        'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz'}

    Returns
    -------
    result: numpy.ndarray, float
        The order-differentiation of kernel function.
    """
    G = 6.67384e-11
    index_source = index_parts[0]
    result = np.zeros(shape)
    if order==0:
        for index in index_parts:
            result += G * density[index] \
                * shift_single_tesseroid(index_source, index, data, shape, tag)
    else:
        for index in index_parts:
            result += G * (r0[index] - rt)**order \
                / np.math.factorial(order) * density[index] \
                * shift_single_tesseroid(index_source, index, data, shape, tag)

    return result


def cal_taylor_term(r_cal, phi_cal, lambda_cal, \
        r_min, r_max, r0, phi_min, phi_max, lambda_min, lambda_max, \
        density, density_gradient, 
        shape, roots, weights, order, ratio, tag, index_parts,
        is_linear_density):
    """
    Calculate the order-differentiation of kernel function.

    Parameters
    ----------
    r_cal: float
        Radius of computation point in meter.
    phi_cal: numpy.ndarray, float
        Latitude of computation point in radian.
    lambda_cal: numpy.ndarray, float
        Longitude of computation point in radian.
    r_min: numpy.ndarray, float
        Min radius of tesseroid in meter.
    r_max: numpy.ndarray, float
        Max radius of tesseroid in meter.
    r0: float
        Expansion point of Taylor series in meter.
    phi_min: float
        Min latitude of tesseroid in radian.
    phi_max: float
        Max latitude of tesseroid in radian.
    lambda_min: float
        Min longitude of tesseroid in radian.
    lambda_max: float
        Max longitude of tesseroid in radian.
    density: numpy.ndarray, float
        Density of tesseroid in kg/m^3.
    density_gradient: numpy.ndarray, float
        Density gradient of tesseroid in kg/m^3.
    shape: tuple
        Number of rows and columns of computation points.
    roots: numpy.ndarray, float
        Roots of legendre polynomial.
    weights: numpy.ndarray, float
        Weights of quadrature.
    order: int
        Differentiation of kernel function.
    ratio: float
        Distance-size ratio, which is specified by the user. 
        The larger ratio is, the smaller tesseroid is divided, 
        and the higher accuracy of calculation is.
    tag: string
        Kernel function to be calculated. 
        tag \in {'V', 'Vx', 'Vy', 'Vz', 
        'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz'}
    index_parts: 
        Index of all data.
    is_linear_density: bool
        If the tesseroid have linear varying density. 
        is_linear_density = true: tesseroid have linear varying density.
        is_linear_density = false: tesseroid have constant density.

    Returns
    -------
    numpy.ndarray, float
        The order-differentiation of kernel function.
    """

    index_source = index_parts[0]
    rt = np.mean(r0[index_parts])

    lambda_min_temp = lambda_min + (lambda_max - lambda_min) * index_parts[0]
    lambda_max_temp = lambda_min_temp + (lambda_max - lambda_min)
    if is_linear_density:
        constant_part, linear_part = cal_order(r_cal, phi_cal, lambda_cal, \
            rt, r_max[0], phi_min, phi_max, lambda_min_temp, lambda_max_temp, \
            shape, roots, weights, order, ratio, \
            tag, is_linear_density)
        constant_part_all = shift_equivalent(r0, rt, \
            shape, constant_part, order, density, index_parts, tag)
        linear_part_all = shift_equivalent(r0, rt, \
            shape, linear_part, order, density_gradient, index_parts, tag)
        
        return constant_part_all + linear_part_all
    else:
        constant_part = cal_order(r_cal, phi_cal, lambda_cal, \
            rt, r_max[0], phi_min, phi_max, lambda_min_temp, lambda_max_temp, \
            shape, roots, weights, order, ratio, \
            tag, is_linear_density)
        
        constant_part_all = shift_equivalent(r0, rt, \
            shape, constant_part, order, density, index_parts, tag)
        
        return constant_part_all


def divide(r, parts_num):
    """
    Divide r into several equally spaced parts.
    
    Parameters
    ----------
    r: numpy.ndarray, float
        The radius need divide.
    parts_num: int
        Number of parts.

    Returns
    -------
    numpy.ndarray, float
        Index after division.
    """
    if parts_num==1:
        return np.zeros((len(r)))
    else:
        bins = np.zeros(parts_num-1)
        for index_part in range(parts_num-1):
            delta_r = (np.max(r) - np.min(r)) / parts_num
            bins[index_part] = np.min(r) + delta_r * (index_part + 1)
        index = np.digitize(r, bins)
        return index


def cal_single_part(r_cal, phi_cal, lambda_cal, \
    r_min, r_max, r0, phi_min, phi_max, lambda_min, lambda_max, \
    density, density_gradient, shape, roots, weights, \
    max_order, ratio, tag, index_parts, \
    is_linear_density):
    """
    Calculate part of gravitational field. 

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
    r0: numpy.ndarray, float
        Expansion point of Taylor series in meter.
    phi_min: float
        Min latitude of tesseroid in radian.
    phi_max: float
        Max latitude of tesseroid in radian.
    lambda_min: float
        Min longitude of tesseroid in radian.
    lambda_max: float
        Max longitude of tesseroid in radian.
    density: numpy.ndarray, float
        Density of tesseroid in kg/m^3.
    density_gradient: numpy.ndarray, float
        Density gradient of tesseroid in kg/m^3.
    shape: tuple
        Number of rows and columns of computation points.
    roots: numpy.ndarray, float
        Roots of legendre polynomial.
    weights: numpy.ndarray, float
        Weights of quadrature.
    max_order: int
        Truncation order of Taylor series.
    ratio: float
        Distance-size ratio, which is specified by the user. 
        The larger ratio is, the smaller tesseroid is divided, 
        and the higher accuracy of calculation is.
    tag: string
        Kernel function to be calculated. 
        tag \in {'V', 'Vx', 'Vy', 'Vz', 
        'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz'}
    index_parts: 
        Index of data.
    is_linear_density: bool
        If the tesseroid have linear varying density. 
        is_linear_density = true: tesseroid have linear varying density.
        is_linear_density = false: tesseroid have constant density.
    
    Returns
    -------
    result: numpy.ndarray, float
        Part of gravitational field.
    """
    result = np.zeros(shape)
    for order in range(0, max_order+1):
        result += cal_taylor_term(r_cal, phi_cal, lambda_cal, \
            r_min, r_max, r0, phi_min, phi_max, \
            lambda_min, lambda_max, density, density_gradient, \
            shape, roots, weights, order, ratio, tag, index_parts, \
            is_linear_density)
    
    return result


def cal_gravitational_field_kernel(r_cal, phi_cal, lambda_cal, \
        r_min, r_max, r0, phi_min, phi_max, 
        lambda_min, lambda_max, \
        density, density_gradient, 
        shape, roots, weights, \
        max_order, ratio, tag, parts_num, r_std_lim, \
        is_linear_density):
    """
    Calculate part of gravitational field. 

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
    r0: numpy.ndarray, float
        Expansion point of Taylor series in meter.
    phi_min: float
        Min latitude of tesseroid in radian.
    phi_max: float
        Max latitude of tesseroid in radian.
    lambda_min: float
        Min longitude of tesseroid in radian.
    lambda_max: float
        Max longitude of tesseroid in radian.
    density: numpy.ndarray, float
        Density of tesseroid in kg/m^3.
    density_gradient: numpy.ndarray, float
        Density gradient of tesseroid in kg/m^3.
    shape: tuple
        Number of rows and columns of computation points.
    roots: numpy.ndarray, float
        Roots of legendre polynomial.
    weights: numpy.ndarray, float
        Weights of quadrature.
    max_order: int
        Truncation order of Taylor series.
    max_node: int
        Max node in $r$ direction, $\phi$ direction, 
        and $\lambda$ direction.
    ratio: float
        Distance-size ratio, which is specified by the user. 
        The larger ratio is, the smaller tesseroid is divided, 
        and the higher accuracy of calculation is.
    tag: string
        Kernel function to be calculated. 
        tag \in {'V', 'Vx', 'Vy', 'Vz', 
        'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz'}
    parts_num: int
        Evenly divide r0 into parts_num parts.
    r_std_lim: float
        Limition of std(radius). 
        If std(r0) > r_std_lim, the latitude band will be divide into parts_num parts.
        If parts_num = 1, r_std_lim can be set to any value.
    is_linear_density: bool
        If the tesseroid have linear varying density. 
        is_linear_density = true: tesseroid have linear varying density.
        is_linear_density = false: tesseroid have constant density.
    
    Returns
    -------
    result: numpy.ndarray, float
        Part of gravitational field.
    """
    result = np.zeros(shape)
    std_r0 = np.std(r0)

    if std_r0>r_std_lim:
        index_parts = divide(r0, parts_num)

        temp_index = np.arange(shape[1])
        for index in range(parts_num):
            result += cal_single_part(r_cal, phi_cal, lambda_cal, \
                r_min, r_max, r0, phi_min, phi_max, 
                lambda_min, lambda_max, \
                density, density_gradient, 
                shape, roots, weights, \
                max_order, ratio, tag, temp_index[index_parts==index], \
                is_linear_density)
    else:
        index_parts = np.arange(shape[1])
        result += cal_single_part(r_cal, phi_cal, lambda_cal, \
            r_min, r_max, r0, phi_min, phi_max, 
            lambda_min, lambda_max, \
            density, density_gradient, 
            shape, roots, weights, \
            max_order, ratio, tag, index_parts, \
            is_linear_density)
    return result


def cal_latitude_band_gravitational_field(
        r_cal, phi_cal, lambda_cal, 
        r_min, r_max, phi_min, phi_max, lambda_min, lambda_max, 
        density, density_gradient, shape, roots, weights, 
        max_order_r_max, max_order_r_min, ratio, tag, 
        parts_num_r_max, parts_num_r_min, r_max_std_lim, r_min_std_lim,
        is_linear_density):
    """
    Calculate the gravitational field 
        generated by tesseroids at the same latitude.

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
        Min latitude of tesseroid in radian.
    phi_max: float
        Max latitude of tesseroid in radian.
    lambda_min: float
        Min longitude of tesseroid in radian.
    lambda_max: float
        Max longitude of tesseroid in radian.
    density: numpy.ndarray, float
        Density of tesseroid in kg/m^3.
    density_gradient: numpy.ndarray, float
        Density gradient of tesseroid in kg/m^3/m.
    shape: tuple
        Number of rows and columns of computation points.
    roots: numpy.ndarray, float
        Roots of legendre polynomial.
    weights: numpy.ndarray, float
        Weights of quadrature.
    max_order_r_max: int
        Truncation order of Taylor series of F(r_2, \varphi', \lambda')
    max_order_r_min: int
        Truncation order of Taylor series of F(r_2, \varphi', \lambda').
    max_node: int
        Max node in $r$ direction, $\phi$ direction, 
        and $\lambda$ direction. In this program, 
        the nodes in radial, latitude, and longitude direction 
        take the same value.
    ratio: float
        Distance-size ratio, which is specified by the user. 
        The larger ratio is, the smaller tesseroid is divided, 
        and the higher accuracy of calculation is.
    tag: string
        Kernel function to be calculated. 
        tag \in {'V', 'Vx', 'Vy', 'Vz', 
        'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz'}
    parts_num_r_max: int
        Evenly divide r_max into parts_num_r_max parts.
    parts_num_r_min: int
        Evenly divide r_min into parts_num_r_min parts.
    r_max_std_lim: float
        Limition of std(r_max). 
    r_min_std_lim: float
        Limition of std(r_max). 
    is_linear_density: bool
        If the tesseroid have linear varying density. 
        
    Returns
    -------
    result: numpy.ndarray, float
        The gravitational field 
            generated by tesseroids at the same latitude.
    """
    r0 = np.copy(r_max)
    result_r_max = cal_gravitational_field_kernel(
        r_cal, phi_cal, lambda_cal, 
        r_min, r_max, r0, phi_min, phi_max, 
        lambda_min, lambda_max, \
        density, density_gradient, 
        shape, roots, weights, \
        max_order_r_max, ratio, tag, parts_num_r_max, r_max_std_lim, \
        is_linear_density)
    
    r0 = np.copy(r_min)
    result_r_min = cal_gravitational_field_kernel(
        r_cal, phi_cal, lambda_cal, \
        r_min, r_max, r0, phi_min, phi_max, 
        lambda_min, lambda_max, \
        density, density_gradient, 
        shape, roots, weights, \
        max_order_r_min, ratio, tag, parts_num_r_min, r_min_std_lim, \
        is_linear_density)
    result = result_r_max - result_r_min
    return result


def is_input_format_correct(r_min, r_max, density, density_gradient, 
        phi_count, lambda_count):
    """
    Check whether the input data format is correct.

    Parameters
    ----------
    r_min: numpy.ndarray, float
        Radius of tesseroid in meter.
    r_max: numpy.ndarray, float
        Radius of tesseroid in meter.
    density: numpy.ndarray, float
        Density of tesseroid in kg/m^3.
    density_gradient: numpy.ndarray, float
        Density gradient of tesseroid in kg/m^3/m.
    phi_count: int
        Number of tesseroids in latitude direction.
    lambda_count: int
        Number of tesseroids in longitude direction.

    Returns
    -------
    bool
        The format of input is correct or not.
    """
    if density.shape!=density_gradient.shape:
        print('The shapes of density and density_gradient are not equal.')
        print('Shape of density: ', density.shape)
        print('Shape of density_gradient: ', density_gradient.shape)

        return False
    elif r_min.shape!=r_max.shape:
        print('The shapes of r_min and r_max are not equal.')
        print('Shape of r_min: ', r_min.shape)
        print('Shape of r_max: ', r_max.shape)

        return False
    elif r_min.shape[0]!=phi_count:
        print('The number of rows in r_min and phi_count is not equal.')
        print('Number of rows in r_min: ', r_min.shape[0])
        print('Number of rows in phi_count: ', phi_count)

        return False
    elif r_min.shape[1]!=lambda_count:
        print('The number of columns in r_min and lambda_count is not equal.')
        print('Number of columns in r_min: ', r_min.shape[1])
        print('Number of columns in lambda_count: ', lambda_count)

        return False
    else:
        return True


def cal_gravitational_field(r_cal, phi_cal, lambda_cal, \
        r_min, r_max, phi_min, phi_max, lambda_min, lambda_max, 
        density, density_gradient, max_order_r_max, max_order_r_min, 
        max_node, ratio, tag, parts_num_r_max, parts_num_r_min,
        r_max_std_lim, r_min_std_lim,
        is_linear_density=False, is_parallel_computing=True):
    """
    Calculate gravitational field of large-scale tesseroid.

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
        Min latitude of tesseroid in radian.
    phi_max: float
        Max latitude of tesseroid in radian.
    phi_count: int
        Number of tesseroids in latitude direction.
    lambda_min: float
        Min longitude of tesseroid in radian.
    lambda_max: float
        Max longitude of tesseroid in radian.
    lambda_count: int
        Number of tesseroids in longitude direction.
    density: numpy.ndarray, float
        Density of tesseroid in kg/m^3.
    density_gradient: numpy.ndarray, float
        Density gradient of tesseroid in kg/m^3/m.
    max_order_r_max: int
        Truncation order of Taylor series of F(r_2, \varphi', \lambda').
    max_order_r_min: int
        Truncation order of Taylor series of F(r_1, \varphi', \lambda').
    max_node: int
        Max node in $r$ direction, $\phi$ direction, 
        and $\lambda$ direction. In this program, 
        the nodes in radial, latitude, and longitude direction 
        take the same value.
    ratio: float
        Distance-size ratio, which is specified by the user. 
        The larger ratio is, the smaller tesseroid is divided, 
        and the higher accuracy of calculation is.
    tag: string
        Kernel function to be calculated. 
        tag \in {'V', 'Vx', 'Vy', 'Vz', 
        'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz'}
    parts_num_r_max: int
        Evenly divide r_max into parts_num_r_max parts.
    parts_num_r_min: int
        Evenly divide r_min into parts_num_r_min parts.
    r_max_std_lim: float
        Limition of std(r_max). 
    r_min_std_lim: float
        Limition of std(r_max). 
    is_linear_density: bool
        If the tesseroid have linear varying density.
        
    Returns
    -------
    result: numpy.ndarray, float
        The gravitational field generated by large-scale tesseroids.
    """
    phi_count = len(phi_cal)
    lambda_count = len(lambda_cal)

    if is_input_format_correct(r_min, r_max, \
        density, density_gradient, phi_count, lambda_count):
        
        apparent_density = density - density_gradient * r_min

        delta_phi = (phi_max - phi_min) / phi_count
        delta_lambda = (lambda_max - lambda_min) / lambda_count
        lambda_max_temp = lambda_min + delta_lambda

        result = np.zeros(r_min.shape)
        roots, weights = np.polynomial.legendre.leggauss(max_node)
        
        if is_parallel_computing:
            pool = mp.Pool(mp.cpu_count())
            
            results = pool.starmap(cal_latitude_band_gravitational_field, \
                [(r_cal, phi_cal, lambda_cal, \
                r_min[index_latitude, :], r_max[index_latitude, :], \
                index_latitude * delta_phi + phi_min, 
                index_latitude * delta_phi + phi_min + delta_phi, \
                lambda_min, lambda_max_temp, apparent_density[index_latitude, :], \
                density_gradient[index_latitude, :], density.shape, \
                roots, weights, max_order_r_max, max_order_r_min, ratio, tag, \
                parts_num_r_max, parts_num_r_min, r_max_std_lim, r_min_std_lim,
                is_linear_density) for index_latitude in range(phi_count)])
            pool.close()
            
            for index in results:
                result = result + index
        else:
            for index_latitude in range(phi_count):
                result += cal_latitude_band_gravitational_field \
                    (r_cal, phi_cal, lambda_cal, \
                    r_min[index_latitude, :], r_max[index_latitude, :], 
                    index_latitude * delta_phi + phi_min, 
                    index_latitude * delta_phi + phi_min + delta_phi, \
                    lambda_min, lambda_max_temp, apparent_density[index_latitude, :], \
                    density_gradient[index_latitude, :], density.shape, \
                    roots, weights, max_order_r_max, max_order_r_min, ratio, tag, \
                    parts_num_r_max, parts_num_r_min, r_max_std_lim, r_min_std_lim,
                    is_linear_density) 
        return result
    else:
        exit

