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
    h = n * np.sin(lambda_tess - lambda_cal)
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
                kernel = g * n * ( 1/2 * a * ( ( a + b ) )**( -2 ) * ( ( a )**( 3 ) + ( -4 * \
                    ( a )**( 2 ) * b + ( -18 * a * ( b )**( 2 ) + -12 * ( b )**( 3 ) ) ) \
                    ) + 6 * ( b )**( 2 ) * np.log( ( a + b ) ) )
            else:
                kernel = 1/2 * g * n * ( ell * ( ( -1 + ( m )**( 2 ) ) )**( -1 ) * ( ( a )**( 3 ) * \
                    ( -1 + ( m )**( 2 ) ) + ( 5 * ( a )**( 2 ) * b * m * ( -1 + ( m )**( \
                    2 ) ) + ( ( b )**( 3 ) * m * ( -13 + 15 * ( m )**( 2 ) ) + a * ( b \
                    )**( 2 ) * ( -3 + ( 31 * ( m )**( 2 ) + -30 * ( m )**( 4 ) ) ) ) ) ) \
                    + -3 * ( b )**( 2 ) * ( -1 + 5 * ( m )**( 2 ) ) * np.log( ( -1 * a \
                    + ( b * m + ell ) ) ) )
        elif order == 1:
            kernel = ( a )**( 4 ) * g * ( ell )**( -3 ) * n
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
                kernel = g * n * ( ( ( ( a )**( 2 ) + ( ( b )**( 2 ) + -2 * a * b * m ) ) \
                    )**( -1/2 ) * ( ( -1 + ( m )**( 2 ) ) )**( -1 ) * ( a * b * m * ( 5 + \
                    -6 * ( m )**( 2 ) ) + ( ( a )**( 2 ) * ( -1 + ( m )**( 2 ) ) + ( b \
                    )**( 2 ) * ( -2 + 3 * ( m )**( 2 ) ) ) ) + -3 * b * m * np.log( ( \
                    -1 * a + ( b * m + ( ( ( a )**( 2 ) + ( ( b )**( 2 ) + -2 * a * b * m \
                    ) ) )**( 1/2 ) ) ) ) )
        elif order==1:
            kernel = ( a )**( 3 ) * g * ( ( ( a )**( 2 ) + ( ( b )**( 2 ) + -2 * a * b * m \
                ) ) )**( -3/2 ) * n
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
    temp6 = np.cos(c)
    temp7 = np.cos(d)*np.cos(e - f)
    temp1 = temp6 * temp7
    temp2 = np.sin(c)*np.sin(d)
    temp3 = temp1 + temp2
    temp4 = a - b*(temp3)
    a2 = a*a
    a3 = a2 * a
    a4 = a2 * a2
    ell = a2 - 2*a*b*(temp3) + b*b
    ell23 = (ell)**(3/2)
    ell25 = ell * ell23
    temp8 = np.sin(e - f)
    temp62 = temp6*temp6
    temp42 = temp4*temp4
    temp44 = temp42*temp42

    if is_linear_density:
        if order==1:
            kernel = a4*temp8*temp62/ell23
        elif order==2:
            kernel = a4*(-3*a + 3*b*temp3)*temp8*temp62/ell25 + 4*a3*temp8*temp62/ell23
        elif order==3:
            kernel = 3*a2*(a2*(5*temp42/(ell) - 1)/(ell) - 8*a*temp4/(ell) + 4)*temp8*temp62/ell23
        elif order==4:
            kernel = 3*a*(-5*a3*temp4*(7*temp42/(ell) - 3)/(ell)**2 + 12*a2*(5*temp42/(ell) - 1)/(ell) - 36*a*temp4/(ell) + 8)*temp8*temp62/ell23
        elif order==5:
            kernel = 3*(15*a4*(21*temp44/(ell)**2 - 14*temp42/(ell) + 1)/(ell)**2 - 80*a3*temp4*(7*temp42/(ell) - 3)/(ell)**2 + 72*a2*(5*temp42/(ell) - 1)/(ell) - 96*a*temp4/(ell) + 8)*temp8*temp62/ell23
        elif order==6:
            kernel = 45*(-7*a4*temp4*(33*temp44/(ell)**2 - 30*temp42/(ell) + 5)/(ell)**2 + 20*a3*(21*temp44/(ell)**2 - 14*temp42/(ell) + 1)/(ell) - 40*a2*temp4*(7*temp42/(ell) - 3)/(ell) + 16*a*(5*temp42/(ell) - 1) - 8*a + 8*b*temp3)*temp8*temp62/ell25
        elif order==7:
            kernel = 45*(7*a4*(429*temp4**6/(ell)**3 - 495*temp44/(ell)**2 + 135*temp42/(ell) - 5)/(ell)**2 - 168*a3*temp4*(33*temp44/(ell)**2 - 30*temp42/(ell) + 5)/(ell)**2 + 180*a2*(21*temp44/(ell)**2 - 14*temp42/(ell) + 1)/(ell) - 160*a*temp4*(7*temp42/(ell) - 3)/(ell) + 120*temp42/(ell) - 24)*temp8*temp62/ell25
        elif order==8:
            kernel = 315*(-9*a4*temp4*(715*temp4**6/(ell)**3 - 1001*temp44/(ell)**2 + 385*temp42/(ell) - 35)/(ell)**2 + 28*a3*(429*temp4**6/(ell)**3 - 495*temp44/(ell)**2 + 135*temp42/(ell) - 5)/(ell) - 252*a2*temp4*(33*temp44/(ell)**2 - 30*temp42/(ell) + 5)/(ell) + 120*a*(21*temp44/(ell)**2 - 14*temp42/(ell) + 1) - 40*temp4*(7*temp42/(ell) - 3))*temp8*temp62/(ell)**(7/2)
        elif order==9:
            kernel = 945*(15*a4*(2431*temp4**8/(ell)**4 - 4004*temp4**6/(ell)**3 + 2002*temp44/(ell)**2 - 308*temp42/(ell) + 7)/(ell)**2 - 96*a3*temp4*(715*temp4**6/(ell)**3 - 1001*temp44/(ell)**2 + 385*temp42/(ell) - 35)/(ell)**2 + 112*a2*(429*temp4**6/(ell)**3 - 495*temp44/(ell)**2 + 135*temp42/(ell) - 5)/(ell) - 448*a*temp4*(33*temp44/(ell)**2 - 30*temp42/(ell) + 5)/(ell) + 1680*temp44/(ell)**2 - 1120*temp42/(ell) + 80)*temp8*temp62/(ell)**(7/2)
        elif order==10:
            kernel = 2835*(-55*a4*temp4*(4199*temp4**8/(ell)**4 - 7956*temp4**6/(ell)**3 + 4914*temp44/(ell)**2 - 1092*temp42/(ell) + 63)/(ell)**2 + 180*a3*(2431*temp4**8/(ell)**4 - 4004*temp4**6/(ell)**3 + 2002*temp44/(ell)**2 - 308*temp42/(ell) + 7)/(ell) - 432*a2*temp4*(715*temp4**6/(ell)**3 - 1001*temp44/(ell)**2 + 385*temp42/(ell) - 35)/(ell) + 224*a*(429*temp4**6/(ell)**3 - 495*temp44/(ell)**2 + 135*temp42/(ell) - 5) - 336*temp4*(33*temp44/(ell)**2 - 30*temp42/(ell) + 5))*temp8*temp62/(ell)**(9/2)
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**3 * sy.cos(c) \
                * (a*sy.cos(c)*sy.sin(e-f)) \
                / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**3
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    else:
        if order==1:    
            kernel = a4*temp8*temp62/ell23
        elif order==2:
            kernel = a4*(-3*a + 3*b*temp3)*temp8*temp62/ell25 + 4*a3*temp8*temp62/ell23
        elif order==3:
            kernel = 3*a2*(a2*(5*temp42/(ell) - 1)/(ell) - 8*a*temp4/(ell) + 4)*temp8*temp62/ell23
        elif order==4:
            kernel = 3*a*(-5*a3*temp4*(7*temp42/(ell) - 3)/(ell)**2 + 12*a2*(5*temp42/(ell) - 1)/(ell) - 36*a*temp4/(ell) + 8)*temp8*temp62/ell23
        elif order==5:
            kernel = 3*(15*a4*(21*temp44/(ell)**2 - 14*temp42/(ell) + 1)/(ell)**2 - 80*a3*temp4*(7*temp42/(ell) - 3)/(ell)**2 + 72*a2*(5*temp42/(ell) - 1)/(ell) - 96*a*temp4/(ell) + 8)*temp8*temp62/ell23
        elif order==6:
            kernel = 45*(-7*a4*temp4*(33*temp44/(ell)**2 - 30*temp42/(ell) + 5)/(ell)**2 + 20*a3*(21*temp44/(ell)**2 - 14*temp42/(ell) + 1)/(ell) - 40*a2*temp4*(7*temp42/(ell) - 3)/(ell) + 16*a*(5*temp42/(ell) - 1) - 8*a + 8*b*temp3)*temp8*temp62/ell25
        elif order==7:
            kernel = 45*(7*a4*(429*temp4**6/(ell)**3 - 495*temp44/(ell)**2 + 135*temp42/(ell) - 5)/(ell)**2 - 168*a3*temp4*(33*temp44/(ell)**2 - 30*temp42/(ell) + 5)/(ell)**2 + 180*a2*(21*temp44/(ell)**2 - 14*temp42/(ell) + 1)/(ell) - 160*a*temp4*(7*temp42/(ell) - 3)/(ell) + 120*temp42/(ell) - 24)*temp8*temp62/ell25
        elif order==8:
            kernel = 315*(-9*a4*temp4*(715*temp4**6/(ell)**3 - 1001*temp44/(ell)**2 + 385*temp42/(ell) - 35)/(ell)**2 + 28*a3*(429*temp4**6/(ell)**3 - 495*temp44/(ell)**2 + 135*temp42/(ell) - 5)/(ell) - 252*a2*temp4*(33*temp44/(ell)**2 - 30*temp42/(ell) + 5)/(ell) + 120*a*(21*temp44/(ell)**2 - 14*temp42/(ell) + 1) - 40*temp4*(7*temp42/(ell) - 3))*temp8*temp62/(ell)**(7/2)
        elif order==9:
            kernel = 945*(15*a4*(2431*temp4**8/(ell)**4 - 4004*temp4**6/(ell)**3 + 2002*temp44/(ell)**2 - 308*temp42/(ell) + 7)/(ell)**2 - 96*a3*temp4*(715*temp4**6/(ell)**3 - 1001*temp44/(ell)**2 + 385*temp42/(ell) - 35)/(ell)**2 + 112*a2*(429*temp4**6/(ell)**3 - 495*temp44/(ell)**2 + 135*temp42/(ell) - 5)/(ell) - 448*a*temp4*(33*temp44/(ell)**2 - 30*temp42/(ell) + 5)/(ell) + 1680*temp44/(ell)**2 - 1120*temp42/(ell) + 80)*temp8*temp62/(ell)**(7/2)
        elif order==10:
            kernel = 2835*(-55*a4*temp4*(4199*temp4**8/(ell)**4 - 7956*temp4**6/(ell)**3 + 4914*temp44/(ell)**2 - 1092*temp42/(ell) + 63)/(ell)**2 + 180*a3*(2431*temp4**8/(ell)**4 - 4004*temp4**6/(ell)**3 + 2002*temp44/(ell)**2 - 308*temp42/(ell) + 7)/(ell) - 432*a2*temp4*(715*temp4**6/(ell)**3 - 1001*temp44/(ell)**2 + 385*temp42/(ell) - 35)/(ell) + 224*a*(429*temp4**6/(ell)**3 - 495*temp44/(ell)**2 + 135*temp42/(ell) - 5) - 336*temp4*(33*temp44/(ell)**2 - 30*temp42/(ell) + 5))*temp8*temp62/(ell)**(9/2)
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**2 * sy.cos(c) \
                * (a*sy.cos(c)*sy.sin(e-f)) \
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
    temp6 = np.cos(c)
    temp7 = np.cos(d)*np.cos(e - f)
    temp1 = temp6 * temp7
    temp2 = np.sin(c)*np.sin(d)
    temp3 = temp1 + temp2
    temp4 = a - b*(temp3)
    temp42 = temp4 * temp4
    temp43 = temp42 * temp4
    temp44 = temp42 * temp42
    temp45 = temp44 * temp4
    temp46 = temp44 * temp42
    temp47 = temp46 * temp4
    temp48 = temp46 * temp42
    temp49 = temp43 * temp46
    a2 = a*a
    a3 = a2 * a
    ell = a2 - 2*a*b*(temp3) + b*b
    ell23 = (ell)**(3/2)
    ell25 = ell * ell23
    ell2 = ell * ell
    ell3 = ell2 * ell
    ell4 = ell3 * ell
    ell5 = ell4 * ell
    ell27 = ell25 * ell
    ell29 = ell27 * ell
    ell211 = ell29 * ell
    temp8 = (np.sin(c)*np.cos(d) - np.sin(d)*temp6*np.cos(e - f))
    temp82 = temp8 * temp8
    temp9 = temp43*temp82
    temp10 = temp42 * temp82
    temp11 = temp4*temp82
    temp12 = temp44*temp82
    temp13 = temp45*temp82

    if is_linear_density:
        if order==1:
            kernel = a3*(3*a2*temp82/ell25 - 1/ell23)*temp6
        elif order==2:
            kernel = a3*(3*a2*(-5*a + 5*b*temp3)*temp82/ell27 + 6*a*temp82/ell25 - (-3*a + 3*b*temp3)/ell25)*temp6 + 3*a2*(3*a2*temp82/ell25 - 1/ell23)*temp6
        elif order==3:
            kernel = 3*a*(6*a2*temp82/ell + a2*(35*a2*temp10/ell2 - 5*a2*temp82/ell - 20*a*temp11/ell - 5*temp42/ell + 2*temp82 + 1)/ell + 6*a*(-5*a2*temp11/ell + 2*a*temp82 + a - b*temp3)/ell - 2)*temp6/ell23
        elif order==4:
            kernel = 3*(-5*a3*(63*a2*temp9/ell2 - 21*a2*temp11/ell - 42*a*temp10/ell + 6*a*temp82 + 3*a - 3*b*temp3 - 7*temp43/ell + 6*temp11)/ell2 + 6*a2*temp82/ell + 9*a2*(35*a2*temp10/ell2 - 5*a2*temp82/ell - 20*a*temp11/ell - 5*temp42/ell + 2*temp82 + 1)/ell + 18*a*(-5*a2*temp11/ell + 2*a*temp82 + a - b*temp3)/ell - 2)*temp6/ell23
        elif order==5:
            kernel = 9*(5*a3*(231*a2*temp12/ell3 - 126*a2*temp10/ell2 + 7*a2*temp82/ell - 168*a*temp9/ell2 + 56*a*temp11/ell - 21*temp44/ell2 + 28*temp10/ell + 14*temp42/ell - 4*temp82 - 1)/ell - 40*a2*temp11/ell - 20*a2*(63*a2*temp9/ell2 - 21*a2*temp11/ell - 42*a*temp10/ell + 6*a*temp82 + 3*a - 3*b*temp3 - 7*temp43/ell + 6*temp11)/ell + 16*a*temp82 + 12*a*(35*a2*temp10/ell2 - 5*a2*temp82/ell - 20*a*temp11/ell - 5*temp42/ell + 2*temp82 + 1) + 8*a - 8*b*temp3)*temp6/ell25
        elif order==6:
            kernel = 45*(-7*a3*(429*a2*temp13/ell3 - 330*a2*temp9/ell2 + 45*a2*temp11/ell - 330*a*temp12/ell2 + 180*a*temp10/ell - 10*a*temp82 - 5*a + 5*b*temp3 - 33*temp45/ell2 + 60*temp9/ell + 30*temp43/ell - 20*temp11)/ell2 + 140*a2*temp10/ell2 - 20*a2*temp82/ell + 15*a2*(231*a2*temp12/ell3 - 126*a2*temp10/ell2 + 7*a2*temp82/ell - 168*a*temp9/ell2 + 56*a*temp11/ell - 21*temp44/ell2 + 28*temp10/ell + 14*temp42/ell - 4*temp82 - 1)/ell - 80*a*temp11/ell - 20*a*(63*a2*temp9/ell2 - 21*a2*temp11/ell - 42*a*temp10/ell + 6*a*temp82 + 3*a - 3*b*temp3 - 7*temp43/ell + 6*temp11)/ell - 20*temp42/ell + 8*temp82 + 4)*temp6/ell25
        elif order==7:
            kernel = 45*(7*a3*(6435*a2*temp13/ell4 - 6435*a2*temp12/ell3 + 1485*a2*temp10/ell2 - 45*a2*temp82/ell - 5148*a*temp13/ell3 + 3960*a*temp9/ell2 - 540*a*temp11/ell - 429*temp46/ell3 + 990*temp12/ell2 + 495*temp44/ell2 - 540*temp10/ell - 135*temp42/ell + 30*temp82 + 5)/ell - 2520*a2*temp9/ell2 + 840*a2*temp11/ell - 126*a2*(429*a2*temp13/ell3 - 330*a2*temp9/ell2 + 45*a2*temp11/ell - 330*a*temp12/ell2 + 180*a*temp10/ell - 10*a*temp82 - 5*a + 5*b*temp3 - 33*temp45/ell2 + 60*temp9/ell + 30*temp43/ell - 20*temp11)/ell + 1680*a*temp10/ell - 240*a*temp82 + 90*a*(231*a2*temp12/ell3 - 126*a2*temp10/ell2 + 7*a2*temp82/ell - 168*a*temp9/ell2 + 56*a*temp11/ell - 21*temp44/ell2 + 28*temp10/ell + 14*temp42/ell - 4*temp82 - 1) - 120*a + 120*b*temp3 + 280*temp43/ell - 240*temp11)*temp6/ell27
        elif order==8:
            kernel = 945*(-3*a3*(12155*a2*temp47*temp82/ell4 - 15015*a2*temp13/ell3 + 5005*a2*temp9/ell2 - 385*a2*temp11/ell - 10010*a*temp13/ell3 + 10010*a*temp12/ell2 - 2310*a*temp10/ell + 70*a*temp82 + 35*a - 35*b*temp3 - 715*temp47/ell3 + 2002*temp13/ell2 + 1001*temp45/ell2 - 1540*temp9/ell - 385*temp43/ell + 210*temp11)/ell2 + 2310*a2*temp12/ell3 - 1260*a2*temp10/ell2 + 70*a2*temp82/ell + 7*a2*(6435*a2*temp13/ell4 - 6435*a2*temp12/ell3 + 1485*a2*temp10/ell2 - 45*a2*temp82/ell - 5148*a*temp13/ell3 + 3960*a*temp9/ell2 - 540*a*temp11/ell - 429*temp46/ell3 + 990*temp12/ell2 + 495*temp44/ell2 - 540*temp10/ell - 135*temp42/ell + 30*temp82 + 5)/ell - 1680*a*temp9/ell2 + 560*a*temp11/ell - 42*a*(429*a2*temp13/ell3 - 330*a2*temp9/ell2 + 45*a2*temp11/ell - 330*a*temp12/ell2 + 180*a*temp10/ell - 10*a*temp82 - 5*a + 5*b*temp3 - 33*temp45/ell2 + 60*temp9/ell + 30*temp43/ell - 20*temp11)/ell - 210*temp44/ell2 + 280*temp10/ell + 140*temp42/ell - 40*temp82 - 10)*temp6/ell27
        elif order==9:
            kernel = 945*(15*a3*(46189*a2*temp48*temp82/ell5 - 68068*a2*temp13/ell4 + 30030*a2*temp12/ell3 - 4004*a2*temp10/ell2 + 77*a2*temp82/ell - 38896*a*temp47*temp82/ell4 + 48048*a*temp13/ell3 - 16016*a*temp9/ell2 + 1232*a*temp11/ell - 2431*temp48/ell4 + 8008*temp13/ell3 + 4004*temp46/ell3 - 8008*temp12/ell2 - 2002*temp44/ell2 + 1848*temp10/ell + 308*temp42/ell - 56*temp82 - 7)/ell - 48048*a2*temp13/ell3 + 36960*a2*temp9/ell2 - 5040*a2*temp11/ell - 72*a2*(12155*a2*temp47*temp82/ell4 - 15015*a2*temp13/ell3 + 5005*a2*temp9/ell2 - 385*a2*temp11/ell - 10010*a*temp13/ell3 + 10010*a*temp12/ell2 - 2310*a*temp10/ell + 70*a*temp82 + 35*a - 35*b*temp3 - 715*temp47/ell3 + 2002*temp13/ell2 + 1001*temp45/ell2 - 1540*temp9/ell - 385*temp43/ell + 210*temp11)/ell + 36960*a*temp12/ell2 - 20160*a*temp10/ell + 1120*a*temp82 + 56*a*(6435*a2*temp13/ell4 - 6435*a2*temp12/ell3 + 1485*a2*temp10/ell2 - 45*a2*temp82/ell - 5148*a*temp13/ell3 + 3960*a*temp9/ell2 - 540*a*temp11/ell - 429*temp46/ell3 + 990*temp12/ell2 + 495*temp44/ell2 - 540*temp10/ell - 135*temp42/ell + 30*temp82 + 5) + 560*a - 560*b*temp3 + 3696*temp45/ell2 - 6720*temp9/ell - 3360*temp43/ell + 2240*temp11)*temp6/ell29
        elif order==10:
            kernel = 2835*(-55*a3*(88179*a2*temp49*temp82/ell5 - 151164*a2*temp47*temp82/ell4 + 83538*a2*temp13/ell3 - 16380*a2*temp9/ell2 + 819*a2*temp11/ell - 75582*a*temp48*temp82/ell4 + 111384*a*temp13/ell3 - 49140*a*temp12/ell2 + 6552*a*temp10/ell - 126*a*temp82 - 63*a + 63*b*temp3 - 4199*temp49/ell4 + 15912*temp47*temp82/ell3 + 7956*temp47/ell3 - 19656*temp13/ell2 - 4914*temp45/ell2 + 6552*temp9/ell + 1092*temp43/ell - 504*temp11)/ell2 + 360360*a2*temp13/ell4 - 360360*a2*temp12/ell3 + 83160*a2*temp10/ell2 - 2520*a2*temp82/ell + 135*a2*(46189*a2*temp48*temp82/ell5 - 68068*a2*temp13/ell4 + 30030*a2*temp12/ell3 - 4004*a2*temp10/ell2 + 77*a2*temp82/ell - 38896*a*temp47*temp82/ell4 + 48048*a*temp13/ell3 - 16016*a*temp9/ell2 + 1232*a*temp11/ell - 2431*temp48/ell4 + 8008*temp13/ell3 + 4004*temp46/ell3 - 8008*temp12/ell2 - 2002*temp44/ell2 + 1848*temp10/ell + 308*temp42/ell - 56*temp82 - 7)/ell - 288288*a*temp13/ell3 + 221760*a*temp9/ell2 - 30240*a*temp11/ell - 216*a*(12155*a2*temp47*temp82/ell4 - 15015*a2*temp13/ell3 + 5005*a2*temp9/ell2 - 385*a2*temp11/ell - 10010*a*temp13/ell3 + 10010*a*temp12/ell2 - 2310*a*temp10/ell + 70*a*temp82 + 35*a - 35*b*temp3 - 715*temp47/ell3 + 2002*temp13/ell2 + 1001*temp45/ell2 - 1540*temp9/ell - 385*temp43/ell + 210*temp11)/ell - 24024*temp46/ell3 + 55440*temp12/ell2 + 27720*temp44/ell2 - 30240*temp10/ell - 7560*temp42/ell + 1680*temp82 + 280)*temp6/ell29
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
        if order==1:
            kernel = a2*(3*a2*temp82/ell25 - 1/ell23)*temp6
        elif order==2:
            kernel = a2*(3*a2*(-5*a + 5*b*temp3)*temp82/ell27 + 6*a*temp82/ell25 - (-3*a + 3*b*temp3)/ell25)*temp6 + 2*a*(3*a2*temp82/ell25 - 1/ell23)*temp6
        elif order==3:
            kernel = (6*a2*temp82/ell + 3*a2*(35*a2*temp10/ell2 - 5*a2*temp82/ell - 20*a*temp11/ell - 5*temp42/ell + 2*temp82 + 1)/ell + 12*a*(-5*a2*temp11/ell + 2*a*temp82 + a - b*temp3)/ell - 2)*temp6/ell23
        elif order==4:
            kernel = 3*(-30*a2*temp11/ell - 5*a2*(63*a2*temp9/ell2 - 21*a2*temp11/ell - 42*a*temp10/ell + 6*a*temp82 + 3*a - 3*b*temp3 - 7*temp43/ell + 6*temp11)/ell + 12*a*temp82 + 6*a*(35*a2*temp10/ell2 - 5*a2*temp82/ell - 20*a*temp11/ell - 5*temp42/ell + 2*temp82 + 1) + 6*a - 6*b*temp3)*temp6/ell25
        elif order==5:
            kernel = 3*(420*a2*temp10/ell2 - 60*a2*temp82/ell + 15*a2*(231*a2*temp12/ell3 - 126*a2*temp10/ell2 + 7*a2*temp82/ell - 168*a*temp9/ell2 + 56*a*temp11/ell - 21*temp44/ell2 + 28*temp10/ell + 14*temp42/ell - 4*temp82 - 1)/ell - 240*a*temp11/ell - 40*a*(63*a2*temp9/ell2 - 21*a2*temp11/ell - 42*a*temp10/ell + 6*a*temp82 + 3*a - 3*b*temp3 - 7*temp43/ell + 6*temp11)/ell - 60*temp42/ell + 24*temp82 + 12)*temp6/ell25
        elif order==6:
            kernel = 15*(-1260*a2*temp9/ell2 + 420*a2*temp11/ell - 21*a2*(429*a2*temp13/ell3 - 330*a2*temp9/ell2 + 45*a2*temp11/ell - 330*a*temp12/ell2 + 180*a*temp10/ell - 10*a*temp82 - 5*a + 5*b*temp3 - 33*temp45/ell2 + 60*temp9/ell + 30*temp43/ell - 20*temp11)/ell + 840*a*temp10/ell - 120*a*temp82 + 30*a*(231*a2*temp12/ell3 - 126*a2*temp10/ell2 + 7*a2*temp82/ell - 168*a*temp9/ell2 + 56*a*temp11/ell - 21*temp44/ell2 + 28*temp10/ell + 14*temp42/ell - 4*temp82 - 1) - 60*a + 60*b*temp3 + 140*temp43/ell - 120*temp11)*temp6/ell27
        elif order==7:
            kernel = 45*(6930*a2*temp12/ell3 - 3780*a2*temp10/ell2 + 210*a2*temp82/ell + 7*a2*(6435*a2*temp13/ell4 - 6435*a2*temp12/ell3 + 1485*a2*temp10/ell2 - 45*a2*temp82/ell - 5148*a*temp13/ell3 + 3960*a*temp9/ell2 - 540*a*temp11/ell - 429*temp46/ell3 + 990*temp12/ell2 + 495*temp44/ell2 - 540*temp10/ell - 135*temp42/ell + 30*temp82 + 5)/ell - 5040*a*temp9/ell2 + 1680*a*temp11/ell - 84*a*(429*a2*temp13/ell3 - 330*a2*temp9/ell2 + 45*a2*temp11/ell - 330*a*temp12/ell2 + 180*a*temp10/ell - 10*a*temp82 - 5*a + 5*b*temp3 - 33*temp45/ell2 + 60*temp9/ell + 30*temp43/ell - 20*temp11)/ell - 630*temp44/ell2 + 840*temp10/ell + 420*temp42/ell - 120*temp82 - 30)*temp6/ell27
        elif order==8:
            kernel = 315*(-18018*a2*temp13/ell3 + 13860*a2*temp9/ell2 - 1890*a2*temp11/ell - 9*a2*(12155*a2*temp47*temp82/ell4 - 15015*a2*temp13/ell3 + 5005*a2*temp9/ell2 - 385*a2*temp11/ell - 10010*a*temp13/ell3 + 10010*a*temp12/ell2 - 2310*a*temp10/ell + 70*a*temp82 + 35*a - 35*b*temp3 - 715*temp47/ell3 + 2002*temp13/ell2 + 1001*temp45/ell2 - 1540*temp9/ell - 385*temp43/ell + 210*temp11)/ell + 13860*a*temp12/ell2 - 7560*a*temp10/ell + 420*a*temp82 + 14*a*(6435*a2*temp13/ell4 - 6435*a2*temp12/ell3 + 1485*a2*temp10/ell2 - 45*a2*temp82/ell - 5148*a*temp13/ell3 + 3960*a*temp9/ell2 - 540*a*temp11/ell - 429*temp46/ell3 + 990*temp12/ell2 + 495*temp44/ell2 - 540*temp10/ell - 135*temp42/ell + 30*temp82 + 5) + 210*a - 210*b*temp3 + 1386*temp45/ell2 - 2520*temp9/ell - 1260*temp43/ell + 840*temp11)*temp6/ell29
        elif order==9:
            kernel = 315*(360360*a2*temp13/ell4 - 360360*a2*temp12/ell3 + 83160*a2*temp10/ell2 - 2520*a2*temp82/ell + 45*a2*(46189*a2*temp48*temp82/ell5 - 68068*a2*temp13/ell4 + 30030*a2*temp12/ell3 - 4004*a2*temp10/ell2 + 77*a2*temp82/ell - 38896*a*temp47*temp82/ell4 + 48048*a*temp13/ell3 - 16016*a*temp9/ell2 + 1232*a*temp11/ell - 2431*temp48/ell4 + 8008*temp13/ell3 + 4004*temp46/ell3 - 8008*temp12/ell2 - 2002*temp44/ell2 + 1848*temp10/ell + 308*temp42/ell - 56*temp82 - 7)/ell - 288288*a*temp13/ell3 + 221760*a*temp9/ell2 - 30240*a*temp11/ell - 144*a*(12155*a2*temp47*temp82/ell4 - 15015*a2*temp13/ell3 + 5005*a2*temp9/ell2 - 385*a2*temp11/ell - 10010*a*temp13/ell3 + 10010*a*temp12/ell2 - 2310*a*temp10/ell + 70*a*temp82 + 35*a - 35*b*temp3 - 715*temp47/ell3 + 2002*temp13/ell2 + 1001*temp45/ell2 - 1540*temp9/ell - 385*temp43/ell + 210*temp11)/ell - 24024*temp46/ell3 + 55440*temp12/ell2 + 27720*temp44/ell2 - 30240*temp10/ell - 7560*temp42/ell + 1680*temp82 + 280)*temp6/ell29
        elif order==10:
            kernel = 2835*(-875160*a2*temp47*temp82/ell4 + 1081080*a2*temp13/ell3 - 360360*a2*temp9/ell2 + 27720*a2*temp11/ell - 55*a2*(88179*a2*temp49*temp82/ell5 - 151164*a2*temp47*temp82/ell4 + 83538*a2*temp13/ell3 - 16380*a2*temp9/ell2 + 819*a2*temp11/ell - 75582*a*temp48*temp82/ell4 + 111384*a*temp13/ell3 - 49140*a*temp12/ell2 + 6552*a*temp10/ell - 126*a*temp82 - 63*a + 63*b*temp3 - 4199*temp49/ell4 + 15912*temp47*temp82/ell3 + 7956*temp47/ell3 - 19656*temp13/ell2 - 4914*temp45/ell2 + 6552*temp9/ell + 1092*temp43/ell - 504*temp11)/ell + 720720*a*temp13/ell3 - 720720*a*temp12/ell2 + 166320*a*temp10/ell - 5040*a*temp82 + 90*a*(46189*a2*temp48*temp82/ell5 - 68068*a2*temp13/ell4 + 30030*a2*temp12/ell3 - 4004*a2*temp10/ell2 + 77*a2*temp82/ell - 38896*a*temp47*temp82/ell4 + 48048*a*temp13/ell3 - 16016*a*temp9/ell2 + 1232*a*temp11/ell - 2431*temp48/ell4 + 8008*temp13/ell3 + 4004*temp46/ell3 - 8008*temp12/ell2 - 2002*temp44/ell2 + 1848*temp10/ell + 308*temp42/ell - 56*temp82 - 7) - 2520*a + 2520*b*temp3 + 51480*temp47/ell3 - 144144*temp13/ell2 - 72072*temp45/ell2 + 110880*temp9/ell + 27720*temp43/ell - 15120*temp11)*temp6/ell211
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
    temp6 = np.cos(c)
    temp7 = np.cos(d)*np.cos(e - f)
    temp1 = temp6 * temp7
    temp2 = np.sin(c)*np.sin(d)
    temp3 = temp1 + temp2
    temp4 = a - b*(temp3)
    temp42 = temp4 * temp4
    temp44 = temp42 * temp42
    temp46 = temp44 * temp42
    temp48 = temp46 * temp42
    a2 = a*a
    a3 = a2 * a
    a4 =a2 * a2
    a5 =a3 * a2
    ell = a2 - 2*a*b*(temp3) + b*b
    ell23 = (ell)**(3/2)
    ell25 = ell * ell23
    temp8 = (np.sin(c)*np.cos(d) - np.sin(d)*temp6*np.cos(e - f))
    temp9 = np.sin(e-f)
    ell2 = ell * ell
    ell3 = ell2 * ell
    ell4 = ell3 * ell
    ell23 = (ell)**(3/2)
    ell25 = ell * ell23
    ell27 = ell25 * ell
    ell29 = ell27 * ell
    ell211 = ell29 * ell
    temp62 = temp6 * temp6

    if is_linear_density:
        if order==1:
            kernel = 3*a5*temp8*temp9*temp62/ell25
        elif order==2:
            kernel = 3*a5*(-5*a + 5*b*temp3)*temp8*temp9*temp62/ell27 + 15*a4*temp8*temp9*temp62/ell25
        elif order==3:
            kernel = 15*a3*temp8*(a2*(7*temp42/ell - 1)/ell - 10*a*temp4/ell + 4)*temp9*temp62/ell25
        elif order==4:
            kernel = 45*a2*temp8*(-7*a3*temp4*(3*temp42/ell - 1)/ell2 + 5*a2*(7*temp42/ell - 1)/ell - 20*a*temp4/ell + 4)*temp9*temp62/ell25
        elif order==5:
            kernel = 45*a*temp8*(7*a4*(33*temp44/ell2 - 18*temp42/ell + 1)/ell2 - 140*a3*temp4*(3*temp42/ell - 1)/ell2 + 40*a2*(7*temp42/ell - 1)/ell - 80*a*temp4/ell + 8)*temp9*temp62/ell25
        elif order==6:
            kernel = 45*temp8*(-21*a5*temp4*(143*temp44/ell2 - 110*temp42/ell + 15)/ell3 + 175*a4*(33*temp44/ell2 - 18*temp42/ell + 1)/ell2 - 1400*a3*temp4*(3*temp42/ell - 1)/ell2 + 200*a2*(7*temp42/ell - 1)/ell - 200*a*temp4/ell + 8)*temp9*temp62/ell25
        elif order==7:
            kernel = 675*temp8*(21*a5*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell2 - 42*a4*temp4*(143*temp44/ell2 - 110*temp42/ell + 15)/ell2 + 140*a3*(33*temp44/ell2 - 18*temp42/ell + 1)/ell - 560*a2*temp4*(3*temp42/ell - 1)/ell + 40*a*(7*temp42/ell - 1) - 16*a + 16*b*temp3)*temp9*temp62/ell27
        elif order==8:
            kernel = 4725*temp8*(-33*a5*temp4*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell3 + 105*a4*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell2 - 84*a3*temp4*(143*temp44/ell2 - 110*temp42/ell + 15)/ell2 + 140*a2*(33*temp44/ell2 - 18*temp42/ell + 1)/ell - 280*a*temp4*(3*temp42/ell - 1)/ell + 56*temp42/ell - 8)*temp9*temp62/ell27
        elif order==9:
            kernel = 4725*temp8*(33*a5*(4199*temp48/ell4 - 6188*temp46/ell3 + 2730*temp44/ell2 - 364*temp42/ell + 7)/ell2 - 1320*a4*temp4*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell2 + 1680*a3*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell - 672*a2*temp4*(143*temp44/ell2 - 110*temp42/ell + 15)/ell + 560*a*(33*temp44/ell2 - 18*temp42/ell + 1) - 448*temp4*(3*temp42/ell - 1))*temp9*temp62/ell29
        elif order==10:
            kernel = 42525*temp8*(-143*a5*temp4*(2261*temp48/ell4 - 3876*temp46/ell3 + 2142*temp44/ell2 - 420*temp42/ell + 21)/ell3 + 165*a4*(4199*temp48/ell4 - 6188*temp46/ell3 + 2730*temp44/ell2 - 364*temp42/ell + 7)/ell2 - 2640*a3*temp4*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell2 + 1680*a2*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell - 336*a*temp4*(143*temp44/ell2 - 110*temp42/ell + 15)/ell + 3696*temp44/ell2 - 2016*temp42/ell + 112)*temp9*temp62/ell29
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
        if order==1:
            kernel = 3*a4*temp8*temp9*temp62/ell25
        elif order==2:
            kernel = 3*a4*(-5*a + 5*b*temp3)*temp8*temp9*temp62/ell27 + 12*a3*temp8*temp9*temp62/ell25
        elif order==3:
            kernel = 3*a2*temp8*(5*a2*(7*temp42/ell - 1)/ell - 40*a*temp4/ell + 12)*temp9*temp62/ell25
        elif order==4:
            kernel = 9*a*temp8*(-35*a3*temp4*(3*temp42/ell - 1)/ell2 + 20*a2*(7*temp42/ell - 1)/ell - 60*a*temp4/ell + 8)*temp9*temp62/ell25
        elif order==5:
            kernel = 9*temp8*(35*a4*(33*temp44/ell2 - 18*temp42/ell + 1)/ell2 - 560*a3*temp4*(3*temp42/ell - 1)/ell2 + 120*a2*(7*temp42/ell - 1)/ell - 160*a*temp4/ell + 8)*temp9*temp62/ell25
        elif order==6:
            kernel = 45*temp8*(-21*a4*temp4*(143*temp44/ell2 - 110*temp42/ell + 15)/ell2 + 140*a3*(33*temp44/ell2 - 18*temp42/ell + 1)/ell - 840*a2*temp4*(3*temp42/ell - 1)/ell + 80*a*(7*temp42/ell - 1) - 40*a + 40*b*temp3)*temp9*temp62/ell27
        elif order==7:
            kernel = 135*temp8*(105*a4*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell2 - 168*a3*temp4*(143*temp44/ell2 - 110*temp42/ell + 15)/ell2 + 420*a2*(33*temp44/ell2 - 18*temp42/ell + 1)/ell - 1120*a*temp4*(3*temp42/ell - 1)/ell + 280*temp42/ell - 40)*temp9*temp62/ell27
        elif order==8:
            kernel = 945*temp8*(-165*a4*temp4*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell2 + 420*a3*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell - 252*a2*temp4*(143*temp44/ell2 - 110*temp42/ell + 15)/ell + 280*a*(33*temp44/ell2 - 18*temp42/ell + 1) - 280*temp4*(3*temp42/ell - 1))*temp9*temp62/ell29
        elif order==9:
            kernel = 945*temp8*(165*a4*(4199*temp48/ell4 - 6188*temp46/ell3 + 2730*temp44/ell2 - 364*temp42/ell + 7)/ell2 - 5280*a3*temp4*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell2 + 5040*a2*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell - 1344*a*temp4*(143*temp44/ell2 - 110*temp42/ell + 15)/ell + 18480*temp44/ell2 - 10080*temp42/ell + 560)*temp9*temp62/ell29
        elif order==10:
            kernel = 8505*temp8*(-715*a4*temp4*(2261*temp48/ell4 - 3876*temp46/ell3 + 2142*temp44/ell2 - 420*temp42/ell + 21)/ell2 + 660*a3*(4199*temp48/ell4 - 6188*temp46/ell3 + 2730*temp44/ell2 - 364*temp42/ell + 7)/ell - 7920*a2*temp4*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell + 3360*a*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1) - 336*temp4*(143*temp44/ell2 - 110*temp42/ell + 15))*temp9*temp62/ell211
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
    # Common part of taylor series.
    temp6 = np.cos(c)
    temp7 = np.cos(d)*np.cos(e - f)
    temp1 = temp6 * temp7
    temp2 = np.sin(c)*np.sin(d)
    temp3 = temp1 + temp2
    temp4 = a - b*(temp3)
    temp42 = temp4 * temp4
    temp44 = temp42 * temp42
    temp46 = temp44 * temp42
    temp48 = temp46 * temp42
    temp5 = a*(temp3) - b
    a2 = a*a
    a3 = a2 * a
    a4 = a2 * a2
    ell = a2 - 2*a*b*(temp3) + b*b
    ell23 = (ell)**(3/2)
    ell25 = ell * ell23
    ell27 = ell25 * ell
    ell29 = ell27 * ell
    ell211 = ell29 * ell
    ell2 = ell * ell
    ell3 = ell2 * ell
    ell4 = ell3 * ell
    temp8 = (np.sin(c)*np.cos(d) - np.sin(d)*temp6*np.cos(e - f))
    
    if is_linear_density:
        if order==1:
            kernel = 3*a4*temp5*temp8*temp6/ell25
        elif order==2:
            kernel = 3*a4*(-5*a + 5*b*temp3)*temp5*temp8*temp6/ell27 + 3*a4*temp3*temp8*temp6/ell25 + 12*a3*temp5*temp8*temp6/ell25
        elif order==3:
            kernel = 3*a2*temp8*(-10*a2*temp4*temp3/ell + 5*a2*temp5*(7*temp42/ell - 1)/ell - 40*a*temp4*temp5/ell + 20*a*temp3 - 12*b)*temp6/ell25
        elif order==4:
            kernel = 9*a*temp8*(-35*a3*temp4*temp5*(3*temp42/ell - 1)/ell2 + 5*a3*(7*temp42/ell - 1)*temp3/ell - 40*a2*temp4*temp3/ell + 20*a2*temp5*(7*temp42/ell - 1)/ell - 60*a*temp4*temp5/ell + 20*a*temp3 - 8*b)*temp6/ell25
        elif order==5:
            kernel = 9*temp8*(-140*a4*temp4*(3*temp42/ell - 1)*temp3/ell2 + 35*a4*temp5*(33*temp44/ell2 - 18*temp42/ell + 1)/ell2 - 560*a3*temp4*temp5*(3*temp42/ell - 1)/ell2 + 80*a3*(7*temp42/ell - 1)*temp3/ell - 240*a2*temp4*temp3/ell + 120*a2*temp5*(7*temp42/ell - 1)/ell - 160*a*temp4*temp5/ell + 40*a*temp3 - 8*b)*temp6/ell25
        elif order==6:
            kernel = 45*temp8*(-21*a4*temp4*temp5*(143*temp44/ell2 - 110*temp42/ell + 15)/ell3 + 35*a4*temp3*(33*temp44/ell2 - 18*temp42/ell + 1)/ell2 - 560*a3*temp4*(3*temp42/ell - 1)*temp3/ell2 + 140*a3*temp5*(33*temp44/ell2 - 18*temp42/ell + 1)/ell2 - 840*a2*temp4*temp5*(3*temp42/ell - 1)/ell2 + 120*a2*(7*temp42/ell - 1)*temp3/ell - 160*a*temp4*temp3/ell + 80*a*temp5*(7*temp42/ell - 1)/ell - 40*temp4*temp5/ell + 8*temp2 + 8*temp6*temp7)*temp6/ell25
        elif order==7:
            kernel = 135*temp8*(-42*a4*temp4*temp3*(143*temp44/ell2 - 110*temp42/ell + 15)/ell2 + 105*a4*temp5*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell2 - 168*a3*temp4*temp5*(143*temp44/ell2 - 110*temp42/ell + 15)/ell2 + 280*a3*temp3*(33*temp44/ell2 - 18*temp42/ell + 1)/ell - 1680*a2*temp4*(3*temp42/ell - 1)*temp3/ell + 420*a2*temp5*(33*temp44/ell2 - 18*temp42/ell + 1)/ell - 1120*a*temp4*temp5*(3*temp42/ell - 1)/ell + 160*a*(7*temp42/ell - 1)*temp3 - 80*temp4*temp3 + 40*temp5*(7*temp42/ell - 1))*temp6/ell27
        elif order==8:
            kernel = 945*temp8*(-165*a4*temp4*temp5*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell3 + 105*a4*temp3*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell2 - 168*a3*temp4*temp3*(143*temp44/ell2 - 110*temp42/ell + 15)/ell2 + 420*a3*temp5*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell2 - 252*a2*temp4*temp5*(143*temp44/ell2 - 110*temp42/ell + 15)/ell2 + 420*a2*temp3*(33*temp44/ell2 - 18*temp42/ell + 1)/ell - 1120*a*temp4*(3*temp42/ell - 1)*temp3/ell + 280*a*temp5*(33*temp44/ell2 - 18*temp42/ell + 1)/ell - 280*temp4*temp5*(3*temp42/ell - 1)/ell + 40*(7*temp42/ell - 1)*temp3)*temp6/ell27
        elif order==9:
            kernel = 945*temp8*(-1320*a4*temp4*temp3*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell2 + 165*a4*temp5*(4199*temp48/ell4 - 6188*temp46/ell3 + 2730*temp44/ell2 - 364*temp42/ell + 7)/ell2 - 5280*a3*temp4*temp5*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell2 + 3360*a3*temp3*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell - 2016*a2*temp4*temp3*(143*temp44/ell2 - 110*temp42/ell + 15)/ell + 5040*a2*temp5*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell - 1344*a*temp4*temp5*(143*temp44/ell2 - 110*temp42/ell + 15)/ell + 2240*a*temp3*(33*temp44/ell2 - 18*temp42/ell + 1) - 2240*temp4*(3*temp42/ell - 1)*temp3 + 560*temp5*(33*temp44/ell2 - 18*temp42/ell + 1))*temp6/ell29
        elif order==10:
            kernel = 8505*temp8*(-715*a4*temp4*temp5*(2261*temp48/ell4 - 3876*temp46/ell3 + 2142*temp44/ell2 - 420*temp42/ell + 21)/ell3 + 165*a4*temp3*(4199*temp48/ell4 - 6188*temp46/ell3 + 2730*temp44/ell2 - 364*temp42/ell + 7)/ell2 - 5280*a3*temp4*temp3*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell2 + 660*a3*temp5*(4199*temp48/ell4 - 6188*temp46/ell3 + 2730*temp44/ell2 - 364*temp42/ell + 7)/ell2 - 7920*a2*temp4*temp5*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell2 + 5040*a2*temp3*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell - 1344*a*temp4*temp3*(143*temp44/ell2 - 110*temp42/ell + 15)/ell + 3360*a*temp5*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell - 336*temp4*temp5*(143*temp44/ell2 - 110*temp42/ell + 15)/ell + 560*temp3*(33*temp44/ell2 - 18*temp42/ell + 1))*temp6/ell29
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
        if order==1:
            kernel = 3*a3*temp5*temp8*temp6/ell25
        elif order==2:
            kernel = 3*a3*(-5*a + 5*b*temp3)*temp5*temp8*temp6/ell27 + 3*a3*temp3*temp8*temp6/ell25 + 9*a2*temp5*temp8*temp6/ell25
        elif order==3:
            kernel = 3*a*temp8*(-10*a2*temp4*temp3/ell + 5*a2*temp5*(7*temp42/ell - 1)/ell - 30*a*temp4*temp5/ell + 12*a*temp3 - 6*b)*temp6/ell25
        elif order==4:
            kernel = 9*temp8*(-35*a3*temp4*temp5*(3*temp42/ell - 1)/ell2 + 5*a3*(7*temp42/ell - 1)*temp3/ell - 30*a2*temp4*temp3/ell + 15*a2*temp5*(7*temp42/ell - 1)/ell - 30*a*temp4*temp5/ell + 8*a*temp3 - 2*b)*temp6/ell25
        elif order==5:
            kernel = 9*temp8*(-140*a3*temp4*(3*temp42/ell - 1)*temp3/ell2 + 35*a3*temp5*(33*temp44/ell2 - 18*temp42/ell + 1)/ell2 - 420*a2*temp4*temp5*(3*temp42/ell - 1)/ell2 + 60*a2*(7*temp42/ell - 1)*temp3/ell - 120*a*temp4*temp3/ell + 60*a*temp5*(7*temp42/ell - 1)/ell - 40*temp4*temp5/ell + 8*temp2 + 8*temp6*temp7)*temp6/ell25
        elif order==6:
            kernel = 45*temp8*(-21*a3*temp4*temp5*(143*temp44/ell2 - 110*temp42/ell + 15)/ell2 + 35*a3*temp3*(33*temp44/ell2 - 18*temp42/ell + 1)/ell - 420*a2*temp4*(3*temp42/ell - 1)*temp3/ell + 105*a2*temp5*(33*temp44/ell2 - 18*temp42/ell + 1)/ell - 420*a*temp4*temp5*(3*temp42/ell - 1)/ell + 60*a*(7*temp42/ell - 1)*temp3 - 40*temp4*temp3 + 20*temp5*(7*temp42/ell - 1))*temp6/ell27
        elif order==7:
            kernel = 135*temp8*(-42*a3*temp4*temp3*(143*temp44/ell2 - 110*temp42/ell + 15)/ell2 + 105*a3*temp5*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell2 - 126*a2*temp4*temp5*(143*temp44/ell2 - 110*temp42/ell + 15)/ell2 + 210*a2*temp3*(33*temp44/ell2 - 18*temp42/ell + 1)/ell - 840*a*temp4*(3*temp42/ell - 1)*temp3/ell + 210*a*temp5*(33*temp44/ell2 - 18*temp42/ell + 1)/ell - 280*temp4*temp5*(3*temp42/ell - 1)/ell + 40*(7*temp42/ell - 1)*temp3)*temp6/ell27
        elif order==8:
            kernel = 945*temp8*(-165*a3*temp4*temp5*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell2 + 105*a3*temp3*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell - 126*a2*temp4*temp3*(143*temp44/ell2 - 110*temp42/ell + 15)/ell + 315*a2*temp5*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell - 126*a*temp4*temp5*(143*temp44/ell2 - 110*temp42/ell + 15)/ell + 210*a*temp3*(33*temp44/ell2 - 18*temp42/ell + 1) - 280*temp4*(3*temp42/ell - 1)*temp3 + 70*temp5*(33*temp44/ell2 - 18*temp42/ell + 1))*temp6/ell29
        elif order==9:
            kernel = 945*temp8*(-1320*a3*temp4*temp3*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell2 + 165*a3*temp5*(4199*temp48/ell4 - 6188*temp46/ell3 + 2730*temp44/ell2 - 364*temp42/ell + 7)/ell2 - 3960*a2*temp4*temp5*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell2 + 2520*a2*temp3*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell - 1008*a*temp4*temp3*(143*temp44/ell2 - 110*temp42/ell + 15)/ell + 2520*a*temp5*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell - 336*temp4*temp5*(143*temp44/ell2 - 110*temp42/ell + 15)/ell + 560*temp3*(33*temp44/ell2 - 18*temp42/ell + 1))*temp6/ell29
        elif order==10:
            kernel = 8505*temp8*(-715*a3*temp4*temp5*(2261*temp48/ell4 - 3876*temp46/ell3 + 2142*temp44/ell2 - 420*temp42/ell + 21)/ell2 + 165*a3*temp3*(4199*temp48/ell4 - 6188*temp46/ell3 + 2730*temp44/ell2 - 364*temp42/ell + 7)/ell - 3960*a2*temp4*temp3*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell + 495*a2*temp5*(4199*temp48/ell4 - 6188*temp46/ell3 + 2730*temp44/ell2 - 364*temp42/ell + 7)/ell - 3960*a*temp4*temp5*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell + 2520*a*temp3*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1) - 336*temp4*temp3*(143*temp44/ell2 - 110*temp42/ell + 15) + 840*temp5*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1))*temp6/ell211
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
    temp6 = np.cos(c)
    temp7 = np.cos(d)*np.cos(e - f)
    temp1 = temp6 * temp7
    temp2 = np.sin(c)*np.sin(d)
    temp3 = temp1 + temp2
    temp4 = a - b*(temp3)
    temp42 = temp4 * temp4
    temp43 = temp42 * temp4
    temp44 = temp42 * temp42
    temp45 = temp44 * temp4
    temp46 = temp44 * temp42
    temp47 = temp46 * temp4
    temp48 = temp46 * temp42
    temp49 = temp43 * temp46
    a2 = a*a
    a3 = a2 * a
    ell = a2 - 2*a*b*(temp3) + b*b
    ell2 = ell * ell
    ell3 = ell2 * ell
    ell4 = ell3 * ell
    ell5 = ell4 * ell
    ell23 = (ell)**(3/2)
    ell25 = ell * ell23
    ell27 = ell25 * ell
    ell29 = ell27 * ell
    ell211 = ell29 * ell
    temp8 = np.sin(e-f)
    temp82 = temp8 * temp8
    temp62 = temp6 * temp6
    temp9 = temp82*temp62
    temp10 = temp4*temp9
    temp11 = temp45*temp9
    temp12 = temp42*temp9
    temp13 = temp44*temp9
    temp14 = temp43*temp9
    temp15 = temp46*temp9

    if is_linear_density:
        if order==1:
            kernel = a3*(3*a2*temp9/ell25 - 1/ell23)*temp6
        elif order==2:
            kernel = a3*(3*a2*(-5*a + 5*b*temp3)*temp9/ell27 + 6*a*temp9/ell25 - (-3*a + 3*b*temp3)/ell25)*temp6 + 3*a2*(3*a2*temp9/ell25 - 1/ell23)*temp6
        elif order==3:
            kernel = 3*a*(a2*(35*a2*temp12/ell2 - 5*a2*temp9/ell - 20*a*temp10/ell - 5*temp42/ell + 2*temp9 + 1)/ell + 6*a2*temp9/ell + 6*a*(-5*a2*temp10/ell + 2*a*temp9 + a - b*temp3)/ell - 2)*temp6/ell23
        elif order==4:
            kernel = 3*(-5*a3*(63*a2*temp14/ell2 - 21*a2*temp10/ell - 42*a*temp12/ell + 6*a*temp9 + 3*a - 3*b*temp3 - 7*temp43/ell + 6*temp10)/ell2 + 9*a2*(35*a2*temp12/ell2 - 5*a2*temp9/ell - 20*a*temp10/ell - 5*temp42/ell + 2*temp9 + 1)/ell + 6*a2*temp9/ell + 18*a*(-5*a2*temp10/ell + 2*a*temp9 + a - b*temp3)/ell - 2)*temp6/ell23
        elif order==5:
            kernel = 9*(5*a3*(231*a2*temp13/ell3 - 126*a2*temp12/ell2 + 7*a2*temp9/ell - 168*a*temp14/ell2 + 56*a*temp10/ell - 21*temp44/ell2 + 28*temp12/ell + 14*temp42/ell - 4*temp9 - 1)/ell - 40*a2*temp10/ell - 20*a2*(63*a2*temp14/ell2 - 21*a2*temp10/ell - 42*a*temp12/ell + 6*a*temp9 + 3*a - 3*b*temp3 - 7*temp43/ell + 6*temp10)/ell + 12*a*(35*a2*temp12/ell2 - 5*a2*temp9/ell - 20*a*temp10/ell - 5*temp42/ell + 2*temp9 + 1) + 16*a*temp9 + 8*a - 8*b*temp3)*temp6/ell25
        elif order==6:
            kernel = 45*(-7*a3*(429*a2*temp11/ell3 - 330*a2*temp14/ell2 + 45*a2*temp10/ell - 330*a*temp13/ell2 + 180*a*temp12/ell - 10*a*temp9 - 5*a + 5*b*temp3 - 33*temp45/ell2 + 60*temp14/ell + 30*temp43/ell - 20*temp10)/ell2 + 140*a2*temp12/ell2 + 15*a2*(231*a2*temp13/ell3 - 126*a2*temp12/ell2 + 7*a2*temp9/ell - 168*a*temp14/ell2 + 56*a*temp10/ell - 21*temp44/ell2 + 28*temp12/ell + 14*temp42/ell - 4*temp9 - 1)/ell - 20*a2*temp9/ell - 80*a*temp10/ell - 20*a*(63*a2*temp14/ell2 - 21*a2*temp10/ell - 42*a*temp12/ell + 6*a*temp9 + 3*a - 3*b*temp3 - 7*temp43/ell + 6*temp10)/ell - 20*temp42/ell + 8*temp9 + 4)*temp6/ell25
        elif order==7:
            kernel = 45*(7*a3*(6435*a2*temp15/ell4 - 6435*a2*temp13/ell3 + 1485*a2*temp12/ell2 - 45*a2*temp9/ell - 5148*a*temp11/ell3 + 3960*a*temp14/ell2 - 540*a*temp10/ell - 429*temp46/ell3 + 990*temp13/ell2 + 495*temp44/ell2 - 540*temp12/ell - 135*temp42/ell + 30*temp9 + 5)/ell - 2520*a2*temp14/ell2 + 840*a2*temp10/ell - 126*a2*(429*a2*temp11/ell3 - 330*a2*temp14/ell2 + 45*a2*temp10/ell - 330*a*temp13/ell2 + 180*a*temp12/ell - 10*a*temp9 - 5*a + 5*b*temp3 - 33*temp45/ell2 + 60*temp14/ell + 30*temp43/ell - 20*temp10)/ell + 1680*a*temp12/ell + 90*a*(231*a2*temp13/ell3 - 126*a2*temp12/ell2 + 7*a2*temp9/ell - 168*a*temp14/ell2 + 56*a*temp10/ell - 21*temp44/ell2 + 28*temp12/ell + 14*temp42/ell - 4*temp9 - 1) - 240*a*temp9 - 120*a + 120*b*temp3 + 280*temp43/ell - 240*temp10)*temp6/ell27
        elif order==8:
            kernel = 945*(-3*a3*(12155*a2*temp47*temp9/ell4 - 15015*a2*temp11/ell3 + 5005*a2*temp14/ell2 - 385*a2*temp10/ell - 10010*a*temp15/ell3 + 10010*a*temp13/ell2 - 2310*a*temp12/ell + 70*a*temp9 + 35*a - 35*b*temp3 - 715*temp47/ell3 + 2002*temp11/ell2 + 1001*temp45/ell2 - 1540*temp14/ell - 385*temp43/ell + 210*temp10)/ell2 + 2310*a2*temp13/ell3 - 1260*a2*temp12/ell2 + 7*a2*(6435*a2*temp15/ell4 - 6435*a2*temp13/ell3 + 1485*a2*temp12/ell2 - 45*a2*temp9/ell - 5148*a*temp11/ell3 + 3960*a*temp14/ell2 - 540*a*temp10/ell - 429*temp46/ell3 + 990*temp13/ell2 + 495*temp44/ell2 - 540*temp12/ell - 135*temp42/ell + 30*temp9 + 5)/ell + 70*a2*temp9/ell - 1680*a*temp14/ell2 + 560*a*temp10/ell - 42*a*(429*a2*temp11/ell3 - 330*a2*temp14/ell2 + 45*a2*temp10/ell - 330*a*temp13/ell2 + 180*a*temp12/ell - 10*a*temp9 - 5*a + 5*b*temp3 - 33*temp45/ell2 + 60*temp14/ell + 30*temp43/ell - 20*temp10)/ell - 210*temp44/ell2 + 280*temp12/ell + 140*temp42/ell - 40*temp9 - 10)*temp6/ell27
        elif order==9:
            kernel = 945*(15*a3*(46189*a2*temp48*temp9/ell5 - 68068*a2*temp15/ell4 + 30030*a2*temp13/ell3 - 4004*a2*temp12/ell2 + 77*a2*temp9/ell - 38896*a*temp47*temp9/ell4 + 48048*a*temp11/ell3 - 16016*a*temp14/ell2 + 1232*a*temp10/ell - 2431*temp48/ell4 + 8008*temp15/ell3 + 4004*temp46/ell3 - 8008*temp13/ell2 - 2002*temp44/ell2 + 1848*temp12/ell + 308*temp42/ell - 56*temp9 - 7)/ell - 48048*a2*temp11/ell3 + 36960*a2*temp14/ell2 - 5040*a2*temp10/ell - 72*a2*(12155*a2*temp47*temp9/ell4 - 15015*a2*temp11/ell3 + 5005*a2*temp14/ell2 - 385*a2*temp10/ell - 10010*a*temp15/ell3 + 10010*a*temp13/ell2 - 2310*a*temp12/ell + 70*a*temp9 + 35*a - 35*b*temp3 - 715*temp47/ell3 + 2002*temp11/ell2 + 1001*temp45/ell2 - 1540*temp14/ell - 385*temp43/ell + 210*temp10)/ell + 36960*a*temp13/ell2 - 20160*a*temp12/ell + 56*a*(6435*a2*temp15/ell4 - 6435*a2*temp13/ell3 + 1485*a2*temp12/ell2 - 45*a2*temp9/ell - 5148*a*temp11/ell3 + 3960*a*temp14/ell2 - 540*a*temp10/ell - 429*temp46/ell3 + 990*temp13/ell2 + 495*temp44/ell2 - 540*temp12/ell - 135*temp42/ell + 30*temp9 + 5) + 1120*a*temp9 + 560*a - 560*b*temp3 + 3696*temp45/ell2 - 6720*temp14/ell - 3360*temp43/ell + 2240*temp10)*temp6/ell29
        elif order==10:
            kernel = 2835*(-55*a3*(88179*a2*temp49*temp9/ell5 - 151164*a2*temp47*temp9/ell4 + 83538*a2*temp11/ell3 - 16380*a2*temp14/ell2 + 819*a2*temp10/ell - 75582*a*temp48*temp9/ell4 + 111384*a*temp15/ell3 - 49140*a*temp13/ell2 + 6552*a*temp12/ell - 126*a*temp9 - 63*a + 63*b*temp3 - 4199*temp49/ell4 + 15912*temp47*temp9/ell3 + 7956*temp47/ell3 - 19656*temp11/ell2 - 4914*temp45/ell2 + 6552*temp14/ell + 1092*temp43/ell - 504*temp10)/ell2 + 360360*a2*temp15/ell4 - 360360*a2*temp13/ell3 + 83160*a2*temp12/ell2 + 135*a2*(46189*a2*temp48*temp9/ell5 - 68068*a2*temp15/ell4 + 30030*a2*temp13/ell3 - 4004*a2*temp12/ell2 + 77*a2*temp9/ell - 38896*a*temp47*temp9/ell4 + 48048*a*temp11/ell3 - 16016*a*temp14/ell2 + 1232*a*temp10/ell - 2431*temp48/ell4 + 8008*temp15/ell3 + 4004*temp46/ell3 - 8008*temp13/ell2 - 2002*temp44/ell2 + 1848*temp12/ell + 308*temp42/ell - 56*temp9 - 7)/ell - 2520*a2*temp9/ell - 288288*a*temp11/ell3 + 221760*a*temp14/ell2 - 30240*a*temp10/ell - 216*a*(12155*a2*temp47*temp9/ell4 - 15015*a2*temp11/ell3 + 5005*a2*temp14/ell2 - 385*a2*temp10/ell - 10010*a*temp15/ell3 + 10010*a*temp13/ell2 - 2310*a*temp12/ell + 70*a*temp9 + 35*a - 35*b*temp3 - 715*temp47/ell3 + 2002*temp11/ell2 + 1001*temp45/ell2 - 1540*temp14/ell - 385*temp43/ell + 210*temp10)/ell - 24024*temp46/ell3 + 55440*temp13/ell2 + 27720*temp44/ell2 - 30240*temp12/ell - 7560*temp42/ell + 1680*temp9 + 280)*temp6/ell29
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**3 * sy.cos(c) \
                * (3 * (a*sy.cos(c)*sy.sin(e-f))**2 \
                / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**5 \
                - 1 / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**3)
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    else:
        if order==1:
            kernel = a2*(3*a2*temp9/ell25 - 1/ell23)*temp6
        elif order==2:
            kernel = a2*(3*a2*(-5*a + 5*b*temp3)*temp9/ell27 + 6*a*temp9/ell25 - (-3*a + 3*b*temp3)/ell25)*temp6 + 2*a*(3*a2*temp9/ell25 - 1/ell23)*temp6
        elif order==3:
            kernel = (3*a2*(35*a2*temp12/ell2 - 5*a2*temp9/ell - 20*a*temp10/ell - 5*temp42/ell + 2*temp9 + 1)/ell + 6*a2*temp9/ell + 12*a*(-5*a2*temp10/ell + 2*a*temp9 + a - b*temp3)/ell - 2)*temp6/ell23
        elif order==4:
            kernel = 3*(-30*a2*temp10/ell - 5*a2*(63*a2*temp14/ell2 - 21*a2*temp10/ell - 42*a*temp12/ell + 6*a*temp9 + 3*a - 3*b*temp3 - 7*temp43/ell + 6*temp10)/ell + 6*a*(35*a2*temp12/ell2 - 5*a2*temp9/ell - 20*a*temp10/ell - 5*temp42/ell + 2*temp9 + 1) + 12*a*temp9 + 6*a - 6*b*temp3)*temp6/ell25
        elif order==5:
            kernel = 3*(420*a2*temp12/ell2 + 15*a2*(231*a2*temp13/ell3 - 126*a2*temp12/ell2 + 7*a2*temp9/ell - 168*a*temp14/ell2 + 56*a*temp10/ell - 21*temp44/ell2 + 28*temp12/ell + 14*temp42/ell - 4*temp9 - 1)/ell - 60*a2*temp9/ell - 240*a*temp10/ell - 40*a*(63*a2*temp14/ell2 - 21*a2*temp10/ell - 42*a*temp12/ell + 6*a*temp9 + 3*a - 3*b*temp3 - 7*temp43/ell + 6*temp10)/ell - 60*temp42/ell + 24*temp9 + 12)*temp6/ell25
        elif order==6:
            kernel = 15*(-1260*a2*temp14/ell2 + 420*a2*temp10/ell - 21*a2*(429*a2*temp11/ell3 - 330*a2*temp14/ell2 + 45*a2*temp10/ell - 330*a*temp13/ell2 + 180*a*temp12/ell - 10*a*temp9 - 5*a + 5*b*temp3 - 33*temp45/ell2 + 60*temp14/ell + 30*temp43/ell - 20*temp10)/ell + 840*a*temp12/ell + 30*a*(231*a2*temp13/ell3 - 126*a2*temp12/ell2 + 7*a2*temp9/ell - 168*a*temp14/ell2 + 56*a*temp10/ell - 21*temp44/ell2 + 28*temp12/ell + 14*temp42/ell - 4*temp9 - 1) - 120*a*temp9 - 60*a + 60*b*temp3 + 140*temp43/ell - 120*temp10)*temp6/ell27
        elif order==7:
            kernel = 45*(6930*a2*temp13/ell3 - 3780*a2*temp12/ell2 + 7*a2*(6435*a2*temp15/ell4 - 6435*a2*temp13/ell3 + 1485*a2*temp12/ell2 - 45*a2*temp9/ell - 5148*a*temp11/ell3 + 3960*a*temp14/ell2 - 540*a*temp10/ell - 429*temp46/ell3 + 990*temp13/ell2 + 495*temp44/ell2 - 540*temp12/ell - 135*temp42/ell + 30*temp9 + 5)/ell + 210*a2*temp9/ell - 5040*a*temp14/ell2 + 1680*a*temp10/ell - 84*a*(429*a2*temp11/ell3 - 330*a2*temp14/ell2 + 45*a2*temp10/ell - 330*a*temp13/ell2 + 180*a*temp12/ell - 10*a*temp9 - 5*a + 5*b*temp3 - 33*temp45/ell2 + 60*temp14/ell + 30*temp43/ell - 20*temp10)/ell - 630*temp44/ell2 + 840*temp12/ell + 420*temp42/ell - 120*temp9 - 30)*temp6/ell27
        elif order==8:
            kernel = 315*(-18018*a2*temp11/ell3 + 13860*a2*temp14/ell2 - 1890*a2*temp10/ell - 9*a2*(12155*a2*temp47*temp9/ell4 - 15015*a2*temp11/ell3 + 5005*a2*temp14/ell2 - 385*a2*temp10/ell - 10010*a*temp15/ell3 + 10010*a*temp13/ell2 - 2310*a*temp12/ell + 70*a*temp9 + 35*a - 35*b*temp3 - 715*temp47/ell3 + 2002*temp11/ell2 + 1001*temp45/ell2 - 1540*temp14/ell - 385*temp43/ell + 210*temp10)/ell + 13860*a*temp13/ell2 - 7560*a*temp12/ell + 14*a*(6435*a2*temp15/ell4 - 6435*a2*temp13/ell3 + 1485*a2*temp12/ell2 - 45*a2*temp9/ell - 5148*a*temp11/ell3 + 3960*a*temp14/ell2 - 540*a*temp10/ell - 429*temp46/ell3 + 990*temp13/ell2 + 495*temp44/ell2 - 540*temp12/ell - 135*temp42/ell + 30*temp9 + 5) + 420*a*temp9 + 210*a - 210*b*temp3 + 1386*temp45/ell2 - 2520*temp14/ell - 1260*temp43/ell + 840*temp10)*temp6/ell29
        elif order==9:
            kernel = 315*(360360*a2*temp15/ell4 - 360360*a2*temp13/ell3 + 83160*a2*temp12/ell2 + 45*a2*(46189*a2*temp48*temp9/ell5 - 68068*a2*temp15/ell4 + 30030*a2*temp13/ell3 - 4004*a2*temp12/ell2 + 77*a2*temp9/ell - 38896*a*temp47*temp9/ell4 + 48048*a*temp11/ell3 - 16016*a*temp14/ell2 + 1232*a*temp10/ell - 2431*temp48/ell4 + 8008*temp15/ell3 + 4004*temp46/ell3 - 8008*temp13/ell2 - 2002*temp44/ell2 + 1848*temp12/ell + 308*temp42/ell - 56*temp9 - 7)/ell - 2520*a2*temp9/ell - 288288*a*temp11/ell3 + 221760*a*temp14/ell2 - 30240*a*temp10/ell - 144*a*(12155*a2*temp47*temp9/ell4 - 15015*a2*temp11/ell3 + 5005*a2*temp14/ell2 - 385*a2*temp10/ell - 10010*a*temp15/ell3 + 10010*a*temp13/ell2 - 2310*a*temp12/ell + 70*a*temp9 + 35*a - 35*b*temp3 - 715*temp47/ell3 + 2002*temp11/ell2 + 1001*temp45/ell2 - 1540*temp14/ell - 385*temp43/ell + 210*temp10)/ell - 24024*temp46/ell3 + 55440*temp13/ell2 + 27720*temp44/ell2 - 30240*temp12/ell - 7560*temp42/ell + 1680*temp9 + 280)*temp6/ell29
        elif order==10:
            kernel = 2835*(-875160*a2*temp47*temp9/ell4 + 1081080*a2*temp11/ell3 - 360360*a2*temp14/ell2 + 27720*a2*temp10/ell - 55*a2*(88179*a2*temp49*temp9/ell5 - 151164*a2*temp47*temp9/ell4 + 83538*a2*temp11/ell3 - 16380*a2*temp14/ell2 + 819*a2*temp10/ell - 75582*a*temp48*temp9/ell4 + 111384*a*temp15/ell3 - 49140*a*temp13/ell2 + 6552*a*temp12/ell - 126*a*temp9 - 63*a + 63*b*temp3 - 4199*temp49/ell4 + 15912*temp47*temp9/ell3 + 7956*temp47/ell3 - 19656*temp11/ell2 - 4914*temp45/ell2 + 6552*temp14/ell + 1092*temp43/ell - 504*temp10)/ell + 720720*a*temp15/ell3 - 720720*a*temp13/ell2 + 166320*a*temp12/ell + 90*a*(46189*a2*temp48*temp9/ell5 - 68068*a2*temp15/ell4 + 30030*a2*temp13/ell3 - 4004*a2*temp12/ell2 + 77*a2*temp9/ell - 38896*a*temp47*temp9/ell4 + 48048*a*temp11/ell3 - 16016*a*temp14/ell2 + 1232*a*temp10/ell - 2431*temp48/ell4 + 8008*temp15/ell3 + 4004*temp46/ell3 - 8008*temp13/ell2 - 2002*temp44/ell2 + 1848*temp12/ell + 308*temp42/ell - 56*temp9 - 7) - 5040*a*temp9 - 2520*a + 2520*b*temp3 + 51480*temp47/ell3 - 144144*temp11/ell2 - 72072*temp45/ell2 + 110880*temp14/ell + 27720*temp43/ell - 15120*temp10)*temp6/ell211
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**2 * sy.cos(c) \
                * (3 * (a*sy.cos(c)*sy.sin(e-f))**2 \
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
    # Common part of taylor series.
    temp6 = np.cos(c)
    temp62 = temp6 * temp6
    temp7 = np.cos(d)*np.cos(e - f)
    temp1 = temp6 * temp7
    temp2 = np.sin(c)*np.sin(d)
    temp3 = temp1 + temp2
    temp4 = a - b*(temp3)
    temp42 = temp4 * temp4
    temp44 = temp42 * temp42
    temp46 = temp44 * temp42
    temp48 = temp46 * temp42
    temp5 = a*(temp3) - b
    a2 = a*a
    a3 = a2 * a
    a4 = a2 * a2
    ell = a2 - 2*a*b*(temp3) + b*b
    ell2 = ell * ell
    ell3 = ell2 * ell
    ell4 = ell3 * ell
    ell23 = (ell)**(3/2)
    ell25 = ell * ell23
    ell27 = ell25 * ell
    ell29 = ell27 * ell
    ell211 = ell29 * ell
    temp8 = np.sin(e-f)
    temp9 = temp4 * temp5
    temp10 = temp4 * temp3
    temp12 = temp8*temp62

    if is_linear_density:
        if order==1:
            kernel = 3*a4*temp5*temp12/ell25
        elif order==2:
            kernel = 3*a4*(-5*a + 5*b*temp3)*temp5*temp12/ell27 + 3*a4*temp3*temp12/ell25 + 12*a3*temp5*temp12/ell25
        elif order==3:
            kernel = 3*a2*(-10*a2*temp10/ell + 5*a2*temp5*(7*temp42/ell - 1)/ell - 40*a*temp9/ell + 20*a*temp3 - 12*b)*temp12/ell25
        elif order==4:
            kernel = 9*a*(-35*a3*temp9*(3*temp42/ell - 1)/ell2 + 5*a3*(7*temp42/ell - 1)*temp3/ell - 40*a2*temp10/ell + 20*a2*temp5*(7*temp42/ell - 1)/ell - 60*a*temp9/ell + 20*a*temp3 - 8*b)*temp12/ell25
        elif order==5:
            kernel = 9*(-140*a4*temp4*(3*temp42/ell - 1)*temp3/ell2 + 35*a4*temp5*(33*temp44/ell2 - 18*temp42/ell + 1)/ell2 - 560*a3*temp9*(3*temp42/ell - 1)/ell2 + 80*a3*(7*temp42/ell - 1)*temp3/ell - 240*a2*temp10/ell + 120*a2*temp5*(7*temp42/ell - 1)/ell - 160*a*temp9/ell + 40*a*temp3 - 8*b)*temp12/ell25
        elif order==6:
            kernel = 45*(-21*a4*temp9*(143*temp44/ell2 - 110*temp42/ell + 15)/ell3 + 35*a4*temp3*(33*temp44/ell2 - 18*temp42/ell + 1)/ell2 - 560*a3*temp4*(3*temp42/ell - 1)*temp3/ell2 + 140*a3*temp5*(33*temp44/ell2 - 18*temp42/ell + 1)/ell2 - 840*a2*temp9*(3*temp42/ell - 1)/ell2 + 120*a2*(7*temp42/ell - 1)*temp3/ell - 160*a*temp10/ell + 80*a*temp5*(7*temp42/ell - 1)/ell - 40*temp9/ell + 8*temp2 + 8*temp6*temp7)*temp12/ell25
        elif order==7:
            kernel = 135*(-42*a4*temp10*(143*temp44/ell2 - 110*temp42/ell + 15)/ell2 + 105*a4*temp5*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell2 - 168*a3*temp9*(143*temp44/ell2 - 110*temp42/ell + 15)/ell2 + 280*a3*temp3*(33*temp44/ell2 - 18*temp42/ell + 1)/ell - 1680*a2*temp4*(3*temp42/ell - 1)*temp3/ell + 420*a2*temp5*(33*temp44/ell2 - 18*temp42/ell + 1)/ell - 1120*a*temp9*(3*temp42/ell - 1)/ell + 160*a*(7*temp42/ell - 1)*temp3 - 80*temp10 + 40*temp5*(7*temp42/ell - 1))*temp12/ell27
        elif order==8:
            kernel = 945*(-165*a4*temp9*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell3 + 105*a4*temp3*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell2 - 168*a3*temp10*(143*temp44/ell2 - 110*temp42/ell + 15)/ell2 + 420*a3*temp5*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell2 - 252*a2*temp9*(143*temp44/ell2 - 110*temp42/ell + 15)/ell2 + 420*a2*temp3*(33*temp44/ell2 - 18*temp42/ell + 1)/ell - 1120*a*temp4*(3*temp42/ell - 1)*temp3/ell + 280*a*temp5*(33*temp44/ell2 - 18*temp42/ell + 1)/ell - 280*temp9*(3*temp42/ell - 1)/ell + 40*(7*temp42/ell - 1)*temp3)*temp12/ell27
        elif order==9:
            kernel = 945*(-1320*a4*temp10*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell2 + 165*a4*temp5*(4199*temp48/ell4 - 6188*temp46/ell3 + 2730*temp44/ell2 - 364*temp42/ell + 7)/ell2 - 5280*a3*temp9*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell2 + 3360*a3*temp3*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell - 2016*a2*temp10*(143*temp44/ell2 - 110*temp42/ell + 15)/ell + 5040*a2*temp5*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell - 1344*a*temp9*(143*temp44/ell2 - 110*temp42/ell + 15)/ell + 2240*a*temp3*(33*temp44/ell2 - 18*temp42/ell + 1) - 2240*temp4*(3*temp42/ell - 1)*temp3 + 560*temp5*(33*temp44/ell2 - 18*temp42/ell + 1))*temp12/ell29
        elif order==10:
            kernel = 8505*(-715*a4*temp9*(2261*temp48/ell4 - 3876*temp46/ell3 + 2142*temp44/ell2 - 420*temp42/ell + 21)/ell3 + 165*a4*temp3*(4199*temp48/ell4 - 6188*temp46/ell3 + 2730*temp44/ell2 - 364*temp42/ell + 7)/ell2 - 5280*a3*temp10*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell2 + 660*a3*temp5*(4199*temp48/ell4 - 6188*temp46/ell3 + 2730*temp44/ell2 - 364*temp42/ell + 7)/ell2 - 7920*a2*temp9*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell2 + 5040*a2*temp3*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell - 1344*a*temp10*(143*temp44/ell2 - 110*temp42/ell + 15)/ell + 3360*a*temp5*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell - 336*temp9*(143*temp44/ell2 - 110*temp42/ell + 15)/ell + 560*temp3*(33*temp44/ell2 - 18*temp42/ell + 1))*temp12/ell29
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**3 * sy.cos(c) \
                * (3*(a*sy.cos(c)*sy.sin(e-f)) \
                * (a * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f)) - b)
                / (sy.sqrt(a**2 + b**2 - 2 * a * b \
                * (sy.sin(d) * sy.sin(c) + sy.cos(d) * sy.cos(c) * sy.cos(e - f))))**5)
            d_kernel = sy.diff(kernel, a, order-1)
            kernel = sy.N(d_kernel.subs({a: r_tess, b: r_cal, c: phi_tess, \
                d: phi_cal, e: lambda_tess, f: lambda_cal}))
    else:
        if order==1:
            kernel = 3*a3*temp5*temp12/ell25
        elif order==2:
            kernel = 3*a3*(-5*a + 5*b*temp3)*temp5*temp12/ell27 + 3*a3*temp3*temp12/ell25 + 9*a2*temp5*temp12/ell25
        elif order==3:
            kernel = 3*a*(-10*a2*temp10/ell + 5*a2*temp5*(7*temp42/ell - 1)/ell - 30*a*temp9/ell + 12*a*temp3 - 6*b)*temp12/ell25
        elif order==4:
            kernel = 9*(-35*a3*temp9*(3*temp42/ell - 1)/ell2 + 5*a3*(7*temp42/ell - 1)*temp3/ell - 30*a2*temp10/ell + 15*a2*temp5*(7*temp42/ell - 1)/ell - 30*a*temp9/ell + 8*a*temp3 - 2*b)*temp12/ell25
        elif order==5:
            kernel = 9*(-140*a3*temp4*(3*temp42/ell - 1)*temp3/ell2 + 35*a3*temp5*(33*temp44/ell2 - 18*temp42/ell + 1)/ell2 - 420*a2*temp9*(3*temp42/ell - 1)/ell2 + 60*a2*(7*temp42/ell - 1)*temp3/ell - 120*a*temp10/ell + 60*a*temp5*(7*temp42/ell - 1)/ell - 40*temp9/ell + 8*temp2 + 8*temp6*temp7)*temp12/ell25
        elif order==6:
            kernel = 45*(-21*a3*temp9*(143*temp44/ell2 - 110*temp42/ell + 15)/ell2 + 35*a3*temp3*(33*temp44/ell2 - 18*temp42/ell + 1)/ell - 420*a2*temp4*(3*temp42/ell - 1)*temp3/ell + 105*a2*temp5*(33*temp44/ell2 - 18*temp42/ell + 1)/ell - 420*a*temp9*(3*temp42/ell - 1)/ell + 60*a*(7*temp42/ell - 1)*temp3 - 40*temp10 + 20*temp5*(7*temp42/ell - 1))*temp12/ell27
        elif order==7:
            kernel = 135*(-42*a3*temp10*(143*temp44/ell2 - 110*temp42/ell + 15)/ell2 + 105*a3*temp5*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell2 - 126*a2*temp9*(143*temp44/ell2 - 110*temp42/ell + 15)/ell2 + 210*a2*temp3*(33*temp44/ell2 - 18*temp42/ell + 1)/ell - 840*a*temp4*(3*temp42/ell - 1)*temp3/ell + 210*a*temp5*(33*temp44/ell2 - 18*temp42/ell + 1)/ell - 280*temp9*(3*temp42/ell - 1)/ell + 40*(7*temp42/ell - 1)*temp3)*temp12/ell27
        elif order==8:
            kernel = 945*(-165*a3*temp9*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell2 + 105*a3*temp3*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell - 126*a2*temp10*(143*temp44/ell2 - 110*temp42/ell + 15)/ell + 315*a2*temp5*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell - 126*a*temp9*(143*temp44/ell2 - 110*temp42/ell + 15)/ell + 210*a*temp3*(33*temp44/ell2 - 18*temp42/ell + 1) - 280*temp4*(3*temp42/ell - 1)*temp3 + 70*temp5*(33*temp44/ell2 - 18*temp42/ell + 1))*temp12/ell29
        elif order==9:
            kernel = 945*(-1320*a3*temp10*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell2 + 165*a3*temp5*(4199*temp48/ell4 - 6188*temp46/ell3 + 2730*temp44/ell2 - 364*temp42/ell + 7)/ell2 - 3960*a2*temp9*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell2 + 2520*a2*temp3*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell - 1008*a*temp10*(143*temp44/ell2 - 110*temp42/ell + 15)/ell + 2520*a*temp5*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1)/ell - 336*temp9*(143*temp44/ell2 - 110*temp42/ell + 15)/ell + 560*temp3*(33*temp44/ell2 - 18*temp42/ell + 1))*temp12/ell29
        elif order==10:
            kernel = 8505*(-715*a3*temp9*(2261*temp48/ell4 - 3876*temp46/ell3 + 2142*temp44/ell2 - 420*temp42/ell + 21)/ell2 + 165*a3*temp3*(4199*temp48/ell4 - 6188*temp46/ell3 + 2730*temp44/ell2 - 364*temp42/ell + 7)/ell - 3960*a2*temp10*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell + 495*a2*temp5*(4199*temp48/ell4 - 6188*temp46/ell3 + 2730*temp44/ell2 - 364*temp42/ell + 7)/ell - 3960*a*temp9*(221*temp46/ell3 - 273*temp44/ell2 + 91*temp42/ell - 7)/ell + 2520*a*temp3*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1) - 336*temp10*(143*temp44/ell2 - 110*temp42/ell + 15) + 840*temp5*(143*temp46/ell3 - 143*temp44/ell2 + 33*temp42/ell - 1))*temp12/ell211
        else:
            a, b, c, d, e, f = sy.symbols('a b c d e f')
            kernel = a**2 * sy.cos(c) \
                * (3*(a*sy.cos(c)*sy.sin(e-f)) \
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

