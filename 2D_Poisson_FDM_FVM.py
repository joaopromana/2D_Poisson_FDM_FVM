# Solution of 2D Poisson's Equation with FDM and FVM
# Joao Pedro Colaco Romana 2022

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la


def FDLaplacian2D(Nx, Ny, dx, dy):
    Dx = sp.diags([(Nx - 1) * [1 / dx], (Nx - 1) * [-1 / dx]], [0, -1], shape=(Nx, Nx - 1))
    Dy = sp.diags([(Ny - 1) * [1 / dy], (Ny - 1) * [-1 / dy]], [0, -1], shape=(Ny, Ny - 1))
    Lxx = Dx.transpose().dot(Dx)
    Lyy = Dy.transpose().dot(Dy)
    A = sp.kron(sp.eye(Ny - 1), Lxx) + sp.kron(Lyy, sp.eye(Nx - 1))
    return A


def sourcefunc(x, y):
    x = np.array(x)
    y = np.array(y)
    if x.ndim == 0 and y.ndim == 0:  # x and y are scalars
        f = 0
        for i in range(1, 10):
            for j in range(1, 5):
                f += np.exp(-alpha * ((x - i) ** 2 + (y - j) ** 2))
        return f
    else:  # x and y are arrays
        f = []
        f_local = 0
        for k in range(0, len(x)):
            for i in range(1, 10):
                for j in range(1, 5):
                    f_local += np.exp(-alpha * ((x[k] - i) ** 2 + (y[k] - j) ** 2))
            f.append(f_local)
            f_local = 0
        return np.array(f)


def coeffK1(x, y):
    x = np.array(x)
    y = np.array(y)
    if x.ndim == 0 and y.ndim == 0:  # x and y are scalars
        K = 1.0
        return K
    else:  # x and y are arrays
        K = np.ones((np.shape(x)[0], np.shape(y)[1]))
        return K


def coeffK2(x, y):
    x = np.array(x)
    y = np.array(y)
    if x.ndim == 0 and y.ndim == 0:  # x and y are scalars
        K = 1.0 + 0.1 * (x + y + x * y)
        return K
    else:  # x and y are arrays
        K = []
        for k in range(0, len(x)):
            K.append(1.0 + 0.1 * (x[k] + y[k] + x[k] * y[k]))
        return np.array(K)


def create2DLFVM(x, y, dx, dy, coeffFun):
    x = x.T[0]
    y = y[0]
    diag_E = []
    diag_S = []
    diag_W = []
    diag_N = []
    diag_C = []
    # Nested loop in lexicographic ordering
    for j in range(0, np.shape(y)[0]):
        for i in range(0, np.shape(x)[0]):
            E = -coeffFun(x[i] - dx / 2, y[j]) / dx ** 2
            S = -coeffFun(x[i], y[j] - dy / 2) / dy ** 2
            W = -coeffFun(x[i] + dx / 2, y[j]) / dx ** 2
            N = -coeffFun(x[i], y[j] + dy / 2) / dy ** 2
            diag_C.append(-E - S - W - N)
            diag_N.append(N)
            if i == 0 and j == 0:
                diag_W.append(W)
            elif j == 0:
                diag_E.append(E)
                if i < np.shape(x)[0] - 1:
                    diag_W.append(W)
                else:
                    diag_W.append(0)
            else:
                diag_S.append(S)
                if i < np.shape(x)[0] - 1:
                    diag_W.append(W)
                else:
                    diag_W.append(0)
                if i == 0:
                    diag_E.append(0)
                else:
                    diag_E.append(E)
    diag_C = np.array(diag_C)
    diag_E = np.array(diag_E)
    diag_S = np.array(diag_S)
    diag_W = np.array(diag_W)
    diag_N = np.array(diag_N)
    A = sp.diags([diag_C, diag_E, diag_S, diag_W, diag_N], [0, -1, -np.shape(x)[0], 1, np.shape(x)[0]], format='csc')
    return A


LeftX = 0
RightX = 10.0
LeftY = 0
RightY = 5.0
alpha = 40

print('1 - FD Method')
print('2 - FV Method')
option = int(input('Choose an option: '))

Nx = int(input('Number of Sub-intervals in x-direction: '))
Ny = int(input('Number of sub-intervals in y-direction: '))

dx = (RightX - LeftX) / Nx  # grid step in x-direction
dy = (RightY - LeftY) / Ny  # grid step in y-direction

if option == 1:
    # 2D FD Laplacian on Rectangular Domain
    A = FDLaplacian2D(Nx, Ny, dx, dy)

    plt.figure('Figure 1')
    plt.spy(A)
    plt.grid('on')
    plt.title(r'Structure of Matrix $A$ for $N_x=$' + str(Nx) + r" and $N_y=$" + str(Ny))
    plt.savefig('FDM_matrix_structure_Nx%02d_Ny%02d.png' % (Nx, Ny), bbox_inches='tight', dpi = 300)

    x, y = np.mgrid[LeftX + dx:RightX:dx, LeftY + dy:RightY:dy]
    print(x, y)

    f = sourcefunc(x, y)

    # Visualizing the Source Function
    plt.ion()
    plt.figure('Figure 2')
    plt.clf()
    plt.imshow(f.T, extent=[LeftX+dx/2, RightX-dx/2, LeftY+dy/2, RightY-dy/2], origin='lower')
    plt.colorbar()
    plt.title(r"Source Function for $N_x =$" + str(Nx) + r" and $N_y=$" + str(Ny))
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.savefig('source_function_Nx%02d_Ny%02d.png' % (Nx, Ny), bbox_inches='tight', dpi = 300)

    # Lexicographic Source Vector
    fLX = np.reshape(f, ((Nx - 1) * (Ny - 1), 1), order='F')

    u = la.spsolve(A, fLX)

    # Reshaping the Solution Vector into 2D array
    uArr = np.reshape(u, ((Nx - 1), (Ny - 1)), order='F')

    # Visualizing the Solution
    plt.figure('Figure 3')
    plt.clf()
    plt.imshow(uArr.T, extent=[LeftX+dx/2, RightX-dx/2, LeftY+dy/2, RightY-dy/2], origin='lower')
    plt.colorbar()
    plt.title(r"Solution for $N_x =$" + str(Nx) + r" and $N_y=$" + str(Ny))
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.savefig('FDM_solution_Nx%02d_Ny%02d.png' % (Nx, Ny), bbox_inches='tight', dpi = 300)

    plt.show(block=True)

elif option == 2:
    x, y = np.mgrid[LeftX + dx:RightX:dx, LeftY + dy:RightY:dy]

    K1 = coeffK1(x, y)
    K2 = coeffK2(x, y)

    # Visualizing the Coefficient Functions
    fig, axs = plt.subplots(2)
    im1 = axs[0].imshow(K1.T, extent=[LeftX+dx/2, RightX-dx/2, LeftY+dy/2, RightY-dy/2], origin='lower')
    axs[0].set_title(r"Coefficient Function $k_1(x,y)$ for $N_x =$" + str(Nx) + r" and $N_y=$" + str(Ny))
    im2 = axs[1].imshow(K2.T, extent=[LeftX+dx/2, RightX-dx/2, LeftY+dy/2, RightY-dy/2], origin='lower')
    axs[1].set_title(r"Coefficient Function $k_2(x,y)$ for $N_x =$" + str(Nx) + r" and $N_y=$" + str(Ny))
    fig.colorbar(im1)
    fig.colorbar(im2)
    for ax in axs.flat:
        ax.set(xlabel='x', ylabel='y')
    # Hide x labels and tick labels for top plots
    axs[0].get_xaxis().set_visible(False)
    for ax in axs.flat:
        ax.label_outer()
    plt.savefig('coefficient_functions_Nx%02d_Ny%02d.png' % (Nx, Ny), bbox_inches='tight', dpi = 300)

    A_1 = create2DLFVM(x, y, dx, dy, coeffK1)
    A_2 = create2DLFVM(x, y, dx, dy, coeffK2)
    # print(A_1.toarray())
    # print(A_2.toarray())

    f = sourcefunc(x, y)

    # Lexicographic Source Vector
    fLX = np.reshape(f, ((Nx - 1) * (Ny - 1), 1), order='F')

    u_1 = la.spsolve(A_1, fLX)
    u_2 = la.spsolve(A_2, fLX)

    # Reshaping the Solutions Vector into 2D array
    uArr_1 = np.reshape(u_1, ((Nx - 1), (Ny - 1)), order='F')
    uArr_2 = np.reshape(u_2, ((Nx - 1), (Ny - 1)), order='F')

    # Visualizing the Solution with constant Coefficient
    plt.figure('Figure 4')
    plt.clf()
    plt.imshow(uArr_1.T, extent=[LeftX+dx/2, RightX-dx/2, LeftY+dy/2, RightY-dy/2], origin='lower')
    plt.colorbar()
    plt.title(r"Solution $u(x,y)$ for $k=k_1(x,y)$, $N_x =$" + str(Nx) + r" and $N_y$=" + str(Ny))
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.savefig('FVM_solution_K1_Nx%02d_Ny%02d.png' % (Nx, Ny), bbox_inches='tight', dpi = 300)

    # Visualizing the Solution with varying Coefficient
    plt.figure('Figure 5')
    plt.clf()
    plt.imshow(uArr_2.T, extent=[LeftX+dx/2, RightX-dx/2, LeftY+dy/2, RightY-dy/2], origin='lower')
    plt.colorbar()
    plt.title(r"Solution $u(x,y)$ for $k=k_2(x,y)$, $N_x =$" + str(Nx) + r" and $N_y=$" + str(Ny))
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.savefig('FVM_solution_K2_Nx%02d_Ny%02d.png' % (Nx, Ny), bbox_inches='tight', dpi = 300)

    plt.show(block=True)

else:
    print('Incorrect option, try again')
