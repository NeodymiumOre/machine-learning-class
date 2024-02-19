import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    # define data (2)
    X = np.array([
        [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1],
        [2.4, 0.7, 2.9, 2.2, 3, 2.7, 1.6, 1.1, 1.6, 0.9]
    ])

    # plot X
    plt.figure()
    plt.style.use('seaborn-v0_8')
    plt.scatter(X[0], X[1], s=20, marker='o', c='b',)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('square')
    plt.savefig('pictures/x.png', bbox_inches='tight')
    plt.show()

    # center data (5)
    X_centered = X
    X_centered[0] = X[0] - np.mean(X[0])
    X_centered[1] = X[1] - np.mean(X[1])

    # plot centered X
    plt.figure()
    plt.style.use('seaborn-v0_8')
    plt.scatter(X[0], X[1], s=20, marker='o', c='b',)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('square')
    plt.savefig('pictures/x_centered.png', bbox_inches='tight')
    plt.show()

    # calculate covariance matrix (6)
    cov_matrix = np.cov(X_centered)
    print(f"Covariance matrix: {cov_matrix}")

    # calculate eigendecomposition (7)
    D, V = np.linalg.eig(cov_matrix)
    print(f"Eigenvalues: {D}")
    print(f"Eigenvectors: {V}")

    # plot eigenvectors (9)
    fig2 = plt.figure()
    plt.style.use('seaborn-v0_8')
    plt.quiver(V[0, 0], V[1, 0], color=['r'], scale=4)
    plt.quiver(V[0, 1], V[1, 1], color=['b'], scale=4)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('square')
    plt.savefig('pictures/eigv.png', bbox_inches='tight')
    plt.show()

    # prepare matrices u and z (10)
    u = V[0] if D[0] > D[1] else V[1]
    u = np.expand_dims(u, 1)
    z = V[0] if D[0] < D[1] else V[1]
    z = np.expand_dims(z, 1)

    print(f"Matrux u: {u}")
    print(f"Matrux z: {z}")

    # perform projection from (11)
    Y = np.outer(np.dot(np.transpose(X), u), np.transpose(u))
    Y = np.transpose(Y)
    print(f"Vectors from Y: {Y}")
    
    # plot Y vector (12)
    fig3 = plt.figure()
    plt.style.use('seaborn-v0_8')
    plt.quiver(Y[0, 0], Y[0, 9], color=['r'])
    plt.quiver(Y[1, 0], Y[1, 9], color=['b'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('square')
    plt.savefig('pictures/eigv_u.png', bbox_inches='tight')
    plt.show()

    # perform projection from (13-1)
    Z = np.outer(np.dot(np.transpose(X), z), np.transpose(z))
    Z = np.transpose(Z)
    print(f"Vectors from Z: {Z}")
    
    # plot Y vector (13-2)
    fig4 = plt.figure()
    plt.style.use('seaborn-v0_8')
    plt.quiver(Z[0, 0], Z[0, 9], color=['r'])
    plt.quiver(Z[1, 0], Z[1, 9], color=['b'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('square')
    plt.savefig('pictures/eigv_z.png', bbox_inches='tight')
    plt.show()

