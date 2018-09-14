import matplotlib.pyplot as plt
import numpy as np

#...some data processing
ndir = 16
nrings = 2

used_theta = np.zeros(ndir)
for i in range(ndir):
    used_theta[i] = 2.*np.pi*i / ndir
used_rad = np.zeros(nrings)
for i in range(nrings):
    used_rad[i] = (1.*(i+1)) / (1.*nrings)

print(used_rad)


def polar_plot(r, phi, data):
    """
    Plots a 2D array in polar coordinates.

    :param r: array of length n containing r coordinates
    :param phi: array of length m containing phi coordinates
    :param data: array of shape (n, m) containing the data to be plotted
    """
    # Generate the mesh
    phi_grid, r_grid = np.meshgrid(phi, r)
    x, y = r_grid*np.cos(phi_grid), r_grid*np.sin(phi_grid)
    plt.pcolormesh(x, y, data)
    plt.show()


# data = np.ones((nrings, ndir))

# polar_plot(used_rad, used_theta, data)


#fake data:
a = np.linspace(0, 2.*np.pi, ndir)
b = np.linspace(1./nrings, 1, nrings)
A, B = np.meshgrid(a, b)
c = np.random.random(A.shape + (3,))

#actual plotting
import matplotlib.cm as cm
ax = plt.subplot(111, polar=True)
ax.set_yticklabels([])
ctf = ax.contourf(a, b, c, cmap=cm.jet)
# plt.colorbar(ctf)
plt.show()


