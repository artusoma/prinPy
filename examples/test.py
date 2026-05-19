from prinpy_rs import clpg

# Some other modules
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
import timeit

theta = np.linspace(0,np.pi*3, 1000)
r = np.linspace(0,1,1000) ** .5

x_data = r * np.cos(theta) + np.random.normal(scale = .02, size = 1000)
y_data = r * np.sin(theta) + np.random.normal(scale = .02, size = 1000)

#`cl = CLPCG()  # Create CLPCG object

# the fit() method calculates the principal curve
# e_max is determined through trial and error as of
# now, but aim for about 1/2 data error and adjust from
# there. 
start = timeit.default_timer()

# cl.fit(x_data, y_data, e_max = .03)  # CLPCG.fit() to fit PC
data = np.array([x_data, y_data], dtype=np.float32).T
res = clpg(data, .1)

stop = timeit.default_timer()

print("Took %f seconds" % (stop - start))

fig, ax = plt.subplots()
ax.scatter(x_data, y_data, s = 3, alpha = .7)
ax.scatter(res[:,0], res[:,1], s = 40, c = 'green')
fig.show()
input()