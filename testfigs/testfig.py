import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.plot(range(10),range(10))
plt.savefig('testfig1.png')

plt.plot(range(30),np.sin(range(30)))
plt.savefig('testfig2.png')

plt.plot(range(100),np.cos(range(100)))
plt.savefig('testfig3.png')

plt.close()

plt.plot(range(10),range(10))
plt.savefig('testfig1_close.png')
plt.close()

plt.plot(range(30),np.sin(range(30)))
plt.savefig('testfig2_close.png')
plt.close()

plt.plot(range(100),np.cos(range(100)))
plt.savefig('testfig3_close.png')
plt.close()
