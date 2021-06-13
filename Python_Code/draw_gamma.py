import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from matplotlib.font_manager import FontProperties
zhfont = FontProperties(fname=r"simhei.ttf", size=18)

fig = plt.figure(figsize=(6, 5))  # 确定绘图区域尺寸
ax1 = fig.add_subplot(1, 1, 1)
x = np.arange(0.01, 20, 0.01) 

# 绘制gamma分布曲线
y1 = st.gamma.pdf(x, 1, scale=2)  # "α=1,β=2"
y2 = st.gamma.pdf(x, 2, scale=2)  # "α=2,β=2"
y3 = st.gamma.pdf(x, 3, scale=2)  # "α=3,β=2"
y4 = st.gamma.pdf(x, 5, scale=1)  # "α=5,β=1"
y5 = st.gamma.pdf(x, 9, scale=0.5)  # "α=9,β=0.5"
# 设置图例
ax1.plot(x, y1, label="α=1,β=2")
ax1.plot(x, y2, label="α=2,β=2")
ax1.plot(x, y3, label="α=3,β=2")
ax1.plot(x, y4, label="α=5,β=1")
ax1.plot(x, y5, label="α=9,β=0.5")

# 设置坐标轴标题
plt.tick_params(labelsize=15)
ax1.set_xlabel('x',size=18)
ax1.set_ylabel('PDF',size=18)
ax1.set_title("Gamma分布示意图",FontProperties=zhfont)
ax1.legend(loc="best",fontsize=10)

plt.show()
