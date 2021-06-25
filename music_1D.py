import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import svd, matrix_rank

twpi = 2*math.pi
derad=np.pi/180

# 到来方向(列ベクトル)
number_user = 2 # 到来波数
deg = 180 # 到来角
dataNum = 1 # 学習データ数

# az = np.random.randint(0,deg,size=(1,number_user)) # 到来角
az = np.array([30, 120])
# 送信信号
ss = 10 # スナップショット数
#SNR = np.random.randint(0,30,1)
SNR = 50
sigma = 10.**(-SNR/10) # 雑音分散
n_p = 10.**(-SNR/10)

# 受信信号
number_ant = 5
c = 3*(10**8)
f0 = 2*(10**9) #　中心周波数
lam = c/f0
space = lam/2 # 素子間隔は半波長

# ステアリングベクトルの計算
A = []
for nu in range(number_user): # 0~number_user-1
    rx = twpi*space*np.cos(az[nu]*derad)/lam
    a_psi = np.exp(-1j*np.matrix(range(number_ant)).T*rx)
    A.append(a_psi)

A = np.concatenate(A, axis=1)

s = np.random.randn(number_user,ss)+1j*np.random.randn(number_user,ss) #ランダム信号(複素数)
rx_wgn = sigma*(np.random.randn(number_ant,ss)+1j*np.random.randn(number_ant,ss)) #雑音
z = rx_wgn.reshape([number_ant,ss])

#==== 相関行列計算
x = A*s + z
Rxx = np.zeros(number_ant*number_ant).reshape((number_ant,number_ant))
for iss in range(ss): #アンサンブル平均
    tmp = np.matrix(x[:,iss])
    Rxx = Rxx + tmp*np.conjugate(tmp.T)
Rxx = Rxx/ss

U,D,V = svd(Rxx) #特異値分解
#==== 雑音部分空間の抽出:固有値分解でも特異値分解でもOK
En = U[:,number_user:(number_ant+1)]

#=== MUSIC Spectrum search
az_search = range(deg)
laz = len(az_search)
SP = np.zeros(laz*1).reshape(laz,1)

for itheta in range(laz):
    th = az_search[itheta]
    psi_search = twpi*space*np.cos(th*derad)/lam
    a_psisearch = np.exp(-1j*np.matrix(range(number_ant)).T*psi_search)
    SP[itheta,:] = abs(1/(np.conjugate(a_psisearch.T)*(En*np.conjugate(En.T))*a_psisearch))

# 正規化＆dB表現
SPmax = np.max(np.max(SP))
SPdb = 10 * np.log10(SP/SPmax)

fig = plt.figure()
plt.plot(az_search,SPdb,label="SPdb")
plt.legend()
plt.show()
plt.savefig("music2D.png")