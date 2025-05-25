from matplotlib import pyplot as plt
import rawpy
import numpy as np
import pywt

raw = rawpy.imread("images/stars_raw/IMG_1355.CR2")

print(np.shape(raw))

wavelet = pywt.ContinuousWavelet('cgau1')

#[phi,psi,x] = wavelet.wavefun(level=2)
print(pywt.wavelist(kind='discrete'))

coeffs = pywt.swt2(raw.raw_image, wavelet='haar',level=1,start_level=1)

#plt.imshow(coeffs[1,:,:])
#plt.savefig("gaus.png")


  


