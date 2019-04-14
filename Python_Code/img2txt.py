import numpy as np
import cv2
from PIL import Image
#img=cv2.imread('picture.jpg')

img=np.array(Image.open('picture.jpg'))
print(img.shape)

maxlen=max(img.shape)

height,width = img.shape[:2]

#if maxlen<=512:
#    tout=img
#else:
divlen = int(maxlen / 512)+1
print(divlen)
tout=cv2.resize(img,(512,512),interpolation=cv2.INTER_LINEAR)
tout=img

fid=open("res.csv",'w+')

ynew=tout.shape[0]
xnew=tout.shape[1]
fid.write("%d\n"%(ynew))
fid.write("%d\n"%(xnew))


print(tout[0:10,0,0])


for n in range(0,3):
    temp= np.reshape(tout[:,:,n],ynew*xnew)
    print(temp.shape)
    for x in temp:
        fid.write("%u\n"%x)


fid.close()

print(tout.shape)

# while True:
#     cv2.imshow('t1',tout)
#     if cv2.waitKey(1) & 0XFF == 27:
#         break

