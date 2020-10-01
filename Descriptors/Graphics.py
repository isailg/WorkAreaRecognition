import matplotlib.pyplot as plt
import time
import numpy as np
import xlrd 


t="%s"%time.strftime("%j %H:%M:%S")

indl=[1,4,7]
indt=[0,3,6]

x= [1,2,3,4,5,6,7,8,9,10]
yl= np.empty ([3,10])
yt= np.empty ([3,10])

#Obtener datos de xls
book = xlrd.open_workbook("FM.xls")
sheet = book.sheet_by_index(0)

############# l ###############    
yl[0,0] = sheet.cell_value( 1,1)
yl[0,1] = sheet.cell_value( 2,1)
yl[0,2] = sheet.cell_value( 3,1)
yl[0,3] = sheet.cell_value( 4,1)
yl[0,4] = sheet.cell_value( 5,1)
yl[0,5] = sheet.cell_value( 6,1)
yl[0,6] = sheet.cell_value( 7,1)
yl[0,7] = sheet.cell_value( 8,1)
yl[0,8] = sheet.cell_value( 9,1)
yl[0,9] = sheet.cell_value( 10,1)

yl[1,0] = sheet.cell_value( 1,4)
yl[1,1] = sheet.cell_value( 2,4)
yl[1,2] = sheet.cell_value( 3,4)
yl[1,3] = sheet.cell_value( 4,4)
yl[1,4] = sheet.cell_value( 5,4)
yl[1,5] = sheet.cell_value( 6,4)
yl[1,6] = sheet.cell_value( 7,4)
yl[1,7] = sheet.cell_value( 8,4)
yl[1,8] = sheet.cell_value( 9,4)
yl[1,9] = sheet.cell_value( 10,4)

yl[2,0] = sheet.cell_value( 1,7)
yl[2,1] = sheet.cell_value( 2,7)
yl[2,2] = sheet.cell_value( 3,7)
yl[2,3] = sheet.cell_value( 4,7)
yl[2,4] = sheet.cell_value( 5,7)
yl[2,5] = sheet.cell_value( 6,7)
yl[2,6] = sheet.cell_value( 7,7)
yl[2,7] = sheet.cell_value( 8,7)
yl[2,8] = sheet.cell_value( 9,7)
yl[2,9] = sheet.cell_value( 10,7)

############# t ###############    
yt[0,0] = sheet.cell_value( 1,0)
yt[0,1] = sheet.cell_value( 2,0)
yt[0,2] = sheet.cell_value( 3,0)
yt[0,3] = sheet.cell_value( 4,0)
yt[0,4] = sheet.cell_value( 5,0)
yt[0,5] = sheet.cell_value( 6,0)
yt[0,6] = sheet.cell_value( 7,0)
yt[0,7] = sheet.cell_value( 8,0)
yt[0,8] = sheet.cell_value( 9,0)
yt[0,9] = sheet.cell_value( 10,0)

yt[1,0] = sheet.cell_value( 1,3)
yt[1,1] = sheet.cell_value( 2,3)
yt[1,2] = sheet.cell_value( 3,3)
yt[1,3] = sheet.cell_value( 4,3)
yt[1,4] = sheet.cell_value( 5,3)
yt[1,5] = sheet.cell_value( 6,3)
yt[1,6] = sheet.cell_value( 7,3)
yt[1,7] = sheet.cell_value( 8,3)
yt[1,8] = sheet.cell_value( 9,3)
yt[1,9] = sheet.cell_value( 10,3)

yt[2,0] = sheet.cell_value( 1,6)
yt[2,1] = sheet.cell_value( 2,6)
yt[2,2] = sheet.cell_value( 3,6)
yt[2,3] = sheet.cell_value( 4,6)
yt[2,4] = sheet.cell_value( 5,6)
yt[2,5] = sheet.cell_value( 6,6)
yt[2,6] = sheet.cell_value( 7,6)
yt[2,7] = sheet.cell_value( 8,6)
yt[2,8] = sheet.cell_value( 9,6)
yt[2,9] = sheet.cell_value( 10,6)
       

########### plot ###############



fig, ax =plt.subplots()
line1, = ax.plot(x, yt[0], label='BF-ORB')
line2, = ax.plot(x, yt[1], label='BF-SIFT')
line3, = ax.plot(x, yt[2], label='FLANN')
ax.set_xlabel('Test Number')
ax.set_ylabel('Seconds')
ax.legend()
plt.xticks([1,2,3,4,5,6,7,8,9,10],
           ["1","2","3","4","5","6","7","8","9","10"])
plt.title('Processing Time')
plt.savefig('../Descriptors/Graphs/Processing Time'+t+'.png')

fig, ax2 =plt.subplots()
line1, = ax2.plot(x, yl[0], label='BF-ORB')
line2, = ax2.plot(x, yl[1], label='BF-SIFT')
line3, = ax2.plot(x, yl[2], label='FLANN')

ax2.set_xlabel('Test Number')
ax2.set_ylabel('Number of Matches')
ax2.legend()
plt.xticks([1,2,3,4,5,6,7,8,9,10],
           ["1","2","3","4","5","6","7","8","9","10"])
plt.title('Matches of Keypoints')
plt.savefig('../Descriptors/Graphs/Matches'+t+'.png')



plt.show()




