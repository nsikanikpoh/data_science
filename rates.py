import matplotlib.pyplot as pl
pl.gca().add_patch(pl.Circle((0,0), radius=3, fc='blue'))
pl.axis('scaled')
pl.show()

pl.gca().add_patch(pl.Rectangle((0,0), 2,3, fc='blue'))
pl.axis('scaled')
pl.show()


rates = [10,9.5,10,8,7.5,5,10,10]
for i in range(0, len(rates)):
	if rates[i] < 6:
		break
	print(rates[i])