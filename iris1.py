from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load iris dataset
dataset = numpy.loadtxt("iris_train.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:4]
Y = dataset[:,4:7]

# create model
model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu', kernel_initializer="uniform"))
model.add(Dense(8, activation='relu', kernel_initializer="uniform"))
model.add(Dense(4, activation='relu', kernel_initializer="uniform"))
model.add(Dense(3, activation='softmax', kernel_initializer="uniform"))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

# evaluate the model
#scores = model.evaluate(X, Y)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#test the model
#above eval was commented out when testing on test data
dataset2 = numpy.loadtxt("iris_test.csv", delimiter=",")
X2 = dataset2[:,0:4]
Y2 = dataset2[:,4:7]
scores = model.evaluate(X2, Y2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
