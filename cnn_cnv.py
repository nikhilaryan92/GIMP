from keras.layers import Input,Dropout, Flatten,Dense
from sklearn.model_selection import StratifiedKFold,train_test_split  
import numpy,math
from sklearn.metrics import roc_curve,auc
from keras.models import Model
from keras.utils import plot_model
from keras import initializers,regularizers,optimizers
from keras.layers.convolutional import Conv1D
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import numpy as np
epochs = 50

def sensitivity(y_true, y_pred):
    #y_pred = K.tf.convert_to_tensor(y_pred, np.float32) #Converting y_pred from numpy to tensor 
    #y_true = K.tf.convert_to_tensor(y_true, np.float32) #Converting y_true from numpy to tensor
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    Sn=(true_positives / (possible_positives + K.epsilon()))
    #with K.tf.Session() as sess: 	#Converting Sn 
        #Sn=sess.run(Sn)		# from tensor  to numpy
    return Sn

def specificity(y_true, y_pred):
    #y_pred = K.tf.convert_to_tensor(y_pred, np.float32) #Converting y_pred from numpy to tensor 
    #y_true = K.tf.convert_to_tensor(y_true, np.float32) #Converting y_true from numpy to tensor
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    Sp= (true_negatives / (possible_negatives + K.epsilon()))
    #with K.tf.Session() as sess: 	#Converting Sp 
       # Sp=sess.run(Sp)		# from tensor  to numpy
    return Sp




# fix random seed for reproducibility
numpy.random.seed(1)
# load CNV dataset
#dataset_cnv = numpy.loadtxt("A:/Project/Incomplete Multiview/Data/Generated/METABRIC_cnv_1600.csv", delimiter=",")# Change the path to your local system
#dataset_cnv = numpy.loadtxt("A:/Project/STACKED RF/Data/METABRIC_cnv_1980.txt", delimiter="\t")# Change the path to your local system
#dataset_cnv = numpy.loadtxt("A:/Project/Incomplete Multiview/Data/Generated/tcga_cnv_generated.csv", delimiter=",") # Change the path to your local system
#dataset_cnv = numpy.loadtxt("A:/Project/Incomplete Multiview/Data/Generated/metabric_cnv_final.csv", delimiter=",") # Change the path to your local system
dataset_cnv = numpy.loadtxt("A:/Project/Incomplete Multiview/Data/Generated/tcga_cnv_final.csv", delimiter=",") # Change the path to your local system
# split into input (X) and output (Y) variables
X_cnv = dataset_cnv[:,0:200]
#Y_cnv = dataset_cnv[:,200]
Y_cnv=np.loadtxt("A:/Project/Incomplete Multiview/Data/tcga5yearCutOff_incomplete.txt")
kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)
accuracy_cnv = []
Sn_cnv = []
Sp_cnv = []
i=1
for train_index, test_index in kfold.split(X_cnv, Y_cnv):
	print(i,'th Fold *******************************')
	i=i+1
	#Spliting the data set into training and testing
	x_train_cnv, x_test_cnv=X_cnv[train_index],X_cnv[test_index]	
	y_train_cnv, y_test_cnv = Y_cnv[train_index],Y_cnv[test_index]
	x_train_cnv = numpy.expand_dims(x_train_cnv, axis=2)
	x_test_cnv = numpy.expand_dims(x_test_cnv, axis=2)
	
	# first CNV Model
	#init=initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None)
	init=initializers.glorot_normal(seed=1)
	bias_init =initializers.Constant(value=0.1)
	main_input1 = Input(shape=(200,1))
	conv1 = Conv1D(filters=4,kernel_size=15,strides=2,activation='tanh',padding='same',kernel_initializer=init,bias_initializer=bias_init)(main_input1)
	flat1 = Flatten()(conv1)
	dropout1 = Dropout(0.50)(flat1)
	dense1 = Dense(150,activation='tanh',name='dense1',kernel_initializer=init,bias_initializer=bias_init)(dropout1)
	dropout2 = Dropout(0.25,name='dropout2')(dense1)
	output = Dense(1, activation='sigmoid',activity_regularizer= regularizers.l2(0.01),kernel_initializer=init,bias_initializer=bias_init)(dropout2)
	model =	Model(inputs=main_input1, outputs=output)
	# summarize layers
	#print(model.summary())
	# plot graph
	#plot_model(model, to_file='/home/nikhil/Desktop/Project/nik/Code/Submodels/ModelDesign/CNN CNV.png')# Change the path to your local system
	def exp_decay(epoch):
		initial_lrate = 0.01
		k = 0.1
		lrate = initial_lrate * math.exp(-k*epoch)
		return lrate
	lrate = LearningRateScheduler(exp_decay)
	adams=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss='binary_crossentropy', optimizer=adams, metrics=['accuracy',sensitivity, specificity])
	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	x_train, x_val, y_train, y_val = train_test_split(x_train_cnv, y_train_cnv, test_size=0.2,stratify=y_train_cnv)
	model.fit(x_train, y_train, epochs=epochs, batch_size=8,verbose=2,validation_data=(x_val,y_val),callbacks=[lrate])
	#model.fit(x_train_cnv, y_train_cnv, epochs=epochs,validation_data=(x_val,y_val), batch_size=8,verbose=2)
	cnv_scores = model.evaluate(x_test_cnv, y_test_cnv,verbose=2)
	print("%s: %.2f%%" % (model.metrics_names[1], cnv_scores[1]*100))
	print("%s: %.2f%%" % (model.metrics_names[2], cnv_scores[2]*100))
	print("%s: %.2f%%" % (model.metrics_names[3], cnv_scores[3]*100))
	accuracy_cnv.append(cnv_scores[1] * 100)
	Sn_cnv.append(cnv_scores[2] * 100)
	Sp_cnv.append(cnv_scores[3] * 100)	
	intermediate_layer_model = Model(inputs=main_input1,outputs=dropout2)
print("Accuracy = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(accuracy_cnv), numpy.std(accuracy_cnv)))
print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_cnv), numpy.std(Sn_cnv)))
print("Specificity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sp_cnv), numpy.std(Sp_cnv)))
#Plotting
X_train, X_test, y_train, y_test = train_test_split(X_cnv, Y_cnv, test_size=0.2,stratify=Y_cnv)
X_test=numpy.expand_dims(X_test, axis=2)
pred_cnv = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, pred_cnv)

X_cnv = numpy.expand_dims(X_cnv, axis=2)
# for extracting final layer features 
#y_pred =  model.predict(X_cnv)
# for extracting one layer before final layer features
y_pred = intermediate_layer_model.predict(X_cnv)
stacked_feature=numpy.concatenate((y_pred,Y_cnv[:,None]),axis=1)
with open('A:/Project/STACKED RF/Data/gatedAtnCnvOutput.csv', 'w') as f:# Change the path to your local system
	for item_cnv in stacked_feature:
		for elem in item_cnv:
			f.write(str(elem)+'\t')
		f.write('\n')
        
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])


for threshold in numpy.arange(0,1,0.05):
    print('********* Threshold = ', threshold, ' ************')
    evaluate_threshold(threshold)
    

roc_auc = auc(fpr, tpr)
plt.plot(fpr,tpr, 'r', label = 'CNN-CNV = %0.3f' %roc_auc)
plt.xlabel('1-Sp (False Positive Rate)')
plt.ylabel('Sn (True Positive Rate)')
plt.title('Receiver Operating Characteristics')
plt.legend()
plt.show()

