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
from sklearn.metrics import classification_report, confusion_matrix 
import keras.backend as K
import numpy as np
epochs = 40


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



def sensitivity_at_specificity(specificity, **kwargs):
    def metric(labels, predictions):
        # any tensorflow metric
        value, update_op = K.tf.metrics.sensitivity_at_specificity(labels, predictions, specificity, **kwargs)

        # find all variables created for this metric
        metric_vars = [i for i in K.tf.local_variables() if 'sensitivity_at_specificity' in i.name.split('/')[2]]

        # Add metric variables to GLOBAL_VARIABLES collection.
        # They will be initialized for new session.
        for v in metric_vars:
            K.tf.add_to_collection(K.tf.GraphKeys.GLOBAL_VARIABLES, v)

        # force to update metric values
        with K.tf.control_dependencies([update_op]):
            value = K.tf.identity(value)
            return value
    return metric




# fix random seed for reproducibility
numpy.random.seed(1)
# load Clinical dataset
#dataset_clinical = numpy.loadtxt("A:/Project/STACKED RF/Data/METABRIC_clinical_1980.txt", delimiter="\t") # Change the path to your local system
#dataset_clinical = numpy.loadtxt("A:/Project/Incomplete Multiview/Data/Generated/METABRIC_clinical_1600.csv", delimiter=",")
#dataset_clinical = numpy.loadtxt("A:/Project/Incomplete Multiview/Data/Generated/metabric_cln_final.csv", delimiter=",") # Change the path to your local system
dataset_clinical = numpy.loadtxt("A:/Project/Incomplete Multiview/Data/Generated/tcga_cln_final.csv", delimiter=",") # Change the path to your local system
# split into input (X) and output (Y) variables
#X_clinical = dataset_clinical[:,0:25]
#Y_clinical = dataset_clinical[:,25]
X_clinical = dataset_clinical[:,0:11]
Y_clinical = np.loadtxt("A:/Project/Incomplete Multiview/Data/tcga5yearCutOff_incomplete.txt")
kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)
accuracy_clinical = []
Sn_clinical = []
Sp_clinical = []
i=1
for train_index, test_index in kfold.split(X_clinical, Y_clinical):
	print(i,"th Fold *****************************************")
	i=i+1
	x_train_clinical, x_test_clinical=X_clinical[train_index],X_clinical[test_index]	
	y_train_clinical, y_test_clinical = Y_clinical[train_index],Y_clinical[test_index] 	
	x_train_clinical = numpy.expand_dims(x_train_clinical, axis=2)
	x_test_clinical = numpy.expand_dims(x_test_clinical, axis=2)
	# first Clinical CNN Model
	init =initializers.glorot_normal(seed=1)
	bias_init =initializers.Constant(value=0.1)
	main_input1 = Input(shape=(11,1),name='Input')
	conv1 = Conv1D(filters=25,kernel_size=15,strides=2,activation='tanh',padding='same',name='Conv1D',kernel_initializer=init,bias_initializer=bias_init)(main_input1)
	flat1 = Flatten(name='Flatten')(conv1)
	dense1 = Dense(150,activation='tanh',name='dense1',kernel_initializer=init,bias_initializer=bias_init)(flat1)
	output = Dense(1, activation='sigmoid',name='output',activity_regularizer= regularizers.l2(0.01),kernel_initializer=init,bias_initializer=bias_init)(dense1)
	model = Model(inputs=main_input1, outputs=output)
	# summarize layers
	#print(model.summary())
	# plot graph
	#plot_model(model, to_file='/home/nikhil/Desktop/Project/nik/Code/Submodels/ModelDesign/CNN Clinical.png') # Change the path to your local system
	#m = K.tf.keras.metrics.SensitivityAtSpecificity(0.95,num_thresholds=1)
	def exp_decay(epoch):
		initial_lrate = 0.001
		k = 0.1
		lrate = initial_lrate * math.exp(-k*epoch)
		return lrate
	lrate = LearningRateScheduler(exp_decay)
	adams=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	#model.compile(loss='binary_crossentropy', optimizer=adams, metrics=['accuracy',sensitivity, specificity])
	model.compile(loss='binary_crossentropy', optimizer=adams, metrics=[sensitivity_at_specificity(0.95)])
	x_train, x_val, y_train, y_val = train_test_split(x_train_clinical, y_train_clinical, test_size=0.2,stratify=y_train_clinical)
	model.fit(x_train, y_train, epochs=epochs, batch_size=8,verbose=2,validation_data=(x_val,y_val),callbacks=[lrate])
	#model.fit(x_train, y_train, epochs=epochs, batch_size=8,verbose=2,validation_data=(x_val,y_val))	

	clinical_scores = model.evaluate(x_test_clinical, y_test_clinical,verbose=2)
	'''print("%s: %.2f%%" % (model.metrics_names[1], clinical_scores[1]*100))
	print("%s: %.2f%%" % (model.metrics_names[2], clinical_scores[2]*100))
	print("%s: %.2f%%" % (model.metrics_names[3], clinical_scores[3]*100))
	accuracy_clinical.append(clinical_scores[1] * 100)
	Sn_clinical.append(clinical_scores[2] * 100)
	Sp_clinical.append(clinical_scores[3] * 100)'''
	print("%s: %.2f%%" % (model.metrics_names[1], clinical_scores[1]*100))
	accuracy_clinical.append(clinical_scores[1] * 100)
	intermediate_layer_model = Model(inputs=main_input1,outputs=dense1)
print("Accuracy = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(accuracy_clinical), numpy.std(accuracy_clinical)))
'''print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_clinical), numpy.std(Sn_clinical)))
print("Specificity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sp_clinical), numpy.std(Sp_clinical)))'''
#Plotting
X_train, X_test, y_train, y_test = train_test_split(X_clinical, Y_clinical, test_size=0.5,stratify=Y_clinical)
X_test=numpy.expand_dims(X_test, axis=2)
pred_clinical = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, pred_clinical,pos_label=1)


X_clinical = numpy.expand_dims(X_clinical, axis=2)
# for extracting final layer features 
#y_pred =  model.predict(X_clinical)
# for extracting one layer before final layer features
y_pred = intermediate_layer_model.predict(X_clinical)
stacked_feature=numpy.concatenate((y_pred,Y_clinical[:,None]),axis=1)
with open('A:/Project/STACKED RF/Data/gatedAtnClnOutput.csv', 'w') as f: # Change the path to your local system
	for item_clinical in stacked_feature:
		#f.write("%s\n"%item_clinical)
		for elem in item_clinical:
			f.write(str(elem)+'\t')
		f.write('\n')

def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])


for threshold in numpy.arange(0,1,0.05):
    print('********* Threshold = ', threshold, ' ************')
    evaluate_threshold(threshold)

roc_auc = auc(fpr, tpr)
plt.plot(fpr,tpr, 'r', label = 'CNN-CLN = %0.2f' %roc_auc)
plt.xlabel('1-Sp (False Positive Rate)')
plt.ylabel('Sn (True Positive Rate)')
plt.title('Receiver Operating Characteristics')
plt.legend()
plt.show()