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

accuracy_exp = []
Sn_exp = []
Sp_exp = []

# fix random seed for reproducibility
numpy.random.seed(1)
# load Gene Exp dataset
#dataset_exp = numpy.loadtxt("A:/Project/Incomplete Multiview/Data/Generated/METABRIC_gene_exp_1600.csv", delimiter=",")# Change the path to your local system
#dataset_exp = numpy.loadtxt("A:/Project/STACKED RF/Data/METABRIC_gene_exp_1980.txt", delimiter="\t")# Change the path to your local system
#dataset_exp = numpy.loadtxt("A:/Project/Incomplete Multiview/Data/Generated/metabric_exp_final.csv", delimiter=",") # Change the path to your local system
dataset_exp = numpy.loadtxt("A:/Project/Incomplete Multiview/Data/Generated/tcga_exp_final.csv", delimiter=",") # Change the path to your local system

# split into input (X) and output (Y) variables
X_exp = dataset_exp[:,0:400]
#Y_exp = dataset_exp[:,400]
Y_exp=np.loadtxt("A:/Project/Incomplete Multiview/Data/tcga5yearCutOff_incomplete.txt")
#Spliting the data set into training and testing
kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)
i=1
for train_index, test_index in kfold.split(X_exp, Y_exp):
	print(i,'th Fold *******************************')
	i=i+1
	#Spliting the data set into training and testing
	x_train_exp, x_test_exp=X_exp[train_index],X_exp[test_index]	
	y_train_exp, y_test_exp = Y_exp[train_index],Y_exp[test_index] 
	x_train_exp = numpy.expand_dims(x_train_exp, axis=2)
	x_test_exp = numpy.expand_dims(x_test_exp, axis=2)
	
	# first CNN EXP Model
	init=initializers.glorot_normal(seed=1)
	bias_init =initializers.Constant(value=0.1)
	main_input1 = Input(shape=(400,1))
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
	#plot_model(model, to_file='/home/nikhil/Desktop/Project/nik/Code/Submodels/ModelDesign/CNN Gene Exp.png')# Change the path to your local system
	def exp_decay(epoch):
		initial_lrate = 0.001
		k = 0.1
		lrate = initial_lrate * math.exp(-k*epoch)
		return lrate
	lrate = LearningRateScheduler(exp_decay)
	adams=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.compile(loss='binary_crossentropy', optimizer=adams, metrics=[sensitivity_at_specificity(0.95)])
	x_train, x_val, y_train, y_val = train_test_split(x_train_exp, y_train_exp, test_size=0.2,stratify=y_train_exp)	
	#model.fit(x_train_exp, y_train_exp, epochs=epochs,validation_split=0.20, batch_size=8,verbose=2)
	model.fit(x_train, y_train, epochs=epochs, batch_size=8,verbose=2,validation_data=(x_val,y_val),callbacks=[lrate])
	exp_scores = model.evaluate(x_test_exp, y_test_exp,verbose=2)
	print("%s: %.2f%%" % (model.metrics_names[1], exp_scores[1]*100))
	#print("%s: %.2f%%" % (model.metrics_names[2], exp_scores[2]*100))
	#print("%s: %.2f%%" % (model.metrics_names[3], exp_scores[3]*100))
	accuracy_exp.append(exp_scores[1] * 100)
	#Sn_exp.append(exp_scores[2] * 100)
	#Sp_exp.append(exp_scores[3] * 100)
	intermediate_layer_model = Model(inputs=main_input1,outputs=dropout2)	
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(accuracy_exp), numpy.std(accuracy_exp)))
#print("%.2f%% (+/- %.2f%%)" % (numpy.mean(Sn_exp), numpy.std(Sn_exp)))
#print("%.2f%% (+/- %.2f%%)" % (numpy.mean(Sp_exp), numpy.std(Sp_exp)))
#Plotting
x_train_exp, x_test_exp, y_train_exp, y_test_exp = train_test_split(X_exp, Y_exp, test_size = 0.20,stratify=Y_exp)
x_test_exp=numpy.expand_dims(x_test_exp, axis=2)
pred_cnv = model.predict(x_test_exp)
fpr, tpr, thresholds = roc_curve(y_test_exp, pred_cnv)



X_exp = numpy.expand_dims(X_exp, axis=2)
# for extracting final layer features 
#y_pred =  model.predict(X_exp)
# for extracting one layer before final layer features
y_pred = intermediate_layer_model.predict(X_exp)
#feature_target=numpy.concatenate((y_pred,X_exp[:,None]),axis=1)
with open('A:/Project/STACKED RF/Data/gatedAtnExpOutput.csv', 'w') as f:# Change the path to your local system
	for item_exp in y_pred:
		for elem in item_exp:
			f.write(str(elem)+'\t')
		f.write('\n')
        
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])


for threshold in numpy.arange(0,1,0.05):
    print('********* Threshold = ', threshold, ' ************')
    evaluate_threshold(threshold)

roc_auc = auc(fpr, tpr)
plt.plot(fpr,tpr, 'r', label = 'SiGaAtCNN-EXPR = %0.3f' %roc_auc)
plt.xlabel('1-Sp (False Positive Rate)')
plt.ylabel('Sn (True Positive Rate)')
plt.title('Receiver Operating Characteristics')
plt.legend()
plt.show()

