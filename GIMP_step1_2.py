# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 18:24:43 2020

@author: Aryan
"""

from keras.layers import Input, Dense, Average, Maximum,dot, multiply, concatenate,Activation, PReLU
from keras.models import Model
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.utils import plot_model





def shows_result(path,arr):
	with open(path, 'w') as f: 
		for item_clinical in arr:
			for elem in item_clinical:
				f.write(str(elem)+',')
			f.write('\n')


def bi_modal_attention(x, y):
    
    ''' 
    .  stands for dot product 
    *  stands for elemwise multiplication
    {} stands for concatenation
        
    m1 = x . transpose(y) ||  m2 = y . transpose(x) 
    n1 = softmax(m1)      ||  n2 = softmax(m2)
    o1 = n1 . y           ||  o2 = m2 . x
    a1 = o1 * x           ||  a2 = o2 * y
       
    return {a1, a2}
        
    '''
     
    m1 = dot([x, y], axes=[0,0])
    n1 = Activation('softmax')(m1)
    
    o1 = dot([ n1, y],axes=[0,1])
    
    a1 = multiply([o1, x])
    
    m2 = dot([y, x], axes=[0, 0])
    
    n2 = Activation('softmax')(m2)
    
    o2 = dot([n2, x],axes=[0,1])
    
    a2 = multiply([o2, y])
    return concatenate([a1, a2],axis=1)



# Metabric Common Complete dataset
clinical_common_data_path = "A:/Project/Incomplete Multiview/Data/METABRIC_clinical_1600_common.csv"
cnv_common_data_path = "A:/Project/Incomplete Multiview/Data/METABRIC_cnv_1600_common.csv"
exp_common_data_path = "A:/Project/Incomplete Multiview/Data/METABRIC_gene_exp_1600_common.csv"

# TCGA BRCA Common Complete Dataset

#clinical_common_data_path = "A:/Project/Incomplete Multiview/Data/tcga_cln_common.csv"
#cnv_common_data_path = "A:/Project/Incomplete Multiview/Data/tcga_cnv_common.csv"
#exp_common_data_path = "A:/Project/Incomplete Multiview/Data/tcga_exp_common.csv"

# Metabric Incomplete dataset
clinical_incomplete_data_path = "A:/Project/Incomplete Multiview/Data/METABRIC_clinical_1600.csv"
cnv_incomplete_data_path = "A:/Project/Incomplete Multiview/Data/METABRIC_cnv_1600.csv"
exp_incomplete_data_path = "A:/Project/Incomplete Multiview/Data/METABRIC_gene_exp_1600.csv"

# TCGA BRCA Incomplete Dataset

#clinical_incomplete_data_path = "A:/Project/Incomplete Multiview/Data/tcga_cln_incomplete.csv"
#cnv_incomplete_data_path = "A:/Project/Incomplete Multiview/Data/tcga_cnv_incomplete.csv"
#exp_incomplete_data_path = "A:/Project/Incomplete Multiview/Data/tcga_exp_incomplete.csv"


common_rep_path = 'A:/Project/Incomplete Multiview/common_rep.csv'
cln_missing_data_path = 'A:/Project/Incomplete Multiview/cln_missing_data.csv'
exp_missing_data_path = 'A:/Project/Incomplete Multiview/exp_missing_data.csv'
cnv_missing_data_path = 'A:/Project/Incomplete Multiview/cnv_missing_data.csv' # Change the path to your local system


# this is the size of our encoded representations
encoding_dim = 25  
np.random.seed(1)
epochs = 50
folds =10
'''load dataset'''
dataset_clinical = np.loadtxt(clinical_common_data_path, delimiter=",") # Change the path to your local system
dataset_exp = np.loadtxt(exp_common_data_path, delimiter=",") # Change the path to your local system
dataset_cnv = np.loadtxt(cnv_common_data_path, delimiter=",") # Change the path to your local system
#dataset_tcga5yrcutoff_common =  np.loadtxt("A:/Project/Incomplete Multiview/Data/tcga5yearCutOff_common.txt")
'''split into input (X) and output (Y) variables'''
X_clinical = dataset_clinical[:,0:25]
Y_clinical = dataset_clinical[:,25]
#X_clinical = dataset_clinical[:,0:11]
#Y_clinical = dataset_tcga5yrcutoff_common
X_exp = dataset_exp[:,0:400]
Y_exp = dataset_exp[:,400]
#Y_exp = dataset_tcga5yrcutoff_common
X_cnv = dataset_cnv[:,0:200]
Y_cnv = dataset_cnv[:,200]
#Y_cnv = dataset_tcga5yrcutoff_common


dataset_clinical_incomplete = np.loadtxt(clinical_incomplete_data_path, delimiter=",") # Change the path to your local system
dataset_exp_incomplete = np.loadtxt(exp_incomplete_data_path, delimiter=",") # Change the path to your local system
dataset_cnv_incomplete = np.loadtxt(cnv_incomplete_data_path, delimiter=",") # Change the path to your local system
dataset_tcga5yrcutoff_incomplete =  np.loadtxt("A:/Project/Incomplete Multiview/Data/tcga5yearCutOff_incomplete.txt")
'''split into input (X) and output (Y) variables'''
X_clinical_incomplete = dataset_clinical_incomplete[:,0:25]
Y_clinical_incomplete = dataset_clinical_incomplete[:,25]
#X_clinical_incomplete = dataset_clinical_incomplete[:,0:11]
#Y_clinical_incomplete = dataset_tcga5yrcutoff_incomplete
X_exp_incomplete = dataset_exp_incomplete[:,0:400]
Y_exp_incomplete = dataset_exp_incomplete[:,400]
#Y_exp_incomplete = dataset_tcga5yrcutoff_incomplete
X_cnv_incomplete = dataset_cnv_incomplete[:,0:200]
Y_cnv_incomplete = dataset_cnv_incomplete[:,200]
#Y_cnv_incomplete = dataset_tcga5yrcutoff_incomplete
# for metabric
valid = np.ones((1980, 1))
fake = np.zeros((1980, 1))
# for tcga
#valid = np.ones((1081, 1))
#fake = np.zeros((1081, 1))
trainable = True
trainable_ = False

i=1
kfold = StratifiedKFold(n_splits=folds, shuffle=trainable, random_state=1)
for train_index, test_index in kfold.split(X_clinical, Y_clinical):
    print(i,'th Fold Running')
    i=i+1
    #  No missing views in this set of data
    X_train_cln, X_test_cln=X_clinical[train_index],X_clinical[test_index]	
    y_train_cln, y_test_cln=Y_clinical[train_index],Y_clinical[test_index]
    X_train_exp, X_test_exp=X_exp[train_index],X_exp[test_index]
    y_train_exp, y_test_exp=Y_exp[train_index],Y_exp[test_index]
    X_train_cnv, X_test_cnv=X_cnv[train_index],X_cnv[test_index]
    y_train_cnv, y_test_cnv=Y_cnv[train_index],Y_cnv[test_index]
    
    #missing views are added in this set of data
    X_train_cln_incomplete, X_test_cln_incomplete=X_clinical_incomplete[train_index],X_clinical_incomplete[test_index]	
    y_train_cln_incomplete, y_test_cln_incomplete=Y_clinical_incomplete[train_index],Y_clinical_incomplete[test_index]
    X_train_exp_incomplete, X_test_exp_incomplete=X_exp_incomplete[train_index],X_exp_incomplete[test_index]
    y_train_exp_incomplete, y_test_exp_incomplete=Y_exp_incomplete[train_index],Y_exp_incomplete[test_index]
    X_train_cnv_incomplete, X_test_cnv_incomplete=X_cnv_incomplete[train_index],X_cnv_incomplete[test_index]
    y_train_cnv_incomplete, y_test_cnv_incomplete=Y_cnv_incomplete[train_index],Y_cnv_incomplete[test_index]
    valid_train_incomplete,valid_test_incomplete=valid[train_index],valid[test_index]
    fake_train_incomplete,fake_test_incomplete=fake[train_index],fake[test_index]
    
    
    # this is our input placeholder
    input_cln = Input(shape=(25,))
    #input_cln = Input(shape=(11,))
    input_exp = Input(shape=(400,))
    input_cnv = Input(shape=(200,))
    # "encoded" is the encoded representation of the input

    z1 = Dense(encoding_dim, activation='tanh',trainable = trainable)(input_cln)

    z2 = Dense(200, activation='tanh',trainable = trainable)(input_exp)
    z2 = Dense(100, activation='tanh',trainable = trainable)(z2)
    z2 = Dense(50, activation='tanh',trainable = trainable)(z2)
    z2 = Dense(encoding_dim, activation='tanh',trainable = trainable)(z2)

    z3 = Dense(100, activation='tanh',trainable = trainable)(input_cnv)
    z3 = Dense(50, activation='tanh',trainable = trainable)(z3)
    z3 = Dense(encoding_dim, activation='tanh',trainable = trainable)(z3)
    
    z1_z2_atn = bi_modal_attention(z1,z2)
    z2_z3_atn = bi_modal_attention(z2,z3)
    z3_z1_atn = bi_modal_attention(z3,z1)

    z = Average(name='common_represenatation',trainable = trainable)([z1_z2_atn,z2_z3_atn,z3_z1_atn])





    # "decoded" is the lossy reconstruction of the input
    g1_z2 = Dense(25,trainable = trainable)(z2)
    g1_z2 = PReLU(alpha_initializer="zeros",alpha_regularizer=None,alpha_constraint=None,shared_axes=None)(g1_z2)
    g1_z3 = Dense(25,trainable = trainable)(z3)
    g1_z3 = PReLU(alpha_initializer="zeros",alpha_regularizer=None,alpha_constraint=None,shared_axes=None)(g1_z3)
    g1_z = Dense(25,trainable = trainable)(z)
    g1_z = PReLU(alpha_initializer="zeros",alpha_regularizer=None,alpha_constraint=None,shared_axes=None)(g1_z)
    
    
    cln_decoded = Maximum(name='cln_Reconstructed',trainable = trainable)([g1_z2,g1_z3,g1_z])


    g2_z1 = Dense(50, activation='tanh',trainable = trainable)(z1)
    g2_z1 = Dense(100, activation='tanh',trainable = trainable)(g2_z1)
    g2_z1 = Dense(200, activation='tanh',trainable = trainable)(g2_z1)
    g2_z1 = Dense(400,trainable = trainable)(g2_z1)
    g2_z1 = PReLU(alpha_initializer="zeros",alpha_regularizer=None,alpha_constraint=None,shared_axes=None)(g2_z1)
    
    g2_z3 = Dense(50, activation='tanh',trainable = trainable)(z3)
    g2_z3 = Dense(100, activation='tanh',trainable = trainable)(g2_z3)
    g2_z3 = Dense(200, activation='tanh',trainable = trainable)(g2_z3)
    g2_z3 = Dense(400,trainable = trainable)(g2_z3)
    g2_z3 = PReLU(alpha_initializer="zeros",alpha_regularizer=None,alpha_constraint=None,shared_axes=None)(g2_z3)

    g2_z= Dense(50, activation='tanh',trainable = trainable)(z)
    g2_z= Dense(100, activation='tanh',trainable = trainable)(g2_z)
    g2_z= Dense(200, activation='tanh',trainable = trainable)(g2_z)
    g2_z= Dense(400,trainable = trainable)(g2_z)
    g2_z = PReLU(alpha_initializer="zeros",alpha_regularizer=None,alpha_constraint=None,shared_axes=None)(g2_z)
    exp_decoded = Maximum(name='exp_Reconstructed',trainable = trainable)([g2_z1,g2_z3,g2_z])


    g3_z1 = Dense(50, activation='tanh',trainable = trainable)(z1)
    g3_z1 = Dense(100, activation='tanh',trainable = trainable)(g3_z1)
    g3_z1 = Dense(200,trainable = trainable)(g3_z1)
    g3_z1 = PReLU(alpha_initializer="zeros",alpha_regularizer=None,alpha_constraint=None,shared_axes=None)(g3_z1)
    
    g3_z2 = Dense(50, activation='tanh',trainable = trainable)(z2)
    g3_z2 = Dense(100, activation='tanh',trainable = trainable)(g3_z2)
    g3_z2 = Dense(200,trainable = trainable)(g3_z2)
    g3_z2 = PReLU(alpha_initializer="zeros",alpha_regularizer=None,alpha_constraint=None,shared_axes=None)(g3_z2)


    g3_z= Dense(50, activation='tanh',trainable = trainable)(z)
    g3_z= Dense(100, activation='tanh',trainable = trainable)(g3_z)
    g3_z= Dense(200,trainable = trainable)(g3_z)
    g3_z = PReLU(alpha_initializer="zeros",alpha_regularizer=None,alpha_constraint=None,shared_axes=None)(g3_z)
    cnv_decoded = Maximum(name='cnv_Reconstructed',trainable = trainable)([g3_z1,g3_z2,g3_z])



    discriminator_cln = Dense(10,trainable = trainable)(cln_decoded)
    discriminator_cln = Dense(1,activation='sigmoid',trainable = trainable)(discriminator_cln)
    
    discriminator_exp= Dense(200,trainable = trainable)(exp_decoded)
    discriminator_exp= Dense(100,trainable = trainable)(discriminator_exp)
    discriminator_exp= Dense(50,trainable = trainable)(discriminator_exp)
    discriminator_exp= Dense(1,activation='sigmoid',trainable = trainable)(discriminator_exp)
    
    
    discriminator_cnv= Dense(100,trainable = trainable)(cnv_decoded)
    discriminator_cnv= Dense(50,trainable = trainable)(cnv_decoded)
    discriminator_cnv= Dense(1,activation='sigmoid',trainable = trainable)(cnv_decoded)
    

    # this model maps an input to its reconstruction
    autoencoder = Model(inputs=[input_cln,input_exp,input_cnv],outputs= [cln_decoded,exp_decoded,cnv_decoded])
    autoencoder.compile(optimizer='Adam', loss='mse')


    plot_model(autoencoder, to_file='autoencoder.png')
    # fix random seed for reproducibility

    autoencoder.fit([X_train_cln,X_train_exp,X_train_cnv], [X_train_cln,X_train_exp,X_train_cnv],
                epochs=epochs,
                shuffle=True,
                batch_size=10,
                validation_data=([X_test_cln,X_test_exp,X_test_cnv], [X_test_cln,X_test_exp,X_test_cnv]))
    
    
    
    
    
    #Train decoder and discriminator
    z1.trainable = trainable_
    z2.trainable = trainable_
    z3.trainable = trainable_
    z.trainable = trainable_
    z1_z2_atn.trainable = trainable_
    z2_z3_atn.trainable = trainable_
    z3_z1_atn.trainable = trainable_
    
    discriminator_real=Model(inputs=[input_cln,input_exp,input_cnv],outputs= [discriminator_cln,discriminator_exp,discriminator_cnv])
    discriminator_real.compile(optimizer='Adam', loss='binary_crossentropy')
    plot_model(discriminator_real, to_file='discriminator_real.png')
    discriminator_real.fit([X_train_cln_incomplete,X_train_exp_incomplete,X_train_cnv_incomplete], [valid_train_incomplete,valid_train_incomplete,valid_train_incomplete],
                epochs=epochs,
                shuffle=True,
                batch_size=10,
                validation_data=([X_test_cln_incomplete,X_test_exp_incomplete,X_test_cnv_incomplete], [valid_test_incomplete,valid_test_incomplete,valid_test_incomplete]))
    
    
    
    common_rep_generator = Model(inputs=[input_cln,input_exp,input_cnv],outputs=z)
    cln_missing_data_generator = Model(inputs=[input_cln,input_exp,input_cnv],outputs=cln_decoded)
    exp_missing_data_generator = Model(inputs=[input_cln,input_exp,input_cnv],outputs=exp_decoded)
    cnv_missing_data_generator = Model(inputs=[input_cln,input_exp,input_cnv],outputs=cnv_decoded)

    # encode and decode some digits
    # note that we take them from the *test* set


common_representation = common_rep_generator.predict([X_clinical_incomplete,X_exp_incomplete,X_cnv_incomplete])
cln_missing_data = cln_missing_data_generator.predict([X_clinical_incomplete,X_exp_incomplete,X_cnv_incomplete])
exp_missing_data = exp_missing_data_generator.predict([X_clinical_incomplete,X_exp_incomplete,X_cnv_incomplete])
cnv_missing_data = cnv_missing_data_generator.predict([X_clinical_incomplete,X_exp_incomplete,X_cnv_incomplete])
shows_result(common_rep_path,common_representation)
shows_result(cln_missing_data_path,cln_missing_data)
shows_result(exp_missing_data_path,exp_missing_data)
shows_result(cnv_missing_data_path,cnv_missing_data)