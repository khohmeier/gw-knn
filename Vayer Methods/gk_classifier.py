#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:02:05 2018

@author: Titouan
"""
from sklearn.svm import SVC
from grakel import GraphKernel
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

class BadInputError(Exception):
    pass
class GKNotDefinedError(Exception):
    pass

class GK_classifier():
    def __init__(self,C=1,precomputed=False,normalize=True,**params):
        self.C=C
        self.gk=self._create_gk_from_dict(**params)
        if self.gk is not None:
            self.gk.set_params(normalize=normalize)
        self.svc=SVC(kernel='precomputed', C=self.C)
        self.precomputed=precomputed
        self.normalize=normalize

    def __eq__(self, another):

        bool1=hasattr(another, 'gk')
        bool2=hasattr(another, 'svc')
        bool3=hasattr(another, 'precomputed')
        bool4=hasattr(another, 'normalize')
        bool5=self._check_eq_kernel_params(self.gk.kernel,another.gk.kernel)
        bool6=self.C==another.C
        bool7=self.precomputed==another.precomputed
        bool8=self.normalize==another.normalize

        return np.all([bool1,bool2,bool3,bool4,bool5,bool6,bool7,bool8])

    def __hash__(self):
        return hash(repr(self.get_params()))

    def fit(self, X, y):
        # Ensure y is properly shaped
        y = np.array(y).ravel()  # Flatten to 1D array

        if self.precomputed:
            self.svc.fit(X, y)
        else:
            K = self.gk.fit_transform(X)
            self.svc.fit(K, y)
        return self

    def predict(self, X):
        if self.precomputed:
            pred = self.svc.predict(X)
        else:
            K = self.gk.transform(X)
            pred = self.svc.predict(K)

        # Ensure prediction is properly shaped
        return np.array(pred).ravel()

    def _create_gk_from_dict(self,**params): 
        ''' params={'kernel_params':[{'name':'shortest_path','with_labels':True}]}
        clf=GK_classifier(C=2,normalize=False,**params)''' 

        if 'kernel_params' not in params:
            print('Warning : no GK defined because kernel_params not in params')
            print('params : ',params)
            print('Available keys in params:', list(params.keys()))
            return None
        elif not isinstance(params['kernel_params'],list):
            raise BadInputError('Input[kernel_params] should be a list')
        else:
            #print('Creating GraphKernel with params:', params['kernel_params'])
            return GraphKernel(params['kernel_params'])

    def set_one_param(self,dicto,key):
        if key in dicto:
            setattr(self, key, dicto[key])

    def set_params(self, **parameters):
        #print(f'GK_classifier.set_params called with: {parameters}')
        self.set_one_param(parameters,"C")
        self.set_one_param(parameters,"normalize")
        self.gk=self._create_gk_from_dict(**parameters)
        if self.gk is not None:
            self.gk.set_params(normalize=self.normalize)
        self.svc=SVC(kernel='precomputed', C=self.C)
        return self

    def get_params(self, deep=True):
        if self.gk is None:
            return {"C":self.C,"kernel":None,"normalize":None}
        else:
            return {"C":self.C,"kernel":self.gk.kernel,"normalize":self.gk.normalize}

    def get_kernel_params(self, deep=True):
        if self.gk is None:
            return {"kernel_params":None ,"normalize":None}
        else:
            return {"kernel_params":self.gk.kernel #is a list
                    ,"normalize":self.gk.normalize}

    def _check_eq_kernel_params(self,param1,param2):
        if len(param1)!=len(param2):
            return False
        else:
            allTrue=[]
            for i in range(len(param1)):
                allTrue.append(param1[i] in param2) #may not be in same order
            return np.all(allTrue)

    def evaluate_comprehensive(self, X, y_true):
        """ Comprehensive evaluation with accuracy, F1 scores, and by-class metrics
        Parameters
        ----------
        X : array of graphs or precomputed kernel matrix
        y_true : true classes
        Returns
        -------
        dict : comprehensive evaluation metrics
        """
        y_pred = self.predict(X)

        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # F1 scores
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')

        # By-class metrics
        class_report = classification_report(y_true, y_pred, output_dict=True)

        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': class_report,
            'predictions': y_pred
        }