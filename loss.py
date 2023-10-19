#loss functions
import numpy as np
import math

#Categorical Cross Entropy Loss Function
class Loss:
    def calculate(self, outputs, y):
        sample_losses = self.forward(outputs, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        sample = len(y_pred)
        # in order to avoid passing to the LCCE a 0 value that would explode due to log
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # check if passed values are categorical/scalar value
        # for example:
        # y_pred_clipped = np.array ([[0.7,0.1,0.2],
        #                             [0.1,0.5,0.4],
        #                             [0.02,0.9,0.08]]
        # y_true = [0, 1, 1]
        # the code below will get the elelemts from clipped proba using y_true as indices
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(sample), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods