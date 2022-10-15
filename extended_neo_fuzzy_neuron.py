import skfuzzy as fuzz
import numpy as np
from numpy.linalg import norm

class ENFN():
    def __init__(self, args):
        self.membership_functions = args.membership_functions
        self.membership_functions = args.membership_functions
        self.history_length = args.history_length
        self.inference_order = args.inference_order
        self.learning_rate = args.learning_rate
        self.membership_function_shape = args.membership_function_shape
        self.w = np.zeros(self.history_length * self.membership_functions * (self.inference_order + 1))
        

    def train(self, series):
        # make history data
        data = []
        for indx in range(series.shape[0] - self.history_length + 1):
            data.append(series[indx : indx + self.history_length])
        ## w - weight matrix
        mu = np.zeros(self.history_length * self.membership_functions * (self.inference_order + 1))
        sim = np.zeros(len(data))
        e = np.zeros(len(data))
        r = 0
        #mean of membership_functions
        for k, input in enumerate(data): 
            for i, series in enumerate(input):
                for l in range(self.membership_functions):
                    muValue = self.membership_function_value(np.array([input[i]]), l)
                    for j in range(self.inference_order + 1):
                        mu[(i) * self.membership_functions + (l) * (self.inference_order + 1) + j] = muValue * (np.power(input[i],j))     
                
            if (k==len(data)-1):
                break
            sim[k] = np.matmul(self.w, mu)
            e[k] = data[k+1][-1] - sim[k]
            ## Training
            r = r * self.learning_rate + (np.power(norm(mu),2))
            self.w = self.w + e[k] * mu / r
        return sim,e,np.array(data)[1:,-1]

    def test(self, series):
        data = []
        for indx in range(series.shape[0] - self.history_length + 1):
            data.append(series[indx : indx + self.history_length])

        ## w - weight matrix
        mu = np.zeros(self.history_length * self.membership_functions * (self.inference_order + 1))
        sim = np.zeros(len(data))
        e = np.zeros(len(data))

        #mean of membership_functions
        for k, input in enumerate(data): 
            for i, series in enumerate(input):
                for l in range(self.membership_functions):
                    muValue = self.membership_function_value(np.array([input[i]]), l)
                    for j in range(self.inference_order + 1):
                        mu[(i) * self.membership_functions + (l) * (self.inference_order + 1) + j] = muValue * (np.power(input[i],j))     
            
            if (k==len(data)-1):
                    break        
            sim[k] = np.matmul(self.w, mu)
            e[k] = data[k+1][-1] - sim[k]
        return sim,e,np.array(data)[1:,-1]
    
    def membership_function_value(self, input, l):
        if self.membership_function_shape=="triangular":
            means = np.zeros(self.membership_functions+2)
            means[1:-1] = np.linspace(0, 1, num=self.membership_functions)
            means[0] = means[1]
            means[-1] = means[-2]
            return fuzz.trimf(input, [means[l], means[l+1], means[l+2]])
        elif self.membership_function_shape=="trapezoid": #Trapezoidal Membership Function
            means = np.zeros(self.membership_functions*2+2)
            means[1:-1] = np.linspace(0, 1, num=self.membership_functions*2)
            means[0] = means[1]
            means[-1] = means[-2]
            return fuzz.trapmf(input, [means[2*l], means[2*l+1], means[2*l+2], means[2*l+3]])
        elif self.membership_function_shape=="guassian": #Gaussian Membership Function
            means = np.linspace(0, 1, num=self.membership_functions)
            return fuzz.gaussmf(input,  means[l], 0.5/self.membership_functions)
        elif self.membership_function_shape=="bell": #Generalized Bell membership function
            means = np.linspace(0, 1, num=self.membership_functions)
            return fuzz.gbellmf(input, 0.1, 2, means[l])
        elif self.membership_function_shape=="sigmoid":
            means = np.linspace(0, 1, num=self.membership_functions)
            return fuzz.sigmf(input, means[l],50)
