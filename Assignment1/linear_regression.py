import numpy as np

class LinearRegression():
    
    def __init__(self, learning_rate=0.001, epochs=1000):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # TODO: Implement
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).reshape(-1,)

        m, n = X.shape
        self.weights = np.zeros(n, dtype=float)
        self.bias = 0.0

        for _ in range(self.epochs):
            y_pred = X.dot(self.weights) + self.bias
            error = y_pred - y

            d_w = (1/m) * X.T.dot(error)
            d_b = error.mean()

            self.weights -= self.learning_rate * d_w
            self.bias -= self.learning_rate * d_b

        
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        # TODO: Implement
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X.dot(self.weights) + self.bias



