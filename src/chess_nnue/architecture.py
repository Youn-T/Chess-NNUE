from chess_nnue.utils import *
import cupy as cp
from operator import itemgetter


class NNUE:
    def __init__(self, INPUT: int = 40960, L1: int = 256, L2: int = 32, L3: int = 32, ALPHA_KEY : float = 0.01):
        """NNUE initialization
        
        Initialize the NNUE's parameters randomly. The gradients are zero before the first backward pass. 
        It uses He Initialization multiplying the random values by sqrt(2 / number of inputs) to maintain the variance of activations across layers and avoid vanishing gradients.

        Parameters
        ----------
        INPUT: int, optional
            The number of input features (e.g., 40960 for HalfKP architecture).
        L1: int, optional
            The number of neurons in the first hidden layer, usually 256 or 512 (default is 256).
        L2: int, optional
            The number of neurons in the second hidden layer (default is 32).
        L3: int, optional
            The number of neurons in the third hidden layer (default is 32).
        ALPHA_KEY: float, optional
            The Leaky ReLU slope for values under 0, it should be a small value (default is 0.01)
        
        Note
        ----
        - The output layer is a single neuron with a sigmoid activation function for binary classification (win/loss).
        - The weight matrix of the second layer is twice as large due to the double perspective (W1 is applied to the two input matrices, which are then concatenated)
        """
        self.params = {
            'W1': cp.random.randn(INPUT, L1) * cp.sqrt( 2.0 / INPUT ),
            'W2': cp.random.randn(L1 * 2, L2) * cp.sqrt( 2.0 / L1 ),
            'W3': cp.random.randn(L2, L3) * cp.sqrt( 2.0 / L2 ),
            'W4': cp.random.randn(L3, 1) * cp.sqrt( 2.0 / L3 ),
            'b1': cp.zeros((1, L1)),
            'b2': cp.zeros((1, L2)),
            'b3': cp.zeros((1, L3)),
            'b4': cp.zeros((1, 1))
        }
        
        self.grads = { k : cp.zeros_like(v) for k, v in self.params.items() }
        
        self.ALPHA_KEY = ALPHA_KEY
    
    def forward_pass(self, X_us, X_them):
        """Evaluation function
        
        It runs the model's forward pass based on the weights of the class.
        
        Parameters
        ----------
        X_us: cupy.ndarray
            The input matrix from player's perspective
        X_them: cupy.ndarray
            The input matrix from enemy's perspective
            
        Returns
        -------
            activations: dict[str, cupy.ndarray]
                A dictionary with the four activation matrices (A1, A2, A3, A4), useful for backward pass
                
        Note
        ----
        The evaluation is the unique value of the A4 matrix
        """
        
        Z1_us = cp.dot(X_us, self.params['W1']) + self.params['b1']
        Z1_them = cp.dot(X_them, self.params['W1']) + self.params['b1']
        
        A1_us = Utils.Leaky_Clipped_ReLU(Z1_us, self.ALPHA_KEY)
        A1_them = Utils.Leaky_Clipped_ReLU(Z1_them, self.ALPHA_KEY)
        
        A1 = cp.concatenate([ A1_us, A1_them ], axis=1)
        
        Z2 = cp.dot(A1, self.params['W2']) + self.params['b2']
        A2 = Utils.Leaky_Clipped_ReLU(Z2, self.ALPHA_KEY)
        
        Z3 = cp.dot(A2, self.params['W3']) + self.params['b3']
        A3 = Utils.Leaky_Clipped_ReLU(Z3, self.ALPHA_KEY)
        
        Z4 = cp.dot(A3, self.params['W4']) + self.params['b4']
        A4 = Utils.Sigmoid(Z4)
        
        activations = {
            'A1': A1,
            'A2': A2,
            'A3': A3,
            'A4': A4
        }
        
        zs = {
            'Z1_us': Z1_us,
            'Z1_them': Z1_them,
            'Z2': Z2,
            'Z3': Z3
        }
        
        return activations, zs
    
    def backward_pass(self, X_us, X_them, Y, activations, zs):
        """Updates the model's gradients
        
        It computes the gradient with respect to each weight and bias.
        
        Parameters
        ----------
        X_us: cupy.ndarray
            The input matrix from player's perspective
        X_them: cupy.ndarray
            The input matrix from enemy's perspective
        Y: cupy.ndarray
            The correct labels matrix
        activations: dict[str, cupy.ndarray]
            A dictionnary with the four activation matrices (A1, A2, A3, A4) computed during the forward pass
        zs: dict[str, cupy.ndarray]
            A dictionnary with the values of each layer (1,2,3) before the activation
                
        Note
        ----
        The gradients are stored in the grads variable of NNUE class
        """
        A1, A2, A3, A4 = itemgetter('A1', 'A2', 'A3', 'A4')(activations)
        Z1_us, Z1_them, Z2, Z3 = itemgetter('Z1_us', 'Z1_them', 'Z2', 'Z3')(zs) # pas besoin de Z4 car la dérivée de Sigmoid et Loss a juste besoin de la valeur activée
        m = Y.shape[0] # m * 1 avec m le nombre d'exemples dans le batch

        d4 = A4 - Y # Delta 4 : chaine de dérivées partielles -> 1x1
        dW4 = (1 / m) * ( cp.dot(A3.T, d4) ) # Sortie en gradient (dL/dW4) -> 32x1 . 1x1 -> 32x1
        db4 = (1 / m) * cp.sum(d4, axis=0, keepdims=True)
        
        d3 = cp.dot(d4, self.params['W4'].T) * Utils.Leaky_Clipped_ReLU_derivative(Z3) # Delta 3 : chaine de dérivées partielles (1x1 . 1x32) * 1x32 -> 1x32
        dW3 = (1 / m) * ( cp.dot(A2.T, d3) ) # Sortie en gradient (dL/dW3) -> 32x1 . 1x32 -> 32x32
        db3 = (1 / m) * cp.sum(d3, axis=0, keepdims=True)
        
        d2 = cp.dot(d3, self.params['W3'].T) * Utils.Leaky_Clipped_ReLU_derivative(Z2) # Delta 2 : chaine de dérivées partielles 
        dW2 = (1 / m) * ( cp.dot(A1.T, d2) ) # Sortie en gradient (dL/dW2)
        db2 = (1 / m) * cp.sum(d2, axis=0, keepdims=True)
        
        d1 = cp.dot(d2, self.params['W2'].T) # Delta 1 : chaine de dérivées partielles 
        
        half = self.params['W1'].shape[1]
        d1_us = d1[:,:half] * Utils.Leaky_Clipped_ReLU_derivative(Z1_us)
        d1_them = d1[:,half:] * Utils.Leaky_Clipped_ReLU_derivative(Z1_them)
        
        dW1_us = X_us.T.dot(d1_us)
        dW1_them = X_them.T.dot(d1_them)
        
        db1_us = cp.sum(d1_us, axis=0, keepdims=True)
        db1_them = cp.sum(d1_them, axis=0, keepdims=True)
                
        dW1 = (1 / m) * (dW1_us + dW1_them)
        db1 = (1 / m) * (db1_us + db1_them)
        
        
        self.grads = {
            'W1': dW1,
            'W2': dW2,
            'W3': dW3,
            'W4': dW4,
            'b1': db1,
            'b2': db2,
            'b3': db3,
            'b4': db4
        }