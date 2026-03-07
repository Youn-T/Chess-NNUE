import cupy as cp

class Utils:
    """
    Utility functions for the neuronal network.
    """
    
    # ============================================
    # ----------- ACTIVATION FUNCTIONS -----------
    # ============================================
    
    # ----------- Leaky_Clipped_ReLU -----------
    @staticmethod
    def Leaky_Clipped_ReLU(x, alpha=0.01):
        """Activation function that combines Leaky ReLU and Clipping.

        For values under 0 a little slope (alpha), between 0 and 1 is applied, between 0 and 1 the function is linear, and above 1 it returns 1.

        Parameters
        ----------
        x: cupy.ndarray
            Input values.
        alpha: float, optional
            Slope for negative values.

        Returns
        -------
            result: cupy.ndarray
                Array of activated values.
        """
        return cp.where(x > 1.0, 1.0, cp.where(x > 0, x, x * alpha))

    @staticmethod
    def Leaky_Clipped_ReLU_derivative(x, alpha=0.01):
        """Derivative of Leaky_Clipped_ReLU.

        Parameters
        ----------
        x: cupy.ndarray
            Input values.
        alpha: float, optional
            Slope for negative values.

        Returns
        -------
            result: cupy.ndarray
                Gradient of the function with respect to x.
        
        See Also
        --------
            Leaky_Clipped_ReLU: The activation function for which this is the derivative.
        """
        grad = cp.where((x >= 0) & (x <= 1.0), 1.0, 0.0)
        grad = cp.where(x < 0, alpha, grad)
        return grad

    # ----------- Leaky_ReLU -----------
    @staticmethod
    def Leaky_ReLU(x, alpha=0.01):
        """Activation function that uses Leaky ReLU.

        For values under 0 a little slope (alpha), between 0 and 1 is applied, after 0 the function is linear.

        Parameters
        ----------
        x: cupy.ndarray
            Input values.
        alpha: float, optional
            Slope for negative values.

        Returns
        -------
            result: cupy.ndarray
                Array of activated values.
        """
        return cp.where(x > 0, x, x * alpha)

    @staticmethod
    def Leaky_ReLU_derivative(x, alpha=0.01):
        """Derivative of Leaky_ReLU.

        Parameters
        ----------
        x: cupy.ndarray
            Input values.
        alpha: float, optional
            Slope for negative values.

        Returns
        -------
            result: cupy.ndarray
                Gradient of the function with respect to x.
        
        See Also
        --------
            Leaky_ReLU: The activation function for which this is the derivative.
        """
        return cp.where(x > 0, 1, alpha)

    # ----------- Sigmoid -----------
    @staticmethod
    def Sigmoid(x):
        """Sigmoid function.

        S-shaped function that converts any number into probability.
        
        Parameters
        ----------
        x: cupy.ndarray
            Input values.

        Returns
        -------
            result: cupy.ndarray
                Array of activated values.
                
        Notes
        -----
            The input is clipped to avoid overflow in the exponential function.
        """
        # Clip pour éviter les overflows d'exponentielle
        x = cp.clip(x, -500, 500)
        return 1 / (1 + cp.exp(-x))


    # @staticmethod
    # def clip_gradients(grads, max_norm=1.0):
    #     for i in range(len(grads)):
    #         cp.clip(grads[i], -max_norm, max_norm, out=grads[i])
    #     return grads