import pytest
import cupy as cp
import numpy as np
from chess_nnue.architecture import *

@pytest.fixture
def nnue_model():
    """Initialise un modèle NNUE avec des dimensions réduites pour les tests."""
    # On utilise des dimensions très petites pour faciliter le debug et le gradient checking
    return NNUE(INPUT=10, L1=4, L2=3, L3=3)

def test_forward_pass_output_shape(nnue_model):
    """Vérifie que la sortie de la forward pass a la bonne dimension."""
    batch_size = 5
    X_us = cp.random.randn(batch_size, 10)
    X_them = cp.random.randn(batch_size, 10)
    
    activations, zs = nnue_model.forward_pass(X_us, X_them)
    
    assert "A4" in activations
    assert activations["A4"].shape == (batch_size, 1)

def test_forward_pass_activation_values(nnue_model):
    """Vérifie que les valeurs de la couche de sortie sont comprises entre 0 et 1 (Sigmoid)."""
    batch_size = 3
    X_us = cp.random.randn(batch_size, 10)
    X_them = cp.random.randn(batch_size, 10)
    
    activations, zs = nnue_model.forward_pass(X_us, X_them)
    A4 = activations["A4"]
    
    assert cp.all(A4 >= 0)
    assert cp.all(A4 <= 1)

def test_forward_pass_intermediate_layers(nnue_model):
    """Vérifie la présence et les dimensions des couches intermédiaires."""
    batch_size = 2
    X_us = cp.random.randn(batch_size, 10)
    X_them = cp.random.randn(batch_size, 10)
    
    activations, zs = nnue_model.forward_pass(X_us, X_them)
    
    # L1 concaténé (2 * L1)
    assert activations["A1"].shape == (batch_size, 8) # L1=4 -> 4*2
    assert activations["A2"].shape == (batch_size, 3)  # L2=3
    assert activations["A3"].shape == (batch_size, 3)  # L3=3

def test_forward_pass_z_values(nnue_model):
    """Vérifie la présence et les dimensions des Z1_us et Z1_them."""
    batch_size = 2
    X_us = cp.random.randn(batch_size, 10)
    X_them = cp.random.randn(batch_size, 10)
    
    activations, zs = nnue_model.forward_pass(X_us, X_them)
    
    assert "Z1_us" in zs
    assert "Z1_them" in zs
    assert "Z2" in zs
    assert "Z3" in zs
    assert zs["Z1_us"].shape == (batch_size, 4)  # L1=4
    assert zs["Z1_them"].shape == (batch_size, 4) # L1=4
    assert zs["Z2"].shape == (batch_size, 3)          # L2=3
    assert zs["Z3"].shape == (batch_size, 3)          # L3=3

def test_backward_pass_shapes(nnue_model):
    """Vérifie que les gradients ont les bonnes dimensions après la backward pass."""
    batch_size = 4
    X_us = cp.random.randn(batch_size, 10)
    X_them = cp.random.randn(batch_size, 10)
    
    Y = cp.random.randn(batch_size, 1)
    
    activations, zs = nnue_model.forward_pass(X_us, X_them)
    
    nnue_model.backward_pass(X_us, X_them, Y, activations, zs)
    
    # Vérifier les dimensions des gradients
    assert nnue_model.grads['W4'].shape == (3, 1)   # L3 -> A4
    assert nnue_model.grads['b4'].shape == (1, 1)
    assert nnue_model.grads['W3'].shape == (3, 3)   # L2 -> L3
    assert nnue_model.grads['b3'].shape == (1, 3)
    assert nnue_model.grads['W2'].shape == (8, 3)   # L1 -> L2
    assert nnue_model.grads['b2'].shape == (1, 3)
    assert nnue_model.grads['W1'].shape == (10, 4)  # Input -> L1
    assert nnue_model.grads['b1'].shape == (1, 4)

def test_numerical_gradient_checking(nnue_model):
    """Vérifie la validité des gradients par approche numérique."""
    batch_size = 2
    X_us = cp.random.randn(batch_size, 10)
    X_them = cp.random.randn(batch_size, 10)
    Y = cp.random.rand(batch_size, 1)

    # Note: L'implémentation de la backward pass d4 = A4 - Y correspond
    # au gradient de la Binary Cross Entropy par rapport à Z4 (Z avant sigmoid).
    # dL/dZ4 = A4 - Y
    
    def compute_bce_loss(model):
        activations, _ = model.forward_pass(X_us, X_them)
        A4 = activations['A4']
        # BCE Loss = - (Y * log(A4) + (1-Y) * log(1-A4))
        # Moyenne sur le batch
        loss = -cp.mean(Y * cp.log(A4 + 1e-15) + (1 - Y) * cp.log(1 - A4 + 1e-15))
        return loss

    # Calculer les gradients analytiques
    activations, zs = nnue_model.forward_pass(X_us, X_them)
    nnue_model.backward_pass(X_us, X_them, Y, activations, zs)
    analytical_grads = {k: v.copy() for k, v in nnue_model.grads.items()}

    epsilon = 1e-4
    threshold = 1e-3 # Un peu plus souple pour BCE à cause du log

    for param_name in ['W4', 'b4', 'W3', 'W2']: 
        param = nnue_model.params[param_name]
        grad_analytical = analytical_grads[param_name]
        grad_numerical = cp.zeros_like(param)

        it = np.nditer(cp.asnumpy(param), flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            original_value = param[idx].item()

            param[idx] = original_value + epsilon
            loss_plus = compute_bce_loss(nnue_model)

            param[idx] = original_value - epsilon
            loss_minus = compute_bce_loss(nnue_model)

            grad_numerical[idx] = (loss_plus - loss_minus) / (2 * epsilon)

            param[idx] = original_value
            it.iternext()

        numerator = cp.linalg.norm(grad_analytical - grad_numerical)
        denominator = cp.linalg.norm(grad_analytical) + cp.linalg.norm(grad_numerical)
        rel_error = numerator / (denominator + 1e-8)

        assert rel_error < threshold, f"Gradient check failed for {param_name}. Error: {rel_error}"

def test_backward_pass_zero_input(nnue_model):
    """Vérifie que les gradients sont cohérents avec des entrées nulles."""
    batch_size = 1
    X_us = cp.zeros((batch_size, 10))
    X_them = cp.zeros((batch_size, 10))
    Y = cp.array([[0.5]])
    
    activations, zs = nnue_model.forward_pass(X_us, X_them)
    nnue_model.backward_pass(X_us, X_them, Y, activations, zs)
    
    # dW1 devrait être nul si X est nul
    assert cp.allclose(nnue_model.grads['W1'], 0)

def test_forward_pass_symmetry(nnue_model):
    """Vérifie que le modèle réagit symétriquement si on inverse us et them (au niveau de A1)."""
    batch_size = 1
    X1 = cp.random.randn(batch_size, 10)
    X2 = cp.random.randn(batch_size, 10)
    
    # Cas 1: us=X1, them=X2
    act1, _ = nnue_model.forward_pass(X1, X2)
    # Cas 2: us=X2, them=X1
    act2, _ = nnue_model.forward_pass(X2, X1)
    
    # A1_us et A1_them sont concaténés. 
    # act1['A1'] = [ReLU(X1*W1), ReLU(X2*W1)]
    # act2['A1'] = [ReLU(X2*W1), ReLU(X1*W1)]
    # Ils ne sont pas identiques mais contiennent les mêmes blocs inversés
    half = nnue_model.params['W1'].shape[1]
    assert cp.allclose(act1['A1'][:, :half], act2['A1'][:, half:])
    assert cp.allclose(act1['A1'][:, half:], act2['A1'][:, :half])

def test_gradient_clipping_derivative_behavior(nnue_model):
    """Vérifie que les gradients s'annulent correctement pour les valeurs clippées (> 1.0)."""
    batch_size = 1
    X_us = cp.random.randn(batch_size, 10)
    X_them = cp.random.randn(batch_size, 10)
    Y = cp.array([[0.5]])
    
    # Forcer un Z2 très grand pour qu'il soit clippé (> 1.0)
    # On met un grand biais positif sur b2
    nnue_model.params['b2'] = cp.full_like(nnue_model.params['b2'], 10.0)
    
    activations, zs = nnue_model.forward_pass(X_us, X_them)
    nnue_model.backward_pass(X_us, X_them, Y, activations, zs)
    
    # Si Z2 > 1.0, Utils.Leaky_Clipped_ReLU_derivative(Z2) == 0.0
    # Donc d2 = cp.dot(d3, W3.T) * 0.0 = 0.0
    # Par conséquent, dW2 = (1/m) * dot(A1.T, d2) = 0.0
    assert cp.allclose(nnue_model.grads['W2'], 0.0), "Le gradient W2 devrait être nul si Z2 est clippé à 1.0"
    assert cp.allclose(nnue_model.grads['b2'], 0.0), "Le gradient b2 devrait être nul si Z2 est clippé à 1.0"
    assert cp.allclose(nnue_model.grads['W1'], 0.0), "Les gradients en amont (W1) devraient être nuls si Z2 est clippé"

def test_gradient_leaky_behavior(nnue_model):
    """Vérifie que la pente 'alpha' est bien appliquée pour les valeurs négatives."""
    batch_size = 1
    X_us = cp.random.randn(batch_size, 10)
    X_them = cp.random.randn(batch_size, 10)
    Y = cp.array([[0.5]])
    
    # Forcer un Z2 très négatif
    nnue_model.params['b2'] = cp.full_like(nnue_model.params['b2'], -10.0)
    nnue_model.ALPHA_KEY = 0.01
    
    activations, zs = nnue_model.forward_pass(X_us, X_them)
    nnue_model.backward_pass(X_us, X_them, Y, activations, zs)
    
    # d2 = cp.dot(d3, W3.T) * ALPHA_KEY
    # Si d3 n'est pas nul, d2 ne doit pas être nul (contrairement au clipping)
    assert not cp.allclose(nnue_model.grads['W2'], 0.0), "Le gradient ne devrait pas être nul avec Leaky ReLU pour des valeurs négatives"
