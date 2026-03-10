import pytest
import cupy as cp
from chess_nnue.architecture import *

@pytest.fixture
def nnue_model():
    """Initialise un modèle NNUE avec des dimensions réduites pour les tests."""
    return NNUE(INPUT=100, L1=16, L2=8, L3=8)

def test_forward_pass_output_shape(nnue_model):
    """Vérifie que la sortie de la forward pass a la bonne dimension."""
    batch_size = 5
    X_us = cp.random.randn(batch_size, 100)
    X_them = cp.random.randn(batch_size, 100)
    
    activations, zs = nnue_model.forward_pass(X_us, X_them)
    
    assert "A4" in activations
    assert activations["A4"].shape == (batch_size, 1)

def test_forward_pass_activation_values(nnue_model):
    """Vérifie que les valeurs de la couche de sortie sont comprises entre 0 et 1 (Sigmoid)."""
    batch_size = 3
    X_us = cp.random.randn(batch_size, 100)
    X_them = cp.random.randn(batch_size, 100)
    
    activations, zs = nnue_model.forward_pass(X_us, X_them)
    A4 = activations["A4"]
    
    assert cp.all(A4 >= 0)
    assert cp.all(A4 <= 1)

def test_forward_pass_intermediate_layers(nnue_model):
    """Vérifie la présence et les dimensions des couches intermédiaires."""
    batch_size = 2
    X_us = cp.random.randn(batch_size, 100)
    X_them = cp.random.randn(batch_size, 100)
    
    activations, zs = nnue_model.forward_pass(X_us, X_them)
    
    # L1 concaténé (2 * L1)
    assert activations["A1"].shape == (batch_size, 32) # L1=16 -> 16*2
    assert activations["A2"].shape == (batch_size, 8)  # L2=8
    assert activations["A3"].shape == (batch_size, 8)  # L3=8

def test_forward_pass_z_values(nnue_model):
    """Vérifie la présence et les dimensions des Z1_us et Z1_them."""
    batch_size = 2
    X_us = cp.random.randn(batch_size, 100)
    X_them = cp.random.randn(batch_size, 100)
    
    activations, zs = nnue_model.forward_pass(X_us, X_them)
    
    assert "Z1_us" in zs
    assert "Z1_them" in zs
    assert zs["Z1_us"].shape == (batch_size, 16)  # L1=16
    assert zs["Z1_them"].shape == (batch_size, 16) # L1=16

def test_backward_pass_shapes(nnue_model):
    """Vérifie que les gradients ont les bonnes dimensions après la backward pass."""
    batch_size = 4
    X_us = cp.random.randn(batch_size, 100)
    X_them = cp.random.randn(batch_size, 100)
    
    Y = cp.random.randn(batch_size, 1)
    
    activations, zs = nnue_model.forward_pass(X_us, X_them)
    
    # Simuler une perte et calculer le gradient de la sortie
    loss_gradient = cp.random.randn(batch_size, 1)  # Gradient de la perte par rapport à A4
    
    nnue_model.backward_pass(X_us, X_them, Y, activations, zs)
    
    # Vérifier les dimensions des gradients
    assert nnue_model.grads['W4'].shape == (8, 1)   # L3 -> A4
    assert nnue_model.grads['b4'].shape == (1,)     # Biais de A4
    assert nnue_model.grads['W3'].shape == (8, 8)   # L2 -> L3
    assert nnue_model.grads['b3'].shape == (8,)     # Biais de L3
    assert nnue_model.grads['W2'].shape == (16, 8)  # L1 -> L2
    assert nnue_model.grads['b2'].shape == (8,)     # Biais de L2
    assert nnue_model.grads['W1'].shape == (200, 32) # Input concaténé -> L1
    assert nnue_model.grads['b1'].shape == (16,)     # Biais de L1