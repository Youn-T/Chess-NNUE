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
    
    activations = nnue_model.forward_pass(X_us, X_them)
    
    assert "A4" in activations
    assert activations["A4"].shape == (batch_size, 1)

def test_forward_pass_activation_values(nnue_model):
    """Vérifie que les valeurs de la couche de sortie sont comprises entre 0 et 1 (Sigmoid)."""
    batch_size = 3
    X_us = cp.random.randn(batch_size, 100)
    X_them = cp.random.randn(batch_size, 100)
    
    activations = nnue_model.forward_pass(X_us, X_them)
    A4 = activations["A4"]
    
    assert cp.all(A4 >= 0)
    assert cp.all(A4 <= 1)

def test_forward_pass_intermediate_layers(nnue_model):
    """Vérifie la présence et les dimensions des couches intermédiaires."""
    batch_size = 2
    X_us = cp.random.randn(batch_size, 100)
    X_them = cp.random.randn(batch_size, 100)
    
    activations = nnue_model.forward_pass(X_us, X_them)
    
    # L1 concaténé (2 * L1)
    assert activations["A1"].shape == (batch_size, 32) # L1=16 -> 16*2
    assert activations["A2"].shape == (batch_size, 8)  # L2=8
    assert activations["A3"].shape == (batch_size, 8)  # L3=8
