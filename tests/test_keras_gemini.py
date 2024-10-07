import pytest
from keras_gemini import KerasGemini, prompt_to_keras
from gemini import Gemini

# Test cases for the keras_gemini package

@pytest.fixture
def gemini_instance():
    """Creates a Gemini instance with the KerasGeminiPlugin for testing."""
    gemini = Gemini()
    keras_plugin = KerasGemini(gemini)
    gemini.add_plugin(keras_plugin)
    return keras_plugin


def test_plugin_initialization(gemini_instance):
    """Test if the KerasGeminiPlugin is initialized correctly."""
    assert gemini_instance.model is None, "Model should be None on initialization."


def test_build_keras_model_sequential(gemini_instance):
    """Test if the plugin correctly builds a 3-layer sequential model."""
    prompt = "Build a 3-layer sequential model"
    response = gemini_instance.gemini.run(prompt)
    assert "Model built successfully." in response, "Model was not built successfully."

    # Check if model is correctly created
    model = gemini_instance.model
    assert model is not None, "Model should not be None after building."
    assert len(model.layers) == 4, "Model should have 4 layers (3 hidden + 1 output)."
    assert model.layers[-1].activation.__name__ == "softmax", "Output layer should have softmax activation."


def test_invalid_model_type(gemini_instance):
    """Test if the plugin handles invalid model types correctly."""
    prompt = "Build a 3-layer convolutional model"
    response = gemini_instance.gemini.run(prompt)
    assert "Unsupported model type" in response, "Invalid model type was not handled correctly."


def test_prompt_to_keras_function():
    """Test the prompt_to_keras helper function to build a model."""
    model = prompt_to_keras("Build a 2-layer sequential model")
    assert model is not None, "Model should be created from prompt."
    assert len(model.layers) == 3, "Model should have 3 layers (2 hidden + 1 output)."


def test_model_output_shape():
    """Test if the output layer of the created model has the correct number of units."""
    model = prompt_to_keras("Build a 5-layer sequential model")
    output_layer = model.layers[-1]
    assert output_layer.units == 10, "Output layer should have 10 units."


def test_model_compile_settings():
    """Test if the model is compiled with correct optimizer, loss function, and metrics."""
    model = prompt_to_keras("Build a 3-layer sequential model")
    assert model.optimizer._name == "Adam", "Optimizer should be Adam."
    assert model.loss == "sparse_categorical_crossentropy", "Loss function should be sparse_categorical_crossentropy."
    assert "accuracy" in model.metrics, "Metrics should include accuracy."
