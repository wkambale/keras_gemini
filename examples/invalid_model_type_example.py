from keras_gemini import KerasGemini
from gemini import Gemini

# Initialize Gemini and the KerasGeminiPlugin
gemini = Gemini()
keras_plugin = KerasGemini(gemini)
gemini.add_plugin(keras_plugin)

# Example: Attempt to build a convolutional model (unsupported)
prompt = "Build a 4-layer convolutional network"
response = gemini.run(prompt)

print(response)  # Should indicate the model type is unsupported

# The model should remain None
assert keras_plugin.model is None, "Model should not be created for an unsupported type."
