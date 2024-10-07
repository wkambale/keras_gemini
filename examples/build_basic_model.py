from keras_gemini import KerasGemini
from gemini import Gemini

# Initialize Gemini and the KerasGeminiPlugin
gemini = Gemini()
keras_plugin = KerasGemini(gemini)
gemini.add_plugin(keras_plugin)

# Example: Build a basic 3-layer sequential model
prompt = "Build a 3-layer sequential model"
response = gemini.run(prompt)

print(response)  # Should indicate the model was built successfully

# Display the model summary
if keras_plugin.model:
    keras_plugin.model.summary()
