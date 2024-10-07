from keras_gemini import KerasGemini
from gemini import Gemini

# Initialize Gemini and the KerasGeminiPlugin
gemini = Gemini()
keras_plugin = KerasGemini(gemini)
gemini.add_plugin(keras_plugin)

# Interactive loop for user input
print("Welcome to the Keras-Gemini Model Builder!")
print("Type a prompt like 'Build a 3-layer sequential model', or 'exit' to quit.")

while True:
    prompt = input("\nEnter a prompt: ")
    
    if prompt.lower() == "exit":
        print("Exiting...")
        break

    response = gemini.run(prompt)
    print(response)

    if keras_plugin.model:
        keras_plugin.model.summary()
