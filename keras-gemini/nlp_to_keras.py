from gemini import Gemini
from .keras_gemini import KerasGemini

def prompt_to_keras(prompt: str):
    """
    Takes a natural language prompt and returns a built Keras model.

    Args:
        prompt (str): The natural language description of the model.

    Returns:
        A Keras model if the prompt is valid and processed correctly.
    """
    gemini = Gemini()
    keras_plugin = KerasGemini(gemini)
    gemini.add_plugin(keras_plugin)
    
    # Run the prompt
    response = gemini.run(prompt)
    print(response)
    
    return keras_plugin.model
