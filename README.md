**Code Tester: Generate, Test, and Debug Python Code with an LLM**

This project allows you to generate Python code using a large language model (LLM), test the generated code, and (partially) debug it.

***Features***

Code Generation: Generate Python code based on natural language prompts using the Llama-2-13b-Chat-GPTQ LLM.

Code Testing: Automatically test the generated code by running it and checking for errors.

Error Handling: Handles common errors, such as syntax errors, runtime errors, and missing libraries.

User-Friendly Interface: Interact with the code tester through a simple console-based interface.

***Requirements***

To use the code tester, you need to install the packages required by using pip:
pip install -r requirements.txt

***Usage***

Run the script. Enter a prompt by typing a natural language prompt describing the Python code you want to generate. For example:
"Write a function that takes two numbers and returns their sum."

The code tester will generate Python code based on your prompt.
The code tester will automatically test the generated code and print the results.
If the code has errors, the code tester will try to identify the error and provide suggestions for fixing it.

***Note***

You need to have an internet connection to use the code tester, as it relies on the Llama-2-13b-Chat-GPTQ LLM, which is hosted online.

***License***
This project is licensed under the MIT License. See the LICENSE file for more information.
