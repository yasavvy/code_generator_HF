import torch
from langchain import HuggingFacePipeline, LLMChain, PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import subprocess
import tempfile
import unittest

#--- Модуль для работы с LLM ---
class LLM:
    def __init__(self, model_name: str = "TheBloke/Llama-2-13b-Chat-GPTQ"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
        )
        self.generation_config = GenerationConfig.from_pretrained(self.model_name)
        self.generation_config.max_new_tokens = 256
        self.generation_config.temperature = 0.1
        self.generation_config.top_p = 0.95
        self.generation_config.do_sample = True
        self.generation_config.repetition_penalty = 1.15
        self.generation_config.pad_token_id = self.tokenizer.eos_token_id
        self.text_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            generation_config=self.generation_config,
        )

    def generate_text(self, prompt: str) -> str:
        """Генерит текст с LLM"""
        return self.text_pipeline(prompt, max_length=256)[0]['generated_text']

#--- Модуль для генерации кода ---
class CodeGenerator:
    def __init__(self, llm: LLM):
        """Инициализирует генератор"""
        self.llm = llm
        self.prompt_template = PromptTemplate(
            input_variables=["user_input"],
            template="Напиши код на Python, который {user_input}.",
        )
        self.chain = LLMChain(llm=HuggingFacePipeline(pipeline=self.llm.text_pipeline, model_kwargs={"temperature": 0.2}), prompt=self.prompt_template)

    def generate_code(self, user_input: str) -> str:
        """Генерит код на основе запроса юзера"""
        return self.chain.run(user_input)

#--- Модуль для проверки ---
class CodeValidator:
    def __init__(self):
        """Инициализирует валидатор"""
        pass

    def _filter_code(self, code: str) -> list[str]:
        """Фильтрует код, оставляя строки, начинающиеся с '#' или пробела/табуляции"""
        code_lines = []
        for line in code.split("\n"):
            if line.strip().startswith("#") or line.strip().startswith((' ', '\t')):
                code_lines.append(line)
        return code_lines

    def _run_code(self, code_lines: list[str]) -> bool:
        """Запускает код и возвращает тру в случае успеха, иначе фолс"""
        with tempfile.NamedTemporaryFile('w', delete=False) as temp_file:
            temp_file.write("\n".join(code_lines))
            temp_file_path = temp_file.name

        try:
            subprocess.run(['python', temp_file_path], capture_output=True, text=True, check=True)
            print("Код успешно протестирован и работает.")
            return True
        except subprocess.CalledProcessError as e:
            self._handle_code_error(e.stderr, temp_file_path)
            return False

    def _handle_code_error(self, stderr: str, temp_file_path: str) -> None:
        """Обрабатывает ошибку при выполнении кода"""
        if "SyntaxError" in stderr:
            print(f"Ошибка компиляции: {stderr}")
        elif "ZeroDivisionError" in stderr:
            print(f"Ошибка деления на ноль: {stderr}")
        elif "RuntimeError" in stderr:
            print(f"Ошибка выполнения: {stderr}")
        elif "ModuleNotFoundError" in stderr:
            self._install_missing_libraries(stderr, temp_file_path)
        else:
            print(f"Ошибка при запуске кода: {stderr}")

    def _install_missing_libraries(self, stderr: str, temp_file_path: str) -> None:
        """Устанавливает отсутствующие библиотеки"""
        library_name = stderr.split("'")[1]
        try:
            subprocess.run(f"pip install {library_name}", capture_output=True, text=True, check=True)
            print(f"Библиотека {library_name} успешно установлена.")
            self._run_code(temp_file_path)  #Повторно запускаем код после установки
        except subprocess.CalledProcessError as e:
            print(f"Ошибка при установке библиотеки: {e.stderr}")

    def test_code(self, code: str) -> None:
        """Тестирует код"""
        code_lines = self._filter_code(code)
        if self._run_code(code_lines):
            print("Код успешно протестирован.")

#--- Модуль для взаимодействия с юзером ---
class CodeTester:
    def __init__(self, code_generator: CodeGenerator, code_validator: CodeValidator):
        """Инициализирует тестировщик"""
        self.code_generator = code_generator
        self.code_validator = code_validator

    def run(self):
        """Запускает цикл взаимодействия с юзером"""
        while True:
            user_input = input("Введите запрос на написание и тестирование кода: ")
            code = self.code_generator.generate_code(user_input)
            print(f"Сгенерированный код:\n{code}")
            self.code_validator.test_code(code)
            if input("Хотите ли вы сделать еще один запрос? (да/нет): ").lower() != "да":
                break

# --- Тесты с использованием unittest ---
class TestCodeGenerator(unittest.TestCase):
    def setUp(self):
        """Настройка тестового окружения."""
        self.llm_model = LLM()
        self.code_generator = CodeGenerator(self.llm_model)
        self.code_validator = CodeValidator()

    def test_generate_code(self):
        """Тест генерации кода"""
        user_input = "создать функцию, которая возвращает сумму двух чисел"
        code = self.code_generator.generate_code(user_input)
        self.assertTrue("def sum(a, b):" in code)
        self.assertTrue("return a + b" in code)

    def test_run_code(self):
        """тест запуска кода"""
        code_lines = ["def sum(a, b):", "return a + b", "print(sum(1, 2))"]
        self.assertTrue(self.code_validator._run_code(code_lines))

if __name__ == "__main__":
    llm_model = LLM()
    code_generator = CodeGenerator(llm_model)
    code_validator = CodeValidator()
    tester = CodeTester(code_generator, code_validator)
    tester.run()
    #запуск тестов
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
