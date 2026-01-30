"""Local LLM integration for the Herbolaria RAG application.

This module provides a drop-in replacement for external LLM APIs,
allowing the Streamlit app to use the finetuned local model.

Usage:
    from training.scripts.local_llm import LocalHerbolariLLM

    llm = LocalHerbolariLLM("./models/herbolaria-dasd-4b-merged")
    response = llm.generate("¿Cuáles son los usos de la sábila?")
"""

from pathlib import Path
from typing import Generator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread


class LocalHerbolariLLM:
    """Local LLM wrapper for the finetuned herbolaria model."""

    def __init__(
        self,
        model_path: str | Path = "./models/herbolaria-dasd-4b-merged",
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        load_in_4bit: bool = False,
    ):
        """Initialize the local LLM.

        Args:
            model_path: Path to the merged model.
            device_map: Device mapping strategy ("auto", "cuda:0", etc.)
            torch_dtype: Model weight data type.
            load_in_4bit: Whether to load in 4-bit quantization for lower memory.
        """
        self.model_path = Path(model_path)

        # Map dtype string to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        self.tokenizer = None
        self.model = None
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit

        self.system_prompt = """Eres un asistente experto en medicina tradicional mexicana. Tu conocimiento abarca:
- Plantas medicinales mexicanas y sus usos terapéuticos
- Síndromes de filiación cultural (susto, mal de ojo, empacho, etc.)
- Prácticas y rituales de curación tradicionales
- Conocimientos de los pueblos indígenas de México
- Historia de la medicina tradicional desde la época prehispánica

Respondes en español con precisión y respeto cultural. Cuando es relevante, mencionas:
- Nombres en lenguas indígenas (náhuatl, maya, etc.)
- Usos regionales específicos
- Precauciones y contraindicaciones
- Fuentes históricas cuando están disponibles

Siempre aclaras que la información es con fines educativos y culturales, no como consejo médico."""

    def load(self) -> None:
        """Load the model and tokenizer."""
        if self.model is not None:
            return

        print(f"Loading model from: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
            "trust_remote_code": True,
        }

        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs,
        )

        print("Model loaded successfully")

    def generate(
        self,
        query: str,
        context: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate a response to a query.

        Args:
            query: The user's question.
            context: Optional context from RAG retrieval.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.

        Returns:
            The generated response text.
        """
        self.load()

        # Build conversation
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add context if provided (from RAG)
        if context:
            user_message = f"""Contexto relevante de la base de conocimientos:

{context}

Pregunta del usuario: {query}

Por favor, responde basándote en el contexto proporcionado cuando sea relevante."""
        else:
            user_message = query

        messages.append({"role": "user", "content": user_message})

        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return response

    def generate_stream(
        self,
        query: str,
        context: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """Generate a streaming response.

        Args:
            query: The user's question.
            context: Optional context from RAG retrieval.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.

        Yields:
            Chunks of the generated response.
        """
        self.load()

        # Build conversation
        messages = [{"role": "system", "content": self.system_prompt}]

        if context:
            user_message = f"""Contexto relevante de la base de conocimientos:

{context}

Pregunta del usuario: {query}

Por favor, responde basándote en el contexto proporcionado cuando sea relevante."""
        else:
            user_message = query

        messages.append({"role": "user", "content": user_message})

        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "streamer": streamer,
        }

        # Run generation in a thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield tokens as they're generated
        for text in streamer:
            yield text

        thread.join()


class VLLMClient:
    """Client for vLLM server (OpenAI-compatible API)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "herbolaria-dasd-4b",
    ):
        """Initialize the vLLM client.

        Args:
            base_url: Base URL of the vLLM server.
            model_name: Name of the model on the server.
        """
        self.base_url = base_url
        self.model_name = model_name

        self.system_prompt = """Eres un asistente experto en medicina tradicional mexicana. Tu conocimiento abarca:
- Plantas medicinales mexicanas y sus usos terapéuticos
- Síndromes de filiación cultural (susto, mal de ojo, empacho, etc.)
- Prácticas y rituales de curación tradicionales
- Conocimientos de los pueblos indígenas de México

Respondes en español con precisión y respeto cultural."""

    def generate(
        self,
        query: str,
        context: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response using the vLLM server."""
        import requests

        messages = [{"role": "system", "content": self.system_prompt}]

        if context:
            user_message = f"""Contexto relevante:

{context}

Pregunta: {query}"""
        else:
            user_message = query

        messages.append({"role": "user", "content": user_message})

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]


def create_llm(
    mode: str = "local",
    model_path: str = "./models/herbolaria-dasd-4b-merged",
    vllm_url: str = "http://localhost:8000/v1",
    load_in_4bit: bool = False,
) -> LocalHerbolariLLM | VLLMClient:
    """Factory function to create the appropriate LLM client.

    Args:
        mode: "local" for transformers, "vllm" for vLLM server.
        model_path: Path to the model (for local mode).
        vllm_url: URL of vLLM server (for vllm mode).
        load_in_4bit: Whether to use 4-bit quantization (local mode only).

    Returns:
        An LLM client instance.
    """
    if mode == "local":
        return LocalHerbolariLLM(model_path, load_in_4bit=load_in_4bit)
    elif mode == "vllm":
        return VLLMClient(base_url=vllm_url)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    # Test the local LLM
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./models/herbolaria-dasd-4b-merged")
    parser.add_argument("--query", default="¿Cuáles son los usos medicinales de la sábila?")
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    llm = LocalHerbolariLLM(args.model_path, load_in_4bit=args.load_in_4bit)
    response = llm.generate(args.query)
    print(f"\nQuery: {args.query}")
    print(f"\nResponse:\n{response}")
