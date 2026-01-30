"""Serve the finetuned herbolaria model for inference.

This script provides multiple serving options:
1. Local inference with transformers
2. vLLM server for production
3. OpenAI-compatible API

Usage:
    # Local inference
    python training/scripts/serve_model.py \
        --model_path models/herbolaria-dasd-4b-merged \
        --mode local

    # vLLM server (recommended for production)
    python training/scripts/serve_model.py \
        --model_path models/herbolaria-dasd-4b-merged \
        --mode vllm \
        --port 8000
"""

import argparse
from pathlib import Path


def serve_with_vllm(model_path: str, port: int, host: str = "0.0.0.0"):
    """Start a vLLM server for the model.

    vLLM provides fast inference with continuous batching.
    """
    import subprocess

    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--trust-remote-code",
        # Enable for faster inference on supported GPUs
        # "--dtype", "bfloat16",
        # "--gpu-memory-utilization", "0.9",
    ]

    print(f"Starting vLLM server on {host}:{port}")
    print(f"Model: {model_path}")
    print("\nOpenAI-compatible API available at:")
    print(f"  http://{host}:{port}/v1/chat/completions")
    print("\nExample curl:")
    print(f'''  curl http://localhost:{port}/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{{
      "model": "{model_path}",
      "messages": [{{"role": "user", "content": "¿Cuáles son los usos medicinales de la sábila?"}}]
    }}'
''')

    subprocess.run(cmd)


def serve_local(model_path: str):
    """Interactive local inference session."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Model loaded. Starting interactive session.")
    print("Type 'exit' to quit, 'clear' to reset conversation.\n")

    system_prompt = """Eres un asistente experto en medicina tradicional mexicana. Tu conocimiento abarca:
- Plantas medicinales mexicanas y sus usos terapéuticos
- Síndromes de filiación cultural (susto, mal de ojo, empacho, etc.)
- Prácticas y rituales de curación tradicionales
- Conocimientos de los pueblos indígenas de México

Respondes en español con precisión y respeto cultural."""

    conversation = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = input("\nUsuario: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n¡Hasta luego!")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("¡Hasta luego!")
            break
        if user_input.lower() == "clear":
            conversation = [{"role": "system", "content": system_prompt}]
            print("Conversación reiniciada.")
            continue

        conversation.append({"role": "user", "content": user_input})

        # Generate response
        input_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        print(f"\nAsistente: {response}")
        conversation.append({"role": "assistant", "content": response})


def create_gradio_interface(model_path: str, port: int):
    """Create a Gradio web interface for the model."""
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run: pip install gradio")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    system_prompt = """Eres un asistente experto en medicina tradicional mexicana. Tu conocimiento abarca:
- Plantas medicinales mexicanas y sus usos terapéuticos
- Síndromes de filiación cultural (susto, mal de ojo, empacho, etc.)
- Prácticas y rituales de curación tradicionales
- Conocimientos de los pueblos indígenas de México

Respondes en español con precisión y respeto cultural."""

    def respond(message, history):
        conversation = [{"role": "system", "content": system_prompt}]

        for user_msg, assistant_msg in history:
            conversation.append({"role": "user", "content": user_msg})
            conversation.append({"role": "assistant", "content": assistant_msg})

        conversation.append({"role": "user", "content": message})

        input_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return response

    demo = gr.ChatInterface(
        respond,
        title="Herbolaria - Medicina Tradicional Mexicana",
        description="Asistente experto en medicina tradicional mexicana, plantas medicinales y prácticas curativas indígenas.",
        examples=[
            "¿Cuáles son los usos medicinales de la sábila?",
            "¿Qué es el mal de ojo y cómo se trata?",
            "¿Qué plantas se usan para tratar la diabetes en la medicina tradicional?",
            "¿Cómo es la medicina tradicional de los nahuas?",
        ],
        theme="soft",
    )

    print(f"Starting Gradio interface on port {port}")
    demo.launch(server_port=port, share=False)


def main():
    parser = argparse.ArgumentParser(description="Serve the herbolaria model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/herbolaria-dasd-4b-merged",
        help="Path to the merged model",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "vllm", "gradio"],
        default="local",
        help="Serving mode",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for server modes",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for server modes",
    )
    args = parser.parse_args()

    if args.mode == "local":
        serve_local(args.model_path)
    elif args.mode == "vllm":
        serve_with_vllm(args.model_path, args.port, args.host)
    elif args.mode == "gradio":
        create_gradio_interface(args.model_path, args.port)


if __name__ == "__main__":
    main()
