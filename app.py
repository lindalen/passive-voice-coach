import gradio as gr
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
import spaces


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    quantization_config=quant_config,
    device_map="auto",
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

system_prompt = """
You are an expert writing assistant specialized in detecting and correcting passive voice.

Task:
1. Identify sentences in passive voice (subject receives action).
2. Provide a corrected active-voice version.
3. Use concise, Markdown-formatted output.

Detection Guidelines:
- Passive voice includes forms of "to be" + past participle (e.g., "was eaten").
- Often includes a "by" phrase (e.g., "by the chef").
- If uncertain, do not mark as passive (avoid false positives).

Output Format:
- Passive: <Original sentence>
- Reason: <Why it is passive>
- Active: <Corrected sentence>

Examples:
- Passive: "The pie was eaten by me"
  Reason: "Subject (pie) receives the action."
  Active: "I ate the pie"

- Passive: "The sun is risen by the east"
  Reason: "Form of 'to be' + past participle."
  Active: "The east rises the sun"

Only highlight true passive sentences. Ensure output is clear and concise.
"""


def user_prompt_for(text: str):
    return f"Please analyze this writing and return passive voice sentences corrected to active voice:\n{text}"


@spaces.GPU
def on_text_submitted(text: str):
    system_message = {"role": "system", "content": system_prompt}
    user_message = {"role": "user", "content": user_prompt_for(text)}
    messages = [system_message, user_message]
    outputs = pipe(
        messages,
        max_new_tokens=3000,
    )
    return outputs[0]["generated_text"][-1]["content"]


with gr.Blocks() as demo:
    gr.Markdown("## ✨ Passive to Active Voice Converter ✨")

    with gr.Row():
        input_text = gr.Textbox(
            label="Your text:", lines=10, placeholder="Enter text here..."
        )
    with gr.Row():
        submit_button = gr.Button("Analyze & Convert")
        clear_button = gr.Button("Clear")

    output_text = gr.Markdown(label="Analysis & Suggestions (Markdown Format)")

    submit_button.click(on_text_submitted, inputs=[input_text], outputs=[output_text])
    clear_button.click(lambda: ("", ""), inputs=[], outputs=[input_text, output_text])

demo.launch(server_name="0.0.0.0", server_port=7860)
