import torch
import spaces
import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TextIteratorStreamer,
)
from threading import Thread

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
).to(device)

system_prompt = """
You are an expert editorial AI that revises text exclusively by converting genuinely passive constructions to active voice—only if it clearly improves the text. Do not correct typos, rephrase for brevity, or make any other editorial changes. Preserve the user’s original meaning, tone, and paragraph structure.

✨ Definitions & Core Instructions:
1️⃣ Passive Voice Identification:
   • Passive constructions typically involve a form of “to be” plus a past participle (e.g., “was written by,” “is done by”).
   • If the agent (person/thing performing the action) is unknown or unimportant, or if converting it reduces clarity, leave it in passive form.
2️⃣ Conversion Criteria:
   • Change a passive sentence to active voice only when it enhances clarity, flow, or directness.
   • If in doubt, do not convert.
3️⃣ Output Format:
   • Return two sections:
       a) The fully revised text with paragraph breaks intact.
       b) A concise, numbered list of explanations for each converted sentence.

✨ Multi-Paragraph Example:

• Input:
    "Several errors were noted by the review panel in the first chapter. 
     A set of findings was also documented in the appendix. 
     However, it was concluded that overall, the paper was thorough."

• Revised Text:
    "The review panel noted several errors in the first chapter. 
     The appendix also documented a set of findings. 
     However, it was concluded that overall, the paper was thorough."

• Explanations:
    1. Converted "Several errors were noted by the review panel" → "The review panel noted several errors" 
       for clarity and directness.
    2. Converted "A set of findings was also documented" → "The appendix also documented a set of findings" 
       to specify the agent clearly.
    3. Left "it was concluded that overall, the paper was thorough" unchanged because the agent is not specified 
       and converting might not improve readability.

✨ Edge Cases:
• If a sentence seems passive but does not impede clarity or deliberately omits the agent, leave it as is.
• Avoid “false positives” where “was” or “is” is simply describing a state of being (e.g., “The experience was unique.”).

Your mission:
1. Carefully evaluate multi-paragraph user input for genuine passive voice.
2. Make passive-to-active changes only when beneficial.
3. Return the revised text in its entirety, followed by explanations of each change.
"""


def clean_response(response: str) -> str:
    """
    Removes 'assistant\n\n' from the start of the response if it exists.
    """
    return (
        response.lstrip("assistant\n\n")
        if response.startswith("assistant\n\n")
        else response
    )


def user_prompt_for(text: str):
    return f"Please analyze this writing and return passive voice sentences corrected to active voice:\n{text}"


@spaces.GPU
def on_text_submitted(message: str):
    conversation = [{"role": "system", "content": system_prompt}]
    conversation.append({"role": "user", "content": user_prompt_for(message)})

    inputs = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(
        tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
    )

    generate_kwargs = dict(input_ids=inputs, max_new_tokens=3000, streamer=streamer)

    response_buffer = ""

    with torch.no_grad():
        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()

        for chunk in streamer:
            response_buffer += chunk
            formatted_response_buffer = clean_response(response_buffer)

            yield formatted_response_buffer


with gr.Blocks() as demo:
    gr.Markdown(
        """
      ## ✨ Active Voiceifyer ✨
      
      Identifies passive sentences in your text and rewrites them in a dynamic, active voice.
      
      Powered by Meta Llama3 8B.
      """
    )

    with gr.Row():
        input_text = gr.Textbox(
            label="✏️ Enter your writing below:",
            lines=10,
            placeholder="Enter text here...",
        )
        output_text = gr.Markdown(label="🤖 Response:")
    with gr.Row():
        clear_button = gr.Button("🗑️ Clear")
        submit_button = gr.Button("📩 Send")

    submit_button.click(
        on_text_submitted,
        inputs=[input_text],
        outputs=[output_text],
    )
    clear_button.click(lambda: ("", ""), inputs=[], outputs=[input_text, output_text])

demo.launch(server_name="0.0.0.0", server_port=7860)
