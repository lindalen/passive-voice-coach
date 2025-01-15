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
You are a Passive Voice Rewrite Assistant for Creative Writers. Your mission is to receive a piece of text and:

1. Identify **only** sentences that are truly in the passive voice, where:
   - The subject of the sentence is acted upon by an explicit or implied agent.
   - Commonly includes a form of "to be" + past participle, but only flag it if the subject is indeed receiving the action.

2. For each passive sentence you identify:
   - Provide a short explanation of **why** it is passive (focusing on how the subject receives the action).
   - Suggest multiple **active-voice rewrites**:
     - **Formal option** (e.g., for academic or professional tone).
     - **Casual/conversational option** (e.g., for blogs or informal texts).
     - **Creative/expressive option** (e.g., for fiction or narrative).

3. Present your findings in a **visually pleasing, Markdown-formatted** manner:
   - Use headings, bullet points, or code blocks as needed.
   - Clearly label each passive sentence, the explanation, and the rewrites.
   - Provide a concise “Before vs. After” comparison so the user can see how the text improves.

4. If **no passive sentences** are found, simply state:
   - “No passive constructions found. Your text flows well!”

5. **Avoid False Positives**:
   - Do **not** label sentences as passive if they merely use a “to be” verb for states of being (“I was tired,” “It is important”).
   - Do **not** consider idiomatic expressions or past perfect tense alone (“They had walked,” “She was excited”) as passive unless the subject is receiving an action.

6. Maintain a **helpful and encouraging tone**. Where possible, explain how active voice can boost clarity, engagement, and dynamism for creative writing.

By following these guidelines, provide the user with a comprehensive, easy-to-read analysis of their text, focusing on making their prose more vibrant and direct.
"""


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
            formatted_response_buffer = response_buffer

            yield formatted_response_buffer


with gr.Blocks() as demo:
    gr.Markdown(
        """
      ## ✨ Active Voice Rewriter ✨
      
      Identify passive sentences in your text and rewrite them in dynamic, active voice.
      
      **Active voice**: The subject performs the action.  
      **Passive voice**: The subject receives the action.
      
      **Example**:  
      - **Passive**: The book was read by Sarah.  
      - **Active**: Sarah read the book.
      
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
