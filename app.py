import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

models = {
    "Model 1 (Aiyan99/theus_concepttagger)": {
        "model_name": "Aiyan99/theus_concepttagger",
        "description": "Model 1: My finetuned model",
    },
    "Model 2 (google/pegasus-xsum)": {
        "model_name": "google/pegasus-xsum",
        "description": "Model 2: google/pegasus-xsum",
    },
}

def summarize_text(input_text, selected_model):
    model_info = models[selected_model]
    tokenizer = AutoTokenizer.from_pretrained(model_info["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_info["model_name"])
    
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids, max_length=10, min_length=1, length_penalty=1.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

custom_title = """
<div style="color: black; text-align: center; background-color: white; padding: 20px;">
    <h1>MLE - Project (Tuning and Infra Project)- THEUS.ai</h1>
</div>
"""

iface = gr.Interface(
    fn=summarize_text,
    inputs=[gr.inputs.Textbox(label="Input Text"), gr.inputs.Radio(list(models.keys()), label="Select Model")],
    outputs="text",
    title=custom_title,  
    description="Choose a model for Concept Assignation and enter the text.",
)

if __name__ == "__main__":
    iface.launch()
