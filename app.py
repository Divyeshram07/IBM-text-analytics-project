import gradio as gr
from transformers import pipeline

# Load multiple models
model_options = {
    "Sentiment Analysis": pipeline("sentiment-analysis"),
    "Emotion Detection": pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion"),
    "Hate Speech Detection": pipeline("text-classification", model="Hate-speech-CNERG/dehatebert-mono-english"),
}

# Function to analyze text
def analyze_text(text, model_choice):
    if not text.strip():
        return "⚠️ Please enter some text to analyze.", []
    
    analyzer = model_options[model_choice]
    result = analyzer(text)[0]
    sentiment = f"**Prediction:** {result['label']}  \n**Confidence:** {result['score']:.2f}"

    # Simple environmental keyword extraction
    keywords = [
        word for word in text.split() 
        if word.lower() in ["ocean", "sea", "coral", "pollution", "plastic", "temperature", "marine", "ecosystem"]
    ]
    
    return sentiment, ", ".join(keywords) if keywords else "No specific environmental keywords found."

# Build Gradio app
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.HTML("""
    <style>
        body {
            background-color: #0a192f;
            color: #ccd6f6;
            font-family: 'Poppins', sans-serif;
        }
        h1 {
            color: #64ffda;
            font-size: 2.4em;
            text-align: center;
            margin-bottom: 0;
        }
        p {
            color: #8892b0;
            text-align: center;
            margin-top: 5px;
        }
        footer {
            text-align: center;
            margin-top: 50px;
            color: #8892b0;
            font-size: 0.9em;
            border-top: 1px solid #233554;
            padding-top: 10px;
        }
        .gr-button {
            background-color: #64ffda !important;
            color: #0a192f !important;
            border: none !important;
            font-weight: bold !important;
        }
    </style>

    <div style="text-align:center; padding: 20px;">
        <h1>🌊 Marine Text Intelligence Dashboard</h1>
        <p>AI-powered insights into marine ecosystem discussions and environmental sentiment 🌍</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Enter Your Text Below 🧠",
                placeholder="Example: Coral reefs are dying due to rising sea temperatures and pollution.",
                lines=4
            )

            model_choice = gr.Dropdown(
                label="Select AI Model 🤖",
                choices=list(model_options.keys()),
                value="Sentiment Analysis"
            )

            analyze_button = gr.Button("🔍 Run Analysis")

        with gr.Column(scale=3):
            sentiment_output = gr.Markdown(label="Analysis Result")
            keyword_output = gr.Textbox(
                label="Extracted Environmental Keywords",
                interactive=False
            )

    analyze_button.click(analyze_text, inputs=[text_input, model_choice], outputs=[sentiment_output, keyword_output])

    gr.HTML("""
    <footer>
        <p><strong>Applied Machine Learning for Text Analysis Project</strong><br>
        Developed by <b>Divyesh Ram</b></p>
    </footer>
    """)

# Run app
demo.launch()
