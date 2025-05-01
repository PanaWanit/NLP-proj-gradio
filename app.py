import gradio as gr
from tabs.train_gss import train_gss_tab
from tabs.tab2 import create_tab2

with gr.Blocks() as demo:
    
    # Create tabs
    with gr.Tabs():
        train_gss_tab() 
        # create_tab2() 
    
    # gr.Textbox(label="Shared History", value=history)

demo.launch()