import gradio as gr
from tabs.train_gss import train_gss_tab
from tabs.extract_feature import gaussian_feature_processing_tab
from tabs.process_pointcloud import point_cloud_search_tab

with gr.Blocks() as demo:
    
    # Create tabs
    with gr.Tabs():
        train_gss_tab() 
        gaussian_feature_processing_tab()
        point_cloud_search_tab()
    
    # gr.Textbox(label="Shared History", value=history)

demo.launch()