import gradio as gr
import torch
import numpy as np
from sklearn.decomposition import PCA
from plyfile import PlyData, PlyElement

import cv2
import matplotlib.pyplot as plt
import joblib
import os
from SigLIP2_encoder import SigLIP2Network as OpenCLIPNetwork
import subprocess
import gc

# clip_model = OpenCLIPNetwork("cuda")
# clip_model.model.cpu()
# class OpenCLIPNetwork:
#     def __init__(self, device):
#         self.device = device
#         # Load your model here
#         # self.model = load_model()  # Placeholder for actual model loading

#     def encode_text(self, texts, device):
#         # Placeholder for text encoding logic
#         # This should return a tensor of shape (batch_size, embedding_dim)
#         return torch.randn(len(texts), 512).to(device)  # Dummy tensor for illustration

#     def encode_image(self, images, device):
#         # Placeholder for image encoding logic
#         return torch.randn(len(images), 512).to(device)  # Dummy tensor for illustration
    

def process_point_cloud(pca_model_path, model_file_path, search_text, output_directory):
    """
    Process a point cloud by searching for features matching the input text using CLIP and PCA.
    Saves the modified point cloud as a PLY file and generates a cosine similarity histogram.

    Args:
        pca_model_path (str): Path to the PCA model file (joblib).
        model_file_path (str): Path to the model checkpoint file (pth).
        search_text (str): Text query to search in the point cloud.
        output_directory (str): Directory to save the output PLY file.

    Returns:
        tuple: (status_message, matplotlib_figure)
            - status_message (str): Success or error message.
            - matplotlib_figure (plt.Figure or None): Histogram of cosine similarities or None if error.

    Warning:
        Do not call this function directly without arguments. Use the Gradio interface to provide inputs.
    """
    # Safeguard against direct calls without arguments
    if not all([pca_model_path, model_file_path, search_text, output_directory]):
        raise ValueError(
            "process_point_cloud requires pca_model_path, model_file_path, search_text, and output_directory. "
            "Use the Gradio interface to provide these inputs."
        )

    # Validate inputs
    if not os.path.exists(pca_model_path):
        return f"Error: PCA model file '{pca_model_path}' does not exist.", None
    if not os.path.exists(model_file_path):
        return f"Error: Model file '{model_file_path}' does not exist.", None

    try:
        # Initialize CLIP model
        # clip_model = my_clip_model

        # Load PCA model
        pca = joblib.load(pca_model_path)

        # Load model checkpoint
        model, _ = torch.load(model_file_path, map_location="cuda")
        lang_feat = model[7]
        xyz = model[1]
        dc = model[2]
        extra = model[3]
        scale = model[4]
        rotation = model[5]
        opacities = model[6]

        # Encode search text
        clip_model = OpenCLIPNetwork("cuda")
        emb, factorr = clip_model.encode_text([search_text], device="cuda").float(), 20
        del clip_model
        gc.collect()
        torch.cuda.empty_cache()
        emb = emb.detach().cpu().numpy()
        emb = pca.transform(emb)
        emb = torch.from_numpy(emb).cuda()
        emb = emb / torch.norm(emb, p=2, dim=1, keepdim=True)

        # Compute cosine similarity
        inner = torch.cosine_similarity(emb, lang_feat, dim=1)
        idx = torch.topk(inner, k=lang_feat.shape[0]//factorr)
        scaled = torch.clamp(inner * 2 - 1, 0, 1)
        selected = inner[idx.indices]
        selected = (selected - selected.min()) / (selected.max() - selected.min())

        # Modify point cloud properties
        with torch.no_grad():
            dc[idx.indices, ...] = torch.tensor([0.0, 10, 0.0], device="cuda")
            extra[idx.indices, ...] = torch.tensor([0.0, 0.0, 0.0], device="cuda")

        # Prepare PLY file header
        header = [
            'x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2',
            'f_rest_0', 'f_rest_1', 'f_rest_2', 'f_rest_3', 'f_rest_4', 'f_rest_5',
            'f_rest_6', 'f_rest_7', 'f_rest_8', 'f_rest_9', 'f_rest_10', 'f_rest_11',
            'f_rest_12', 'f_rest_13', 'f_rest_14', 'f_rest_15', 'f_rest_16', 'f_rest_17',
            'f_rest_18', 'f_rest_19', 'f_rest_20', 'f_rest_21', 'f_rest_22', 'f_rest_23',
            'f_rest_24', 'f_rest_25', 'f_rest_26', 'f_rest_27', 'f_rest_28', 'f_rest_29',
            'f_rest_30', 'f_rest_31', 'f_rest_32', 'f_rest_33', 'f_rest_34', 'f_rest_35',
            'f_rest_36', 'f_rest_37', 'f_rest_38', 'f_rest_39', 'f_rest_40', 'f_rest_41',
            'f_rest_42', 'f_rest_43', 'f_rest_44', 'opacity', 'scale_0', 'scale_1', 'scale_2',
            'rot_0', 'rot_1', 'rot_2', 'rot_3'
        ]

        # Prepare data for PLY file
        xyz_np = xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz_np)
        f_dc = dc.detach().cpu().numpy().reshape(-1, 3)
        f_rest = extra.detach().cpu().numpy().reshape(-1, 45)
        scale_np = scale.detach().cpu().numpy()
        rotation_np = rotation.detach().cpu().numpy()
        opacities_np = opacities.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in header]
        elements = np.empty(xyz_np.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz_np, normals, f_dc, f_rest, opacities_np, scale_np, rotation_np), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        # Save PLY file
        os.makedirs(output_directory, exist_ok=True)
        output_ply_path = os.path.join(output_directory, "point_cloud.ply")
        PlyData([el]).write(output_ply_path)

        # Generate histogram of cosine similarities

        plt.figure(figsize=(8, 6))
        plt.hist(inner.detach().cpu().numpy(), bins=50, color='blue', alpha=0.7)
        plt.title("Cosine Similarity Distribution")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.grid(True)

        return f"Point cloud saved successfully to {output_ply_path}", plt

    except FileNotFoundError as e:
        return f"Error: File not found: {str(e)}", None
    except RuntimeError as e:
        return f"Error: CUDA or model error: {str(e)}", None
    except Exception as e:
        return f"Error: An unexpected error occurred: {str(e)}", None

def point_cloud_search_tab():
    with gr.TabItem("Point Cloud Search"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Point Cloud Search and Processing")
                pca_model_path = gr.Textbox(
                    label="PCA Model File",
                    value="./dataset/lerf_ovs/sofa/pca_model.joblib",
                    placeholder="Path to PCA model file (joblib)"
                )
                model_file_path = gr.Textbox(
                    label="Model File",
                    value="./output/sofa_small_3_3/chkpnt15000_langfeat_1.pth",
                    placeholder="Path to model checkpoint file (pth)"
                )
                search_text = gr.Textbox(
                    label="Text to Search",
                    placeholder="Enter text to search in point cloud"
                )
                output_directory = gr.Textbox(
                    label="Output Directory",
                    value="./output/sofa_small_3_3/point_cloud/iteration_150001",
                    placeholder="Directory to save point cloud PLY file"
                )
            with gr.Column():
                gr.Markdown("## Processing Output")
                output_status = gr.Textbox(label="Status", lines=5)
                output_plot = gr.Plot(label="Cosine Similarity Histogram")
        run_button = gr.Button("Process Point Cloud")
        run_button.click(
            fn=process_point_cloud,
            inputs=[pca_model_path, model_file_path, search_text, output_directory],
            outputs=[output_status, output_plot]
        )
        gr.Markdown("## View Model (After Prompting)")
        output_dir = gr.Textbox(
                    label="Output Directory (-m)",
                    value="./output/sofa_small_3_3",
                    placeholder="Path to output directory"
                )
        show_iteration = gr.Number(
                    label="show Iterations (--iteration)",
                    value=150001,
                    precision=0
                )
        view_model_button = gr.Button("View Model")
        view_model_button.click(
            fn=lambda output_dir, show_iteration: subprocess.Popen(
                ["./SIBR_viewers/install/bin/SIBR_gaussianViewer_app", "-m", output_dir, "--iteration", str(show_iteration)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            ),
            inputs=[output_dir, show_iteration],
            outputs=[]
        )

    return gr.TabItem

# Create and launch the interface
if __name__ == "__main__":
    with gr.Blocks() as demo:
        point_cloud_search_tab()
    demo.launch()
    # IMPORTANT: Do not call process_point_cloud() directly, as it requires four arguments:
    # pca_model_path, model_file_path, search_text, and output_directory.
    # Use the Gradio interface to provide these inputs by filling the textboxes and clicking "Process Point Cloud".