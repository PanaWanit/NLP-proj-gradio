import gradio as gr
import subprocess
import os
import platform
import shutil
import sys

def langsplat_preprocess(dataset_path):
    # Validate inputs
    if not dataset_path:
        return "Error: Dataset path is required."
    if not os.path.exists(dataset_path):
        return f"Error: Dataset path '{dataset_path}' does not exist."

    # Construct the command
    command = [
        "python",
        "LangSplt_preprocess.py",
        "--dataset_path", dataset_path
    ]

    try:
        # Run the command and capture output
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
        if not output:
            output = "Command executed successfully, but no output was produced."
        # Remove the old language features directory
        old_lang_feat_dir = os.path.join(dataset_path, "language_features")
        new_lang_feat_dir = os.path.join(dataset_path, "language_features_sip")
        if os.path.exists(new_lang_feat_dir):
            shutil.rmtree(new_lang_feat_dir)
        # Rename the new language features directory
        if os.path.exists(old_lang_feat_dir):
            os.rename(old_lang_feat_dir, new_lang_feat_dir)
        else:
            return f"Error: New language features directory '{new_lang_feat_dir}' does not exist."
    
        return output
    except subprocess.CalledProcessError as e:
        return f"Error: Command failed with exit code {e.returncode}\n{e.stderr}"
    except Exception as e:
        return f"Error: An unexpected error occurred: {str(e)}"

def run_gaussian_feature_extractor(pca_model_path, model_path, iteration, feature_level):
    # Validate inputs
    if not model_path:
        return "Error: Model path is required."
    if not os.path.exists(model_path):
        return f"Error: Model path '{model_path}' does not exist."
    try:
        iteration = int(iteration)
        if iteration <= 0:
            return "Error: Iteration must be a positive integer."
    except ValueError:
        return "Error: Iteration must be a valid integer."
    try:
        feature_level = int(feature_level)
        if feature_level < 1 or feature_level > 3:
            return "Error: Feature level must be between 1 and 3."
    except ValueError:
        return "Error: Feature level must be a valid integer."

    # Construct the command
    command = [
        "python",
        "gaussian_feature_extractor.py",
        "-p", pca_model_path,
        "-m", model_path,
        "--iteration", str(iteration),
        "--feature_level", str(feature_level)
    ]

    try:
        # Run the command and capture output
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
        if not output:
            output = "Command executed successfully, but no output was produced."
        return output
    except subprocess.CalledProcessError as e:
        return f"Error: Command failed with exit code {e.returncode}\n{e.stderr}"
    except Exception as e:
        return f"Error: An unexpected error occurred: {str(e)}"

def run_feature_map_renderer(model_path, iteration, feature_level, skip_test):
    # Validate inputs
    if not model_path:
        return "Error: Model path is required."
    if not os.path.exists(model_path):
        return f"Error: Model path '{model_path}' does not exist."
    try:
        iteration = int(iteration)
        if iteration <= 0:
            return "Error: Iteration must be a positive integer."
    except ValueError:
        return "Error: Iteration must be a valid integer."
    try:
        feature_level = int(feature_level)
        if feature_level < 1 or feature_level > 3:
            return "Error: Feature level must be between 1 and 3."
    except ValueError:
        return "Error: Feature level must be a valid integer."

    # Construct the command
    command = [
        "python",
        "feature_map_renderer.py",
        "-m", model_path,
        "--iteration", str(iteration),
        "--feature_level", str(feature_level)
    ]
    if skip_test:
        command.append("--skip_test")

    try:
        # Run the command and capture output
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
        if not output:
            output = "Command executed successfully, but no output was produced."
        return output
    except subprocess.CalledProcessError as e:
        return f"Error: Command failed with exit code {e.returncode}\n{e.stderr}"
    except Exception as e:
        return f"Error: An unexpected error occurred: {str(e)}"

def run_evaluation(model_path, iteration, feature_level, dataset_name, feat_folder, gt_folder):
    # Validate inputs
    if not model_path:
        return "Error: Model path is required."
    if not os.path.exists(model_path):
        return f"Error: Model path '{model_path}' does not exist."
    try:
        iteration = int(iteration)
        if iteration <= 0:
            return "Error: Iteration must be a positive integer."
    except ValueError:
        return "Error: Iteration must be a valid integer."
    try:
        feature_level = int(feature_level)
        if feature_level < 1 or feature_level > 3:
            return "Error: Feature level must be between 1 and 3."
    except ValueError:
        return "Error: Feature level must be a valid integer."
    if not dataset_name:
        return "Error: Dataset name is required."
    if not feat_folder:
        return "Error: Feature folder is required."
    if not gt_folder:
        return "Error: Ground truth folder is required."
    if not os.path.exists(gt_folder):
        return f"Error: Ground truth folder '{gt_folder}' does not exist."

    # Construct paths for symbolic link
    source_path = os.path.join(
        model_path, "train", f"ours_{iteration}_langfeat_{feature_level}", "renders_npy"
    )
    target_path = os.path.join(
        "./output", "3DOVS", dataset_name, "test", f"feat_{feature_level}", "renders_npy"
    )

    source_path = os.path.abspath(source_path)
    target_path = os.path.abspath(target_path)

    # Validate source path
    if not os.path.exists(source_path):
        return f"Error: Source path for symbolic link '{source_path}' does not exist."

    # Create symbolic link
    try:
        # Ensure target directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        # Remove existing link if it exists
        if os.path.exists(target_path):
            if os.path.islink(target_path) or os.path.isdir(target_path):
                os.remove(target_path)
            else:
                return f"Error: Target path '{target_path}' exists and is not a directory or symbolic link."
        # Create symbolic link
        if platform.system() == "Windows":
            # Windows: Use mklink (requires admin privileges) or copy as fallback
            try:
                subprocess.run(
                    ["mklink", "/D", target_path, source_path],
                    shell=True,
                    check=True,
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError:
                return "Error: Failed to create symbolic link on Windows. Ensure you have admin privileges or run on a Unix-like system."
        else:
            # Unix-like: Use ln -s
            subprocess.run(
                ["ln", "-s", source_path, target_path],
                check=True,
                capture_output=True,
                text=True
            )
    except subprocess.CalledProcessError as e:
        return f"Error creating symbolic link: {e.stderr}"
    except Exception as e:
        return f"Error: Failed to create symbolic link: {str(e)}"

    # Construct evaluation command
    command = [
        "python",
        "eval/evaluate_iou_3dovs.py",
        "--dataset_name", dataset_name,
        "--feat_folder", feat_folder,
        "--gt_folder", gt_folder,
    ]

    try:
        # Run the command and capture output
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
        if not output:
            output = "Evaluation completed successfully, but no output was produced."
        return output
    except subprocess.CalledProcessError as e:
        return f"Error: Evaluation failed with exit code {e.returncode}\n{e.stderr}"
    except Exception as e:
        return f"Error: An unexpected error occurred during evaluation: {str(e)}"

def gaussian_feature_processing_tab():
    with gr.TabItem("Gaussian Feature Processing"):
        with gr.Column():
            # LangSplat
            '''
            python LangSplt_preprocess.py --dataset_path ./dataset/lerf_ovs/sofa/
            rm -rf ./dataset/lerf_ovs/sofa/language_features_sip
            mv ./dataset/lerf_ovs/sofa/language_features/ ./dataset/lerf_ovs/sofa/language_features_sip
            '''
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Run LangSplat Preprocess")
                        dataset_path = gr.Textbox(
                            label="Dataset Path",
                            value="./dataset/lerf_ovs/sofa/",
                            placeholder="Path to dataset"
                        )
                    with gr.Column():
                        gr.Markdown("## Preprocess Output")
                        preprocess_output = gr.Textbox(label="Preprocess Output", lines=10)
                preprocess_run_button = gr.Button("Run Preprocess")
                preprocess_run_button.click(
                    fn=langsplat_preprocess,
                    inputs=dataset_path,
                    outputs=preprocess_output
                )



            # Gaussian Feature Extractor Section
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Run Gaussian Feature Extractor")
                        pca_model_path = gr.Textbox(
                            label="pca model path",
                            value="./dataset/lerf_ovs/sofa/pca_model.joblib",
                            placeholder="Path to pca model directory"
                        )
                        extractor_model_path = gr.Textbox(
                            label="Model Path (-m)",
                            value="./output/sofa_small_3_3",
                            placeholder="Path to model directory"
                        )
                        extractor_iteration = gr.Number(
                            label="Iteration (--iteration)",
                            value=15000,
                            precision=0
                        )
                        extractor_feature_level = gr.Slider(
                            label="Feature Level (--feature_level)",
                            minimum=1,
                            maximum=3,
                            step=1,
                            value=1
                        )
                    with gr.Column():
                        gr.Markdown("## Extractor Output")
                        extractor_output = gr.Textbox(label="Extractor Output", lines=10)
                extractor_run_button = gr.Button("Run Extractor")
                extractor_run_button.click(
                    fn=run_gaussian_feature_extractor,
                    inputs=[pca_model_path, extractor_model_path, extractor_iteration, extractor_feature_level],
                    outputs=extractor_output
                )

            # Render Feature Map Section
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Run Feature Map Renderer (Optional)")
                        renderer_model_path = gr.Textbox(
                            label="Model Path (-m)",
                            value="./output/sofa_small_3_3",
                            placeholder="Path to model directory"
                        )
                        renderer_iteration = gr.Number(
                            label="Iteration (--iteration)",
                            value=15000,
                            precision=0
                        )
                        renderer_feature_level = gr.Slider(
                            label="Feature Level (--feature_level)",
                            minimum=1,
                            maximum=3,
                            step=1,
                            value=1
                        )
                        renderer_skip_test = gr.Checkbox(
                            label="Skip Test (--skip_test)",
                            value=True
                        )
                    with gr.Column():
                        gr.Markdown("## Renderer Output")
                        renderer_output = gr.Textbox(label="Renderer Output", lines=11)
                renderer_run_button = gr.Button("Run Renderer")
                renderer_run_button.click(
                    fn=run_feature_map_renderer,
                    inputs=[renderer_model_path, renderer_iteration, renderer_feature_level, renderer_skip_test],
                    outputs=renderer_output
                )

            # Evaluate Gaussian Features Section
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Run Evaluation Script")
                        eval_model_path = gr.Textbox(
                            label="Model Path",
                            value="./output/sofa_small_3_3",
                            placeholder="Path to model directory"
                        )
                        eval_iteration = gr.Number(
                            label="Iteration",
                            value=15000,
                            precision=0
                        )
                        eval_feature_level = gr.Slider(
                            label="Feature Level",
                            minimum=1,
                            maximum=3,
                            step=1,
                            value=1
                        )
                        eval_dataset_name = gr.Textbox(
                            label="Dataset Name (--dataset_name)",
                            value="sofa",
                            placeholder="Name of the dataset"
                        )
                        eval_feat_folder = gr.Textbox(
                            label="Feature Folder (--feat_folder)",
                            value="feat",
                            placeholder="Feature folder name"
                        )
                        eval_gt_folder = gr.Textbox(
                            label="Ground Truth Folder (--gt_folder)",
                            value="./dataset/lerf_ovs/sofa/",
                            placeholder="Path to ground truth folder"
                        )
                    with gr.Column():
                        gr.Markdown("## Evaluation Output")
                        eval_output = gr.Textbox(label="Evaluation Output", lines=20)
                eval_run_button = gr.Button("Run Evaluation")
                eval_run_button.click(
                    fn=run_evaluation,
                    inputs=[eval_model_path, eval_iteration, eval_feature_level, eval_dataset_name, eval_feat_folder, eval_gt_folder],
                    outputs=eval_output
                )

    return gr.TabItem

# Create and launch the interface
if __name__ == "__main__":
    with gr.Blocks() as demo:
        gaussian_feature_processing_tab()
    demo.launch()