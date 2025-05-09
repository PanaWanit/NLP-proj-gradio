import gradio as gr
import subprocess
import os
# import threading


def train_gss_tab():
    def run_training(
    dataset_path,
    output_dir,
    iterations,
    save_iterations,
    checkpoint_iterations,
    densify_until_iter,
    percent_dense
    ):
    # Construct the command with arguments
        cmd = [
            "python",
            "train.py",
            "-s", dataset_path,
            "-m", output_dir,
            "--iterations", str(iterations),
            "--save_iterations", str(save_iterations),
            "--checkpoint_iterations", str(checkpoint_iterations),
            "--densify_until_iter", str(densify_until_iter),
            "--percent_dense", str(percent_dense)
        ]

        # Ensure the dataset path and output directory exist
        if not os.path.exists(dataset_path):
            return f"Error: Dataset path '{dataset_path}' does not exist."
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Run the command asynchronously and capture output
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                # universal_newlines=True,
            )



            # process = subprocess.run(
            #     cmd,
            #     stdin=subprocess.PIPE,
            #     stdout=subprocess.PIPE,
            #     stderr=subprocess.PIPE,
            #     text=True,
            #     bufsize=1,
            #     universal_newlines=True,
            # )

            output, err = process.communicate()
            # return_code = process.wait()
            # output = []
            # for line in iter(process.stdout.readline, ""):
            #     output.append(line)
            #     yield "\n".join(output)

            # stderr = process.stderr.read()
            # process.stdout.close()
            # process.stderr.close()

            if process.returncode != 0:
                    return f"Training failed with error:\n{err}"
            return output + "\nTraining completed successfully!"
        

        except Exception as e:
            return f"Error running train.py: {str(e)}"

    with gr.TabItem("Train GSS"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Run Training Script (train.py)")
            
                # Input fields for arguments
                dataset_path = gr.Textbox(
                    label="Dataset Path (-s)",
                    value="./dataset/lerf_ovs/sofa/",
                    placeholder="Path to dataset directory"
                )
                output_dir = gr.Textbox(
                    label="Output Directory (-m)",
                    value="./output/sofa_small_3_3",
                    placeholder="Path to output directory"
                )
                iterations = gr.Number(
                    label="Iterations (--iterations)",
                    value=15000,
                    precision=0
                )
                save_iterations = gr.Number(
                    label="Save Iterations (--save_iterations)",
                    value=15000,
                    precision=0
                )
                checkpoint_iterations = gr.Number(
                    label="Checkpoint Iterations (--checkpoint_iterations)",
                    value=15000,
                    precision=0
                )
                densify_until_iter = gr.Number(
                    label="Densify Until Iter (--densify_until_iter)",
                    value=4000,
                    precision=0
                )
                percent_dense = gr.Number(
                    label="Percent Dense (--percent_dense)",
                    value=0.000005,
                    precision=10
                )
                
            with gr.Column():
                gr.Markdown("## Training Output")
                output = gr.Textbox(label="Training Output", lines=28)

        run_button = gr.Button("Run Training")
        # Button to start training

        # Link button to training function
        run_button.click(
            fn=run_training,
            inputs=[
                dataset_path,
                output_dir,
                iterations,
                save_iterations,
                checkpoint_iterations,
                densify_until_iter,
                percent_dense
            ],
            # outputs=output
        )
        gr.Markdown("## View Model (After Training)")
        show_iteration = gr.Number(
                    label="show Iterations (--iteration)",
                    value=15000,
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
        # This will open the viewer in a new window
        # Note: This may not work as expected in all environments
            

    return gr.TabItem