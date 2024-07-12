import os
import subprocess


def process_binary_with_notebook(file_path, notebook_script, output_folder):
    try:
        # 调用Python脚本并传递二进制文件路径
        print(f"Processing file: {file_path}")
        result = subprocess.run(['python', notebook_script, file_path, output_folder], check=True, capture_output=True,
                                text=True)
        print(f"Processed {file_path} successfully.")
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {file_path}: {e}")
        print(e.stdout)
        print(e.stderr)


def process_directory(directory_path, notebook_script, output_folder):
    print(f"Processing directory: {directory_path}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(directory_path):
        for file in files:
            if 'gcc-6.4.0_x86' in file:
                file_path = os.path.join(root, file)
                print(f"Found matching file: {file_path}")
                process_binary_with_notebook(file_path, notebook_script, output_folder)


# Example usage
directory_path = "~/PycharmProjects/final_project_file/BAAI/binary_file"
output_folder = "~/PycharmProjects/final_project_file/BAAI/output_CFG"
notebook_script = "~/PycharmProjects/final_project_file/BAAI/output_CFG/binary_function_extra.py"

print(f"Directory Path: {directory_path}")
print(f"Output Folder: {output_folder}")
print(f"Notebook Script: {notebook_script}")

process_directory(directory_path, notebook_script, output_folder)