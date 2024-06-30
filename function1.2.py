import os
import pickle
import numpy as np
from itertools import combinations
from tqdm import tqdm
import torch
import networkx as nx

# 定义输入文件夹路径和输出文件夹路径（根据需要修改）
input_folder_path = r"C:\Users\93293\OneDrive\桌面\FILE\LMU课程\SS24\Prak PL&AI\BAAIprak-main\BAAIprak-main\output"
output_folder_path = r"C:\Users\93293\OneDrive\桌面\FILE\LMU课程\SS24\Prak PL&AI\BAAIprak-main\BAAIprak-main\negativ item"


# 确保输出文件夹存在
os.makedirs(output_folder_path, exist_ok=True)

# 获取所有文件列表
all_files = os.listdir(input_folder_path)

# 筛选出所有graphs和names文件
graphs_files = sorted([f for f in all_files if 'graphs' in f])
names_files = sorted([f for f in all_files if 'names' in f])


# 定义一个函数来加载pickle文件
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# 函数：从文件名中提取基名和优化等级
def extract_base_and_opt_level(filename):
    parts = filename.split('_')
    base_name = '_'.join(parts[:-2])
    opt_level = parts[-2]
    return base_name, opt_level


# 函数：将DiGraph对象转换为邻接矩阵和属性矩阵
def graph_to_tensors(graph):
    # 获取邻接矩阵
    adjacency_matrix = nx.adjacency_matrix(graph).todense()
    adjacency_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32)

    # 获取节点属性
    node_attributes = []
    for node in graph.nodes(data=True):
        node_attributes.append(node[1]['block_attr'])

    node_attributes_tensor = torch.tensor(node_attributes, dtype=torch.float32)

    return adjacency_tensor, node_attributes_tensor


# 函数：将列表转换为字典
def convert_to_dict(graphs_list, names):
    return {name: graph for name, graph in zip(names, graphs_list)}


# 函数：处理每对文件组合生成负例样本
def process_combination(graph_file1, graph_file2, input_folder_path, output_folder_path, negative_counter):
    base1, opt_level1 = extract_base_and_opt_level(graph_file1)
    base2, opt_level2 = extract_base_and_opt_level(graph_file2)

    graph_path1 = os.path.join(input_folder_path, graph_file1)
    graph_path2 = os.path.join(input_folder_path, graph_file2)
    names_path1 = os.path.join(input_folder_path, graph_file1.replace('graphs', 'names'))
    names_path2 = os.path.join(input_folder_path, graph_file2.replace('graphs', 'names'))

    graphs1 = load_pickle(graph_path1)
    graphs2 = load_pickle(graph_path2)
    names1 = load_pickle(names_path1)
    names2 = load_pickle(names_path2)

    # 如果graphs1或graphs2是列表，转换为字典
    if isinstance(graphs1, list):
        graphs1 = convert_to_dict(graphs1, names1)
    if isinstance(graphs2, list):
        graphs2 = convert_to_dict(graphs2, names2)

    results = []

    for func_name1, graph1 in graphs1.items():
        for func_name2, graph2 in graphs2.items():
            if func_name1 != func_name2 or base1 != base2 or opt_level1 == opt_level2:
                adj_tensor1, attr_tensor1 = graph_to_tensors(graph1)
                adj_tensor2, attr_tensor2 = graph_to_tensors(graph2)
                label = -1
                results.append((adj_tensor1, attr_tensor1, adj_tensor2, attr_tensor2, label))
                negative_counter += 1
                output_file = os.path.join(output_folder_path, f'negative_sample_{negative_counter}.pkl')
                with open(output_file, 'wb') as f:
                    pickle.dump((adj_tensor1, attr_tensor1, adj_tensor2, attr_tensor2, label), f)
                print(f'Saved negative sample {negative_counter}')

    return negative_counter


# 单线程逐对处理生成负例样本
def main():
    negative_counter = 0
    total_combinations = len(list(combinations(graphs_files, 2)))

    for graph_file1, graph_file2 in tqdm(combinations(graphs_files, 2), total=total_combinations, desc="Processing"):
        negative_counter = process_combination(graph_file1, graph_file2, input_folder_path, output_folder_path,
                                               negative_counter)

    print(f"共生成 {negative_counter} 个负例样本，每个样本已单独保存为pickle文件")


if __name__ == "__main__":
    main()
