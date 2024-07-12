import sys
import os
import angr
import pickle
import networkx as nx


def is_printable_string(byte_string):
    try:
        decoded_string = byte_string.decode('utf-8')
        return all(32 <= ord(c) <= 126 for c in decoded_string)
    except UnicodeDecodeError:
        return False


def main(binary_path, output_folder):
    print(f"Processing binary: {binary_path}")
    print(f"Output folder: {output_folder}")

    project = angr.Project(binary_path, load_options={'auto_load_libs': False})
    cfg = project.analyses.CFGFast()
    function_cfgs = {}

    for func_addr in cfg.kb.functions:
        func = cfg.kb.functions[func_addr]
        length = sum(1 for _ in func.blocks)
        if func.name.startswith('sub_') or length <= 3 or func.is_syscall:
            continue

        G = nx.DiGraph()
        blocks = list(func.blocks)
        for block in blocks:
            G.add_node(block.addr, block=block)

        for block in blocks:
            node = cfg.get_any_node(block.addr)
            successors = cfg.get_successors(node)
            for succ in successors:
                if succ.addr in G:
                    G.add_edge(block.addr, succ.addr)

        function_cfgs[func.name] = G

    def cfg_to_acfg(cfg_graph):
        acfg_graph = nx.DiGraph()
        for block_addr in list(cfg_graph.nodes):
            block = cfg_graph.nodes[block_addr]['block']
            block_attr = [
                count_string_constants(block),
                count_num_constants(block),
                count_transfer(block),
                count_calls(block),
                count_instructions(block),
                count_arithmetic(block),
                count_offspring(block_addr, cfg_graph)
            ]
            acfg_graph.add_node(block_addr, block_attr=block_attr)

        for u, v in cfg_graph.edges:
            acfg_graph.add_edge(u, v)

        blocks_between = nx.betweenness_centrality(cfg_graph)
        for block_addr, centrality in blocks_between.items():
            acfg_graph.nodes[block_addr]['block_attr'].append(centrality)

        return acfg_graph

    def count_string_constants(block):
        constants = 0
        for insn in block.capstone.insns:
            for operand in insn.operands:
                if operand.type == 2:  # memory.data
                    mem_addr = operand.mem.disp
                    try:
                        string = project.loader.memory.load(mem_addr, 20).tobytes()
                        if is_printable_string(string):
                            constants += 1
                    except:
                        continue
        return constants

    def count_num_constants(block):
        numeric_constants = 0
        for insn in block.capstone.insns:
            for operand in insn.operands:
                if operand.type == 1:  # immediate
                    numeric_constants += 1
        return numeric_constants

    def count_transfer(block):
        transfer_commands = {'jmp', 'je', 'jne', 'jg', 'jge', 'jl', 'jle', 'ja', 'jae', 'jb', 'jbe',
                             'loop', 'loope', 'loopne', 'jo', 'jno', 'js', 'jns', 'jp', 'jnp'}
        transfers = 0
        for insn in block.capstone.insns:
            if insn.mnemonic in transfer_commands:
                transfers += 1
        return transfers

    def count_calls(block):
        calls = 0
        for insn in block.capstone.insns:
            if insn.mnemonic == 'call':
                calls += 1
        return calls

    def count_instructions(block):
        return block.size

    def count_arithmetic(block):
        arithmetic_commands = {
            'add', 'sub', 'mul', 'imul', 'div', 'idiv', 'inc', 'dec', 'neg', 'adc', 'sbb'
        }
        arithmetics = 0
        for insn in block.capstone.insns:
            if insn.mnemonic in arithmetic_commands:
                arithmetics += 1
        return arithmetics

    def count_offspring(block_addr, cfg_graph):
        visited_node = set()
        offsprings = 0
        stack = [block_addr]
        while stack:
            position = stack.pop()
            if position not in visited_node:
                visited_node.add(position)
                for successor in cfg_graph.successors(position):
                    if successor not in visited_node:
                        stack.append(successor)
                        offsprings += 1
        return offsprings

    def refin(acfg_graph):
        new_acfg = nx.DiGraph()
        for block_addr in acfg_graph.nodes:
            new_acfg.add_node(tuple(acfg_graph.nodes[block_addr]['block_attr']))
        for block_addr_1, block_addr_2 in acfg_graph.edges:
            new_acfg.add_edge(tuple(acfg_graph.nodes[block_addr_1]['block_attr']),
                              tuple(acfg_graph.nodes[block_addr_2]['block_attr']))
        return new_acfg

    graph_list = []
    name_list = []

    for name in function_cfgs:
        cfg_graph = function_cfgs[name]
        acfg_graph = cfg_to_acfg(cfg_graph)
        graph_list.append(acfg_graph)
        name_list.append(name)

    # Generate unique output filenames based on the input binary filename
    input_filename = os.path.basename(binary_path)
    base_filename, _ = os.path.splitext(input_filename)
    graphs_filename = f"{base_filename}_graphs.pkl"
    names_filename = f"{base_filename}_names.pkl"

    graphs_path = os.path.join(output_folder, graphs_filename)
    names_path = os.path.join(output_folder, names_filename)

    print(f"Saving graphs to {graphs_path}")
    with open(graphs_path, 'wb') as f:
        pickle.dump(graph_list, f)

    print(f"Saving names to {names_path}")
    with open(names_path, 'wb') as f:
        pickle.dump(name_list, f)


if __name__ == "__main__":
    binary_path = sys.argv[1]
    output_folder = sys.argv[2]
    main(binary_path, output_folder)
