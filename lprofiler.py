import matplotlib.pyplot as plt
try: 
    import networkx as nx
    from networkx.drawing.nx_pydot import graphviz_layout
except:
    print("networkx not installed: no graph image")

import numpy as np
import copy
import argparse
import matplotlib.pyplot as plt

import seaborn as sns
try: 
    import pandas as pd
except:
    print("pandas not installed: no xlsx")


def filter_lines(func_call, min_perc_time):
    func_call_f = copy.deepcopy(func_call)
    func_call_f['lines'] = [ line for line in func_call['lines'] if line['Perc_Time'] >= min_perc_time ]
    return func_call_f   
    
def plot_func_call(func_call, ax=None, min_perc_time=0.0, unit_time=1e-6, set_x_label=True, color='tab:blue'):
    header = func_call['header']
    func_call_f = filter_lines(func_call, min_perc_time)
    function = header['Function']
    lines = func_call_f['lines']
    if ax is None:
        fig,ax = plt.subplots(1)
    acc_time = 0.0
    min_line_no, max_line_no = 1e13,0
    idx = 0
    line_nos = []
    for line in lines:
        line_no = line['LineNo']
        mean_time = line['Per_Hit']*unit_time
        perc_time = line['Perc_Time']
        min_line_no = min(min_line_no, line_no)
        max_line_no = max(max_line_no, line_no)
        ax.plot([acc_time, acc_time+mean_time],[idx, idx], solid_capstyle='butt', marker='|', markersize=30, linewidth=20, c=color)
        idx += 1
        line_nos += [line_no]
        acc_time += mean_time
    ax.set_yticks(range(idx),line_nos)
    ax.set_ylabel("line (#)")
    if set_x_label: ax.set_xlabel("time (s)")
    lineno = header['Line']
    file = header['File']
    ax.set_title("{}\nfile:{}/{}".format(function,file,lineno))
    ax.invert_yaxis()

def get_cmap(n, name='plasma'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    cmap = plt.cm.get_cmap(name, n)
    return [ cmap(idx) for idx in range(n) ]

def plot_graph_calls(graph_calls, min_node_size=1000, max_node_size=10000, font_size=14, legendHandlesSize=300, cmap='plasma', figsize=None,
                     alpha_nodes=0.7, alpha_edges=0.7, node_shape='o', width_edges=5):
    if figsize is not None: plt.figure(figsize=figsize)
    G = nx.DiGraph()
    G.add_nodes_from(graph_calls.keys())
    node_size = np.array([graph_calls[k]['header']['Total time'] for k in G.nodes])
    node_size = (node_size - np.min(node_size))/(np.max(node_size)+1e-13)
    node_size = node_size**(1.3)
    node_size = node_size*( max_node_size - min_node_size ) + min_node_size
    node_size = {n: ns for n,ns in zip(G.nodes,node_size)}
    tags = {k: graph_calls[k]['tag'] for k in G.nodes}
    for node, entry in graph_calls.items():
        calls = entry["calls"]
        for call in calls:
            G.add_edge(node, call["function_name"])
    pos = graphviz_layout(G, prog="dot")
    label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
    nx.draw_networkx_labels(G, pos, font_size=font_size, bbox=label_options )
    tag2node = {}
    for node,tag in tags.items():
        try: tag2node[tag]['nodes'] += [node]
        except: tag2node[tag] = { "nodes" : [node] }
    sc_le = []
    n_tags = len(tag2node)
    colors = get_cmap(n_tags+1, name=cmap)
    for idx,(tag,item) in enumerate(tag2node.items()):
        #if tag != "OTHER": continue
        nodes = item['nodes']
        subG = nx.DiGraph()
        subG.add_nodes_from(nodes)
        subpos = {n:pos[n] for n in nodes}
        sub_node_size = [node_size[n] for n in subG.nodes]
        color = colors[idx+1]
        tag2node[tag]['color'] = color
        _ = nx.draw_networkx_nodes(subG, subpos, node_size=sub_node_size, alpha=alpha_nodes, node_shape=node_shape, node_color=[color]*len(sub_node_size), edgecolors='black', label=tag)
    nx.draw_networkx_edges(G, pos, alpha=alpha_edges, width=width_edges)
    lgnd = plt.legend(loc="upper left", fontsize=font_size)
    #change the marker size manually for both lines
    for _ in lgnd.legendHandles:
        _._sizes = [legendHandlesSize]
    plt.tight_layout()
    return tag2node

def process_profile_file(profile_file_path):
    prof_lines = open(profile_file_path, 'r').readlines()
    calls_delimiter = "Total time"
    calls_idx = [ i for i in range(len(prof_lines)) if calls_delimiter in prof_lines[i] ] + [len(prof_lines)]
    calls = [ prof_lines[calls_idx[idx]:calls_idx[idx+1]] for idx in range(len(calls_idx)-1) ]
    return calls

def find_tag_in_lines(lines, tagger="##### PROFILERTAG:"):
    tag = "OTHER"
    for line in lines:
        Line_Contents = line["Line_Contents"]
        if tagger in Line_Contents:
            tag = Line_Contents[Line_Contents.find(tagger)+len(tagger):].strip()
            return tag
    return tag

def get_line_entry(call_lines, descs = [ "Line #","Hits","Time","Per Hit","% Time","Line Contents" ]):
    desc_delimiters = []
    for desc_line_idx in range(len(call_lines)):
        line = call_lines[desc_line_idx]
        if all([ desc in line for desc in descs ]):
            for desc in descs:
                desc_delimiters += [line.find(desc)+len(desc)]
            desc_delimiters = [0] + desc_delimiters
            break
    lines = []
    for i in range(desc_line_idx+2,len(call_lines)):
        if len(call_lines[i])<2: break
        LineNo = int(call_lines[i][desc_delimiters[0]:desc_delimiters[1]])
        Hits = (call_lines[i][desc_delimiters[1]:desc_delimiters[2]]).strip()
        Hits = int(Hits) if len(Hits)> 0 else 0
        Time = (call_lines[i][desc_delimiters[2]:desc_delimiters[3]]).strip()
        Time = float(Time) if len(Time)> 0 else 0.0
        Per_Hit = (call_lines[i][desc_delimiters[3]:desc_delimiters[4]]).strip()
        Per_Hit = float(Per_Hit) if len(Per_Hit)> 0 else 0.0
        Perc_Time = (call_lines[i][desc_delimiters[4]:desc_delimiters[5]]).strip()
        Perc_Time = float(Perc_Time) if len(Perc_Time)> 0 else 0.0
        Line_Contents = call_lines[i][desc_delimiters[5]:]
        lines += [{"LineNo": LineNo, "Hits": Hits, "Time": Time, 
                   "Per_Hit": Per_Hit,"Perc_Time": Perc_Time, "Line_Contents": Line_Contents}]

    header = call_lines[0:desc_line_idx]
    total_time = float(header[0].split(' ')[2])
    file = header[1].split(' ')[1]
    function = header[2].split(' ')[1]
    line = int(header[2].split(' ')[4])

    func_call = {
        "header": {"Total time": total_time, "File": file, "Function": function, "Line": line},
        "lines": lines, 
        "tag": find_tag_in_lines(lines)
     }
    return func_call

def define_graph_calls(func_calls):
    functions_names = [call['header']['Function'] for call in func_calls]
    headers = [call['header'] for call in func_calls]
    tags = [call["tag"] for call in func_calls]
    graph_calls = {f: {"calls": [], "header": header, "tag": tag } for f,header,tag in zip(functions_names, headers, tags)}

    for call in func_calls:
        caller = call['header']['Function']
        lines = call['lines']
        for line in lines:
            content_line = line['Line_Contents']
            for function_name in functions_names:
                if function_name == caller: continue
                idx_content = content_line.find(function_name)
                if idx_content == -1: continue
                if idx_content == 0 or content_line[idx_content-1] == '.' or content_line[idx_content-1] == ' ' or content_line[idx_content-1] == '=': 
                    graph_calls[caller]["calls"] += [{"line": line, "function_name": function_name}]
                    continue
                    
    return graph_calls

def plot_func_lines(func_calls, min_time_perc_filt=0.0):
    header = func_calls["header"]
    lines = func_calls["lines"]
    start = 0
    acc = []
    line_no = []
    for idx,line in enumerate([ line for line in lines if line["Perc_Time"] >= min_time_perc_filt ]):
        Perc_Time = line["Perc_Time"]
        line_no +=  [line["LineNo"]] 
        plt.plot([start, start+Perc_Time],[idx,idx],linewidth=5)
        start += Perc_Time
        acc += [int((start*10))/10]
    plt.gca().invert_yaxis()
    plt.xlabel('% time')
    plt.ylabel('line #')
    plt.title("Function: {} \
              \nfile: {}\
              \nat line {} total time {} s".format(header['Function'],header['File'],
                                                            header['Line'],
                                                            header['Total time'],
                                                            ))
    plt.yticks(range(0,len(line_no)),line_no)
    plt.xticks(acc,acc,rotation='vertical')
    plt.show()
    
class LineProfile():
    
    def __init__(self, profile_file_path):
        self.profile_file_path = profile_file_path
        self.calls = process_profile_file(profile_file_path)
        self.func_calls = [ get_line_entry(call) for call in self.calls ]
        self.graph_calls = define_graph_calls(self.func_calls)
        
    def draw_graph_calls(self, **kwargs):
        tag2node = plot_graph_calls(self.graph_calls, **kwargs)
        self.tag2node = tag2node
        return tag2node
        
    def draw_func_lines(self, **kwargs):
        for func_call in self.func_calls:
            plot_func_lines(func_call, **kwargs)
            
            
def get_args():
    parser = argparse.ArgumentParser(description='Param Test')
    parser.add_argument('--profile_file_path', action="store")
    parser.add_argument('--graph_image_h', action="store", default=20, type=int)
    parser.add_argument('--graph_image_w', action="store", default=100, type=int)
    
    parser.add_argument('--graph_font_size', action="store", default=30, type=int)
    parser.add_argument('--graph_legendHandlesSize', action="store", default=500, type=int)
    parser.add_argument('--graph_max_node_size', action="store", default=30000, type=int)
    
    parser.add_argument('--line_image_w', action="store", default=10, type=int)
    parser.add_argument('--line_image_h_ligne', action="store", default=2, type=int)
    parser.add_argument('--line_min_perc_time', action="store", default=0.1, type=float)
    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = get_args()
    profile_file_path = args.profile_file_path
    graph_image_h,graph_image_w = args.graph_image_h, args.graph_image_w
    graph_font_size, graph_legendHandlesSize, graph_max_node_size = args.graph_font_size, args.graph_legendHandlesSize, args.graph_max_node_size
    line_image_w, line_image_h_ligne, line_min_perc_time = args.line_image_w, args.line_image_h_ligne, args.line_min_perc_time
    
    sns.set_theme(style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
    lprofile = LineProfile(profile_file_path)
    try:
        tag2node = lprofile.draw_graph_calls(figsize=(graph_image_w,graph_image_h), min_node_size=0, max_node_size=graph_max_node_size, legendHandlesSize=graph_legendHandlesSize, font_size=graph_font_size)
        plt.savefig('{}.graph.png'.format(profile_file_path))
    except:
        pass
    
    func_calls_ = lprofile.func_calls
    func_calls = [ func_call for func_call in func_calls_ if func_call['header']['Total time'] > 0 ]

    n = len(func_calls)
    min_perc_time = 0.01
    fig,axs = plt.subplots(n, figsize=(10,4.0*n), sharex=False)

    for i in range(n):
        function = func_calls[i]['header']['Function']
        tag = func_calls[i]['tag']
        color = tag2node[tag]['color']
        plot_func_call(func_calls[i], min_perc_time=min_perc_time, ax=axs[i], set_x_label=True, color=color)
    plt.tight_layout()
    plt.savefig('{}_line.png'.format(profile_file_path))
    
    try:
        with pd.ExcelWriter('{}.xlsx'.format(profile_file_path), engine='xlsxwriter') as writer:
            for function in func_calls:
                df = pd.DataFrame(function['lines'])
                df.to_excel(writer, sheet_name=function['header']['Function'])
    except:
        pass
