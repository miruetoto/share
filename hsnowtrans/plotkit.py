import graph_tool.all as gt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch

def plot_undirected_unweighted_without_y(graph, node_names=None, output_path=None, layout_options=None, draw_options=None):
    """
    Visualize an undirected and unweighted graph without signal values.

    Parameters:
        graph (torch_geometric.data.Data): Input graph in PyTorch Geometric format.
        node_names (list, optional): List of node names. Default is None.
        output_path (str, optional): File path to save the plot. Default is None (no saving).
        layout_options (dict, optional): Dictionary of layout options for the layout algorithm. Default is None.
        draw_options (dict, optional): Dictionary of drawing options for the graph drawing. Default is None.

    Returns:
        None
    """
    # Prepare layout and drawing options as dictionaries
    layout_options = layout_options or {}
    draw_options = draw_options or {}

    # Create a graph_tool graph
    gt_graph = gt.Graph(directed=False)  # Undirected graph
    links = graph.edge_index 
    num_nodes = graph.num_nodes

    # Add nodes and edges to the graph
    unique_edges = set(tuple(sorted(edge)) for edge in links.t().tolist()) # Remove duplicate edges and add unique edges to the graph
    gt_graph.add_edge_list(list(unique_edges))    

    # Set node names if provided
    v_text_prop = None
    if node_names:
        v_text_prop = gt_graph.new_vertex_property("string")
        for v, name in enumerate(node_names):
            v_text_prop[v] = name

    # Specify the output path if provided
    if output_path:
        draw_options["output"] = output_path

    # Set default output size if not provided in draw_options
    draw_options.setdefault('output_size', (150 + num_nodes, 150 + num_nodes))
        
    # Perform graph layout using sfdf_layout and draw the graph using graph_draw
    pos = gt.sfdp_layout(gt_graph, **layout_options)
    gt.graph_draw(gt_graph, pos=pos, vertex_text=v_text_prop, **draw_options)



def plot_directed_unweighted_without_y(graph, node_names=None, output_path=None, layout_options=None, draw_options=None):
    """
    Visualize a directed and unweighted graph without signal values.

    Parameters:
        graph (torch_geometric.data.Data): Input graph in PyTorch Geometric format.
        node_names (list, optional): List of node names. Default is None.
        output_path (str, optional): File path to save the plot. Default is None (no saving).
        layout_options (dict, optional): Dictionary of layout options for the layout algorithm. Default is None.
        draw_options (dict, optional): Dictionary of drawing options for the graph drawing. Default is None.

    Returns:
        None
    """
    # Prepare layout and drawing options as dictionaries
    layout_options = layout_options or {}
    draw_options = draw_options or {}

    # Create a graph_tool graph
    gt_graph = gt.Graph(directed=True)
    links = graph.edge_index
    num_nodes = graph.num_nodes

    # Add nodes and edges to the graph
    gt_graph.add_edge_list(links.t().tolist())

    # Set node names if provided
    v_text_prop = None
    if node_names:
        v_text_prop = gt_graph.new_vertex_property("string")
        for v, name in enumerate(node_names):
            v_text_prop[v] = name

    # Specify the output path if provided
    if output_path:
        draw_options["output"] = output_path

    # Set default output size if not provided in draw_options
    draw_options.setdefault('output_size', (150 + num_nodes, 150 + num_nodes))
        
    # Perform graph layout using sfdf_layout and draw the graph using graph_draw
    pos = gt.sfdp_layout(gt_graph, **layout_options)
    gt.graph_draw(gt_graph, pos=pos, vertex_text=v_text_prop, **draw_options)

def plot_undirected_weighted_without_y(graph, node_names=None, output_path=None, layout_options=None, draw_options=None, edge_weight_format=".2f"):
    """
    Visualize a weighted undirected graph without signal values.

    Parameters:
        graph (torch_geometric.data.Data): Input graph in PyTorch Geometric format.
        node_names (list, optional): List of node names. Default is None.
        output_path (str, optional): File path to save the plot. Default is None (no saving).
        layout_options (dict, optional): Dictionary of layout options for the layout algorithm. Default is None.
        draw_options (dict, optional): Dictionary of drawing options for the graph drawing. Default is None.

    Returns:
        None
    """
    # Prepare layout and drawing options as dictionaries
    layout_options = layout_options or {}
    draw_options = draw_options or {}

    # Create a graph_tool graph
    gt_graph = gt.Graph(directed=False)  # Undirected graph
    unique_edges_dict = {tuple(e.tolist()): w.item() for e, w in zip(graph.edge_index.sort(axis=0)[0].t(), graph.edge_attr)}
    links = torch.stack([torch.tensor(k) for k in unique_edges_dict]).t().long()
    weights = torch.tensor([v for v in unique_edges_dict.values()]).float()
    num_nodes = graph.num_nodes

    # Add edges and set edge weights       
    vertices = {i: gt_graph.add_vertex() for i in range(num_nodes)}
    e_weight = gt_graph.new_edge_property("string")
    for idx, (start, end) in enumerate(links.t().tolist()):
        e = gt_graph.add_edge(vertices[start], vertices[end])
        e_weight[e] = format(weights[idx].item(),edge_weight_format)

    # Specify the output path if provided
    if output_path:
        draw_options["output"] = output_path

    # Set default output size if not provided in draw_options
    draw_options.setdefault('output_size', (150 + num_nodes, 150 + num_nodes))
    draw_options['edge_text'] = e_weight  # Set edge text property

    # Perform graph layout using sfdf_layout and draw the graph using graph_draw
    pos = gt.sfdp_layout(gt_graph, **layout_options)
    gt.graph_draw(gt_graph, pos=pos, **draw_options)

def plot_directed_weighted_without_y(graph, node_names=None, output_path=None, layout_options=None, draw_options=None, edge_weight_format=".2f"):
    """
    Visualize a directed and unweighted graph without signal values.

    Parameters:
        graph (torch_geometric.data.Data): Input graph in PyTorch Geometric format.
        node_names (list, optional): List of node names. Default is None.
        output_path (str, optional): File path to save the plot. Default is None (no saving).
        layout_options (dict, optional): Dictionary of layout options for the layout algorithm. Default is None.
        draw_options (dict, optional): Dictionary of drawing options for the graph drawing. Default is None.

    Returns:
        None
    """
    # Prepare layout and drawing options as dictionaries
    layout_options = layout_options or {}
    draw_options = draw_options or {}

    # Create a graph_tool graph
    gt_graph = gt.Graph(directed=True)
    links = graph.edge_index
    weights = graph.edge_attr
    num_nodes = graph.num_nodes
    
    # Set node names if provided
    v_text_prop = None
    if node_names:
        v_text_prop = gt_graph.new_vertex_property("string")
        for v, name in enumerate(node_names):
            v_text_prop[v] = name

    # Add edges and set edge weights       
    vertices = {i: gt_graph.add_vertex() for i in range(num_nodes)}
    e_weight = gt_graph.new_edge_property("string")
    for idx, (start, end) in enumerate(links.t().tolist()):
        e = gt_graph.add_edge(vertices[start], vertices[end])
        e_weight[e] = format(weights[idx].item(),edge_weight_format)

    # Specify the output path if provided
    if output_path:
        draw_options["output"] = output_path

    # Set default output size if not provided in draw_options
    draw_options.setdefault('output_size', (150 + num_nodes, 150 + num_nodes))
    draw_options['edge_text'] = e_weight  # Set edge text property

    # Perform graph layout using sfdf_layout and draw the graph using graph_draw
    pos = gt.sfdp_layout(gt_graph, **layout_options)
    gt.graph_draw(gt_graph, pos=pos, vertex_text=v_text_prop, **draw_options)




# def graph_draw(graph, node_names=None, color_by_y=False, vertex_size=50, edge_marker_size=10, edge_font_size=10, edge_pen_width=2.2, vertex_font_size=10, output_size=(400, 400)):
#     # Provided data
#     links = getattr(graph, 'edge_index', None)
#     weights = getattr(graph, 'edge_attr', None)
#     y = getattr(graph, 'y', None)

#     # Create a new graph using graph_tool
#     g = gt.Graph(directed=True)

#     # Add vertices
#     v_list = []
#     for _ in range(links.max().item() + 1):
#         v_list.append(g.add_vertex())

#     # Add edges and set edge weights
#     if weights is not None:
#         e_weight = g.new_edge_property("double")  # Create a double (float) edge weight property map
#         for i in range(links.shape[1]):
#             source, target = links[:, i]
#             e = g.add_edge(v_list[source], v_list[target])
#             e_weight[e] = weights[i]
#     else: 
#         e_weight = None

#     # Convert y values to colors if color_by_y is True
#     if color_by_y:
#         norm = plt.Normalize(y.min().item(), y.max().item())  # Normalize y values between its min and max
#         colormap = mpl.colormaps['bwr']
#         vertex_fill_colors = [colormap(norm(value.item())) for value in y]
#     else:
#         vertex_fill_colors = [(0.5, 0.5, 0.5, 1) for _ in y]  # default gray color

#     v_fill_color_prop = g.new_vertex_property("vector<double>")
#     for i, v in enumerate(g.vertices()):
#         v_fill_color_prop[v] = vertex_fill_colors[i]

#     # Prepare vertex text (node names) if node_names is provided
#     v_text_prop = g.new_vertex_property("string")
#     if node_names:
#         for i, v in enumerate(g.vertices()):
#             v_text_prop[v] = node_names[i]

#     # Draw the graph with adjusted aesthetics
#     pos = gt.sfdp_layout(g, max_iter=0)  # Use force-directed layout

#     gt.graph_draw(g, pos, 
#                 vertex_fill_color=v_fill_color_prop, vertex_size=vertex_size, vertex_pen_width=1.5,
#                 edge_marker_size=edge_marker_size, edge_text=e_weight, edge_font_size=edge_font_size, edge_pen_width=edge_pen_width, 
#                 vertex_font_size=vertex_font_size, vertex_text=v_text_prop if node_names else None, output_size=output_size)

                    
    
