import streamlit as st
import networkx as nx
from pyvis.network import Network
import pandas as pd
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Graph Analytics")

st.title("🕸️ GraphInsights")

st.write("""### Name: 
### 1. Michael Fernandes
### 2. Manav Williams
### 3. Anshul Shashidhar""")

st.divider()

# --- 1. SIDEBAR CONTROL ---
st.sidebar.header("⚙️ Configuration")

# Dropdown Selection for Network Topology
graph_type = st.sidebar.selectbox(
    "Select Graph Topology",
    (
        "Erdős-Rényi (Random)", 
        "Barabási-Albert (Scale-free)", 
        "Watts-Strogatz (Small-world)"
    )
)

nodes_count = st.sidebar.slider("Number of Nodes", 5, 20, 15)

# Dynamically show sliders based on the selected graph type
if graph_type == "Erdős-Rényi (Random)":
    st.sidebar.caption("Connects nodes randomly based on a probability.")
    edge_density = st.sidebar.slider("Edge Probability (p)", 0.1, 0.5, 0.2)
    params = {"p": edge_density}

elif graph_type == "Barabási-Albert (Scale-free)":
    st.sidebar.caption("Nodes attach to existing hubs (Rich-get-richer).")
    max_m = min(5, nodes_count - 1)
    m_edges = st.sidebar.slider("Edges to attach per new node (m)", 1, max(1, max_m), min(2, max_m))
    params = {"m": m_edges}

elif graph_type == "Watts-Strogatz (Small-world)":
    st.sidebar.caption("High clustering with short path lengths.")
    max_k = min(8, nodes_count - 1)
    k_neighbors = st.sidebar.slider("Nearest Neighbors (k)", 2, max(2, max_k), min(4, max_k), step=2) 
    rewire_prob = st.sidebar.slider("Rewiring Probability (p)", 0.0, 1.0, 0.3)
    params = {"k": k_neighbors, "p": rewire_prob}

# --- NEW: SIDEBAR FILTERS ---
st.sidebar.divider()
st.sidebar.header("✂️ Network Pruning")
hide_isolates = st.sidebar.checkbox("Hide Isolated Nodes", value=False)
min_degree = st.sidebar.slider("Minimum Connections to Show", 0, 5, 0)

st.sidebar.divider()
st.sidebar.header("🎨 Visual Engine")
size_metric = st.sidebar.selectbox(
    "Size Nodes By:",
    ("Degree (Popularity)", "Betweenness (Bottlenecks)", "Uniform (All same size)")
)

st.sidebar.divider()
st.sidebar.header("⚙️ Physics Engine")
enable_physics = st.sidebar.toggle("Enable Network Physics", value=True)


# Generate Graph
@st.cache_resource
def get_graph(n, g_type, parameters):
    if g_type == "Erdős-Rényi (Random)":
        return nx.gnp_random_graph(n, parameters["p"])
    elif g_type == "Barabási-Albert (Scale-free)":
        return nx.barabasi_albert_graph(n, parameters["m"])
    elif g_type == "Watts-Strogatz (Small-world)":
        return nx.watts_strogatz_graph(n, parameters["k"], parameters["p"])

# Use .copy() so pruning doesn't alter the cached graph globally
G = get_graph(nodes_count, graph_type, params).copy()

# Apply Pruning Filters
if hide_isolates:
    G.remove_nodes_from(list(nx.isolates(G)))

if min_degree > 0:
    low_degree_nodes = [node for node, degree in dict(G.degree()).items() if degree < min_degree]
    G.remove_nodes_from(low_degree_nodes)

# Stop execution gracefully if the user prunes every single node
if G.number_of_nodes() == 0:
    st.error("All nodes have been filtered out! Please adjust your pruning settings in the sidebar.")
    st.stop()


# --- 2. SHORTEST PATH FINDER & INTERACTIVE NETWORK ---
st.header("Interactive Network & Shortest Path Visualizer")

# Path Finder UI
node_list = list(G.nodes())
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    start_node = st.selectbox("Start Node", node_list, index=0)
with col2:
    end_node = st.selectbox("End Node", node_list, index=min(len(node_list)-1, 1))

path = []

st.write("") # Spacing
st.write("") # Spacing
if st.button("🚀 Visualize Path"):
    try:
        path = nx.shortest_path(G, source=start_node, target=end_node)
        st.success(f"Path found: {' ➔ '.join(map(str, path))}")
    except nx.NetworkXNoPath:
        st.error("No path exists between these nodes!")

# Pyvis Graph Generation
net = Network(height="450px", width="100%", bgcolor="#ffffff", font_color="black", directed=False)

# Calculate betweenness centrally early if needed for node sizing
if size_metric == "Betweenness (Bottlenecks)":
    temp_betweenness = nx.betweenness_centrality(G)

# Add Nodes with Path Highlighting & Dynamic Sizing
for node in G.nodes():
    degree = G.degree(node)
    node_color = "#97c2fc" # Default blue
    
    # Determine sizing based on user selection
    if size_metric == "Degree (Popularity)":
        node_size = 10 + (degree * 5)
    elif size_metric == "Betweenness (Bottlenecks)":
        node_size = 10 + (temp_betweenness[node] * 100) 
    else:
        node_size = 15 # Uniform
    
    # Highlight if in path
    if node in path:
        node_color = "#ff4b4b" # Red
        node_size += 10 
        
    net.add_node(node, label=f"Node {node}", size=node_size, color=node_color, title=f"Degree: {degree}")

# Add Edges with Path Highlighting
path_edges = list(zip(path, path[1:])) 

for source, target in G.edges():
    edge_color = "#848484" # Default grey
    edge_width = 1
    
    # Check if this edge is part of the shortest path
    if (source, target) in path_edges or (target, source) in path_edges:
        edge_color = "#ff4b4b"
        edge_width = 5 
        
    net.add_edge(source, target, color=edge_color, width=edge_width)

# Apply Physics Engine Toggle
net.toggle_physics(enable_physics)
net.save_graph("graph.html")

# Render HTML
with open("graph.html", 'r', encoding='utf-8') as f:
    components.html(f.read(), height=500)

st.divider()

# --- 3. CENTRALITY & CONNECTIVITY SECTION ---
st.subheader("🔍 Deep Network Insights")

degree_dict = nx.degree_centrality(G)
between_dict = nx.betweenness_centrality(G)

try:
    eigen_dict = nx.eigenvector_centrality(G, max_iter=1000)
except nx.PowerIterationFailedConvergence:
    st.warning("Eigenvector centrality failed to converge. Returning zeros.")
    eigen_dict = {n: 0 for n in G.nodes()}

centrality_df = pd.DataFrame({
    'Node': list(G.nodes()),
    'Degree': [degree_dict[n] for n in G.nodes()],
    'Betweenness (Bridge)': [between_dict[n] for n in G.nodes()],
    'Eigenvector (Influence)': [eigen_dict[n] for n in G.nodes()]
}).round(3)

st.write("**Centrality Comparison Table**")
st.dataframe(centrality_df.sort_values(by='Betweenness (Bridge)', ascending=False), use_container_width=True)

st.write("""🔵 **Node (Vertex)**
A node is the fundamental unit of a network, representing an individual entity such as a person, a computer, or a city.

📈 **Degree**
Degree is the simplest measure of connectivity, representing the total number of direct edges (links) connected to a single node. 

🌉 **Betweenness Centrality**
Betweenness measures how often a node acts as a "bridge" along the shortest paths between all other pairs of nodes in the network. 

👑 **Eigenvector Centrality**
Eigenvector centrality determines a node's influence by looking at the quality, not just the quantity, of its connections.""")

st.divider()
st.subheader("🔗 Detailed Connectivity Map (Adjacency List)")

adjacency_data = []
for node in sorted(G.nodes()):
    neighbors = list(G.neighbors(node))
    neighbor_count = len(neighbors)
    neighbor_list_str = ", ".join(map(str, sorted(neighbors))) if neighbors else "No Connections (Isolated)"
    
    adjacency_data.append({
        "Source Node": f"Node {node}",
        "Connection Count": neighbor_count,
        "Connected To": neighbor_list_str
    })

df_adj = pd.DataFrame(adjacency_data)
st.write("Use the table below to inspect specific node relationships:")
st.dataframe(
    df_adj, 
    use_container_width=True, 
    hide_index=True,
    column_config={
        "Connection Count": st.column_config.NumberColumn(format="%d "),
        "Connected To": st.column_config.TextColumn("Direct Neighbors")
    }
)
st.divider()

st.subheader("**Global Connectivity Stats**")
is_conn = nx.is_connected(G)
density = nx.density(G)
st.metric("Graph Density", f"{density:.2%}")

if is_conn:
    avg_path = nx.average_shortest_path_length(G)
    st.metric("Avg. Hops to Reach Anyone", f"{avg_path:.2f}")
else:
    comp_count = nx.number_connected_components(G)
    st.warning(f"Network is split into {comp_count} separate islands.")

status = "✅ Fully Connected" if is_conn else "⚠️ Disconnected"
st.subheader(f"Network Status: {status}")
if not is_conn:
    st.write(f"Isolated Sub-networks: {nx.number_connected_components(G)}")


# --- 4. DEGREE DISTRIBUTION (VISUAL ANALYTICS) ---
st.divider()
st.subheader("📊 Network Topology & Degree Profile")

degrees = [val for (node, val) in G.degree()]
avg_degree = sum(degrees) / len(degrees) if degrees else 0
max_degree = max(degrees) if degrees else 0
min_degree = min(degrees) if degrees else 0

df_deg = pd.DataFrame(degrees, columns=['Degree']).value_counts().reset_index(name='Count')
df_deg = df_deg.sort_values(by='Degree')

st.metric("Average Connections", f"{avg_degree:.2f}")
st.metric("Most Connected (Hub)", f"{max_degree} edges")
st.metric("Least Connected", f"{min_degree} edges")

st.divider()

# Create the first row of two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Network Topology & Degree Profile")
    fig = px.bar(
        df_deg, x='Degree', y='Count',
        title="Degree Frequency (How many nodes have 'X' neighbors?)",
        labels={'Degree': 'Number of Connections (k)', 'Count': 'Node Count'},
        text='Count', color='Degree', color_continuous_scale='Viridis'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, xaxis=dict(tickmode='linear'), yaxis_title="Number of Nodes", plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🎯 Node Roles: Centrality Correlation")
    fig_scatter = px.scatter(
        centrality_df, x="Degree", y="Betweenness (Bridge)",
        size="Eigenvector (Influence)", hover_name="Node",
        color="Degree", color_continuous_scale='Plasma',
        title="Degree vs. Betweenness (Size = Eigenvector)", size_max=30
    )
    fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# Create the second row of two columns
col3, col4 = st.columns(2)

with col3:
    st.subheader("🍩 Network Demographics (Hubs vs. Loners)")
    def categorize_node(degree):
        if degree == 0: return "Isolate (0 links)"
        elif degree <= 2: return "Loner (1-2 links)"
        elif degree <= 5: return "Average (3-5 links)"
        else: return "Hub (6+ links)"

    demographics = [categorize_node(d) for n, d in G.degree()]
    df_demo = pd.DataFrame(demographics, columns=["Category"]).value_counts().reset_index(name="Count")
    fig_donut = px.pie(
        df_demo, names="Category", values="Count", hole=0.4,
        title="Node Popularity Breakdown", color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col4:
    st.subheader("🤝 Degrees of Separation")
    if nx.is_connected(G):
        path_lengths = dict(nx.shortest_path_length(G))
        lengths = [length for source, targets in path_lengths.items() for target, length in targets.items() if length > 0]
        df_paths = pd.DataFrame(lengths, columns=['Hops']).value_counts().reset_index(name='Frequency').sort_values(by='Hops')

        fig_paths = px.bar(
            df_paths, x='Hops', y='Frequency',
            title="How Many Hops Between Any Two Nodes?",
            labels={'Hops': 'Number of Hops', 'Frequency': 'Number of Node Pairs'},
            text='Frequency', color='Hops', color_continuous_scale='Sunset'
        )
        fig_paths.update_traces(textposition='outside')
        fig_paths.update_layout(xaxis=dict(tickmode='linear'), showlegend=False)
        st.plotly_chart(fig_paths, use_container_width=True)
    else:
        st.info("The network is disconnected, so we cannot calculate degrees of separation for the entire graph.")

st.divider()

# Create the third row of two columns to balance the 5th metric
col5, col6 = st.columns(2)

with col5:
    st.subheader("🏕️ Community Detection (Friend Groups)")
    st.write("Algorithms can automatically detect 'cliques' or communities within the network.")

    communities = list(nx.community.greedy_modularity_communities(G))
    community_sizes = [len(c) for c in communities]
    df_comm = pd.DataFrame({
        "Community": [f"Group {i+1}" for i in range(len(communities))],
        "Size (Number of Nodes)": community_sizes
    })

    fig_comm = px.bar(
        df_comm, x="Community", y="Size (Number of Nodes)",
        title=f"Detected {len(communities)} Distinct Communities",
        color="Community", text="Size (Number of Nodes)",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_comm.update_traces(textposition='outside')
    fig_comm.update_layout(showlegend=False)
    st.plotly_chart(fig_comm, use_container_width=True)

with col6:
    st.subheader("📈 Overall Network Density")
    st.write("\n")
    # A gauge chart showing how close the network is to being fully connected
    density_pct = nx.density(G) * 100
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = density_pct,
        number = {'suffix': "%", 'valueformat': '.1f'},
        title = {'text': "How intertwined is everyone?", 'font': {'size': 14}},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 20], 'color': "lightgray"},
                {'range': [20, 50], 'color': "gray"},
                {'range': [50, 100], 'color': "lightgreen"}
            ],
        }
    ))
    fig_gauge.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()
st.subheader("🏆 Network Leaderboards & Health")
st.write("Quick, easy-to-read highlights of the most important nodes in your network.")

col_a, col_b = st.columns(2)

with col_a:
    # Grab the top 5 nodes by Degree
    top_deg = centrality_df.nlargest(5, 'Degree').sort_values('Degree', ascending=True)
    # Convert Node ID to a clean string label so Plotly treats it as a category
    top_deg['Node_Label'] = 'Node ' + top_deg['Node'].astype(str)
    
    fig_top_deg = px.bar(
        top_deg, x='Degree', y='Node_Label', orientation='h',
        text_auto=True, # Automatically formats the text
        color_discrete_sequence=['#3b82f6'] # Clean, solid blue
    )
    fig_top_deg.update_traces(textposition='outside', textfont=dict(color='black'))
    fig_top_deg.update_layout(
        title="Top 5 Most Connected Nodes (VIPs)",
        xaxis_title="Number of Connections", 
        yaxis_title="Node ID", 
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)', # Transparent background
        margin=dict(l=0, r=30, t=40, b=0), height=350
    )
    st.plotly_chart(fig_top_deg, use_container_width=True)

with col_b:
    # Grab the top 5 nodes by Betweenness
    top_bet = centrality_df.nlargest(5, 'Betweenness (Bridge)').sort_values('Betweenness (Bridge)', ascending=True)
    top_bet['Node_Label'] = 'Node ' + top_bet['Node'].astype(str)
    
    fig_top_bet = px.bar(
        top_bet, x='Betweenness (Bridge)', y='Node_Label', orientation='h',
        text_auto='.3f', # Format decimal places cleanly
        color_discrete_sequence=['#ef4444'] # Clean, solid red
    )
    fig_top_bet.update_traces(textposition='outside', textfont=dict(color='black'))
    fig_top_bet.update_layout(
        title="Top 5 Biggest Bottlenecks (Bridges)",
        xaxis_title="Betweenness Score", 
        yaxis_title="", 
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=40, t=40, b=0), height=350
    )
    st.plotly_chart(fig_top_bet, use_container_width=True)

st.write("\n\n\n\n") 
with st.expander("See who is in each group", expanded=True):
    st.write("### Community Members")
    for i, comm in enumerate(communities):
        st.write(f"**Group {i+1}:** Nodes {', '.join(map(str, sorted(list(comm))))}")
