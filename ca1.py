import streamlit as st
import networkx as nx
from pyvis.network import Network
import pandas as pd
import streamlit.components.v1 as components
import plotly.express as px

st.set_page_config(layout="wide", page_title="Graph Analytics")

st.title("🕸️ GraphInsights ")

st.write("""### Name: 
### 1. Michael Fernandes
### 2. Manav Williams
### 3. Anshul Shashidhar""")

st.divider()

# --- 1. IMPROVED SIDEBAR CONTROL ---
st.sidebar.header("Configuration")

# Dropdown Selection for Network Topology
graph_type = st.sidebar.selectbox(
    "Select Graph Topology",
    (
        "Erdős-Rényi (Random)", 
        "Barabási-Albert (Scale-free)", 
        "Watts-Strogatz (Small-world)"
    )
)

# Constraining to 20 nodes as requested
nodes_count = st.sidebar.slider("Number of Nodes", 5, 20, 15)

# Dynamically show sliders based on the selected graph type
if graph_type == "Erdős-Rényi (Random)":
    st.sidebar.caption("Connects nodes randomly based on a probability.")
    edge_density = st.sidebar.slider("Edge Probability (p)", 0.1, 0.5, 0.2)
    params = {"p": edge_density}

elif graph_type == "Barabási-Albert (Scale-free)":
    st.sidebar.caption("Nodes attach to existing hubs (Rich-get-richer).")
    # Ensure 'm' is strictly less than nodes_count
    max_m = min(5, nodes_count - 1)
    m_edges = st.sidebar.slider("Edges to attach per new node (m)", 1, max(1, max_m), min(2, max_m))
    params = {"m": m_edges}

elif graph_type == "Watts-Strogatz (Small-world)":
    st.sidebar.caption("High clustering with short path lengths.")
    # k must be less than nodes_count
    max_k = min(8, nodes_count - 1)
    k_neighbors = st.sidebar.slider("Nearest Neighbors (k)", 2, max(2, max_k), min(4, max_k), step=2) 
    rewire_prob = st.sidebar.slider("Rewiring Probability (p)", 0.0, 1.0, 0.3)
    params = {"k": k_neighbors, "p": rewire_prob}

# Generate Graph
@st.cache_resource
def get_graph(n, g_type, parameters):
    if g_type == "Erdős-Rényi (Random)":
        return nx.gnp_random_graph(n, parameters["p"])
    
    elif g_type == "Barabási-Albert (Scale-free)":
        return nx.barabasi_albert_graph(n, parameters["m"])
    
    elif g_type == "Watts-Strogatz (Small-world)":
        return nx.watts_strogatz_graph(n, parameters["k"], parameters["p"])

# Pass the dynamic parameters to the function
G = get_graph(nodes_count, graph_type, params)


# --- 2. MAIN ANALYTICS LAYOUT ---

st.subheader("Interactive Network")
net = Network(height="450px", width="100%", bgcolor="#ffffff", font_color="black")

# Customize node appearance based on degree
for node in G.nodes():
    degree = G.degree(node)
    net.add_node(node, label=f"Node {node}", size=10 + (degree * 5), title=f"Degree: {degree}")

net.from_nx(G)
net.toggle_physics(True)
net.save_graph("graph.html")

with open("graph.html", 'r', encoding='utf-8') as f:
    components.html(f.read(), height=500)

st.divider()

# --- IMPROVED CENTRALITY & CONNECTIVITY SECTION ---
st.subheader("🔍 Deep Network Insights")

# Calculate Multiple Centrality Measures
degree_dict = nx.degree_centrality(G)
between_dict = nx.betweenness_centrality(G)

# Handle potential convergence issues with eigenvector centrality gracefully
try:
    eigen_dict = nx.eigenvector_centrality(G, max_iter=1000)
except nx.PowerIterationFailedConvergence:
    st.warning("Eigenvector centrality failed to converge for this specific graph. Returning zeros for this metric.")
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
A node is the fundamental unit of a network, representing an individual entity such as a person, a computer, or a city. In your Streamlit visualization, each circle is a node, acting as a data point where connections originate or terminate. They are the "actors" within your system whose relationships you are analyzing.

📈 **Degree**
Degree is the simplest measure of connectivity, representing the total number of direct edges (links) connected to a single node. In a social network, this is equivalent to your number of "friends"; in your dashboard, a higher degree typically results in a larger visual node size. It highlights the local popularity or "busyness" of a specific point in the graph.

🌉 **Betweenness Centrality**
Betweenness measures how often a node acts as a "bridge" along the shortest paths between all other pairs of nodes in the network. A node with high betweenness is a critical gatekeeper; if it is removed, communication between different clusters may break down entirely. It identifies the bottlenecks and mediators rather than just the most popular nodes.

👑 **Eigenvector Centrality**
Eigenvector centrality determines a node's influence by looking at the quality, not just the quantity, of its connections. A node scores high if it is connected to other nodes that are themselves highly central (the "it's not what you know, but who you know" principle). This is the logic behind algorithms like Google’s PageRank, identifying truly prestigious nodes within the 20-node limit.""")

st.divider()
st.subheader("🔗 Detailed Connectivity Map (Adjacency List)")

# Generate the data for the table
adjacency_data = []

for node in sorted(G.nodes()):
    # Get neighbors and format them as a comma-separated string
    neighbors = list(G.neighbors(node))
    neighbor_count = len(neighbors)
    neighbor_list_str = ", ".join(map(str, sorted(neighbors))) if neighbors else "No Connections (Isolated)"
    
    adjacency_data.append({
        "Source Node": f"Node {node}",
        "Connection Count": neighbor_count,
        "Connected To": neighbor_list_str
    })

# Convert to DataFrame
df_adj = pd.DataFrame(adjacency_data)

# Display with a search/filter feature
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

# Displaying 'Density': How close the graph is to being complete
density = nx.density(G)

st.metric("Graph Density", f"{density:.2%}")

if is_conn:
    # Average Shortest Path: How many 'hops' on average to reach anyone?
    avg_path = nx.average_shortest_path_length(G)
    st.metric("Avg. Hops to Reach Anyone", f"{avg_path:.2f}")
else:
    comp_count = nx.number_connected_components(G)
    st.warning(f"Network is split into {comp_count} separate islands.")


st.write("**Shortest Path Finder**")
# Interactive UI for finding paths in the 20-node limit
node_list = list(G.nodes())
start_n = st.selectbox("Start Node", node_list, index=0)
end_n = st.selectbox("End Node", node_list, index=min(len(node_list)-1, 1))

if st.button("Find Shortest Path"):
    try:
        path = nx.shortest_path(G, source=start_n, target=end_n)
        st.success(f"Path: {' ➔ '.join(map(str, path))}")
    except nx.NetworkXNoPath:
        st.error("No path exists between these nodes!")

# Connectivity Check
status = "✅ Fully Connected" if is_conn else "⚠️ Disconnected"
st.subheader(f"Network Status: {status}")

if not is_conn:
    st.write(f"Isolated Sub-networks: {nx.number_connected_components(G)}")

# --- 3. DEGREE DISTRIBUTION (VISUAL ANALYTICS) ---
st.divider()
st.subheader("📊 Network Topology & Degree Profile")

# Calculate metrics for the profile
degrees = [val for (node, val) in G.degree()]
avg_degree = sum(degrees) / len(degrees)
max_degree = max(degrees)
min_degree = min(degrees)

# Prepare DataFrame for plotting
df_deg = pd.DataFrame(degrees, columns=['Degree']).value_counts().reset_index(name='Count')
df_deg = df_deg.sort_values(by='Degree')

st.metric("Average Connections", f"{avg_degree:.2f}")
st.metric("Most Connected (Hub)", f"{max_degree} edges")
st.metric("Least Connected", f"{min_degree} edges")

st.divider()
fig = px.bar(
    df_deg, 
    x='Degree', 
    y='Count',
    title="Degree Frequency (How many nodes have 'X' neighbors?)",
    labels={'Degree': 'Number of Connections (k)', 'Count': 'Node Count'},
    text='Count',
    color='Degree',
    color_continuous_scale='Viridis'
)

fig.update_traces(textposition='outside')
fig.update_layout(
    showlegend=False,
    xaxis=dict(tickmode='linear'), # Ensures we see 1, 2, 3... clearly
    yaxis_title="Number of Nodes",
    plot_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("🎯 Node Roles: Centrality Correlation")
st.write("Compare how different types of influence overlap. Nodes high on the Y-axis act as critical bridges, while nodes further right are the most locally connected.")

fig_scatter = px.scatter(
    centrality_df, 
    x="Degree", 
    y="Betweenness (Bridge)",
    size="Eigenvector (Influence)", 
    hover_name="Node",
    color="Degree", 
    color_continuous_scale='Plasma',
    title="Degree vs. Betweenness (Size = Eigenvector)",
    size_max=30
)

# Improve layout
fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)')
fig_scatter.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
fig_scatter.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("🍩 Network Demographics (Hubs vs. Loners)")
st.write("What kind of 'social network' is this? This chart categorizes nodes based on their popularity.")

# Categorize nodes based on their degree
def categorize_node(degree):
    if degree == 0: return "Isolate (0 links)"
    elif degree <= 2: return "Loner (1-2 links)"
    elif degree <= 5: return "Average (3-5 links)"
    else: return "Hub (6+ links)"

# Create a dataframe for the pie chart
demographics = [categorize_node(d) for n, d in G.degree()]
df_demo = pd.DataFrame(demographics, columns=["Category"]).value_counts().reset_index(name="Count")

fig_donut = px.pie(
    df_demo, 
    names="Category", 
    values="Count", 
    hole=0.4, # Makes it a donut chart
    title="Node Popularity Breakdown",
    color_discrete_sequence=px.colors.qualitative.Pastel
)

st.plotly_chart(fig_donut, use_container_width=True)

st.subheader("🤝 Degrees of Separation")
st.write("How many 'handshakes' does it take for any two nodes to connect? This shows the distribution of all shortest paths in the network.")

if nx.is_connected(G):
    # Calculate all shortest paths
    path_lengths = dict(nx.shortest_path_length(G))
    
    # Extract just the lengths (ignoring paths to oneself which are 0)
    lengths = []
    for source, targets in path_lengths.items():
        for target, length in targets.items():
            if length > 0:
                lengths.append(length)
                
    # Create a DataFrame and count frequencies
    df_paths = pd.DataFrame(lengths, columns=['Hops']).value_counts().reset_index(name='Frequency')
    df_paths = df_paths.sort_values(by='Hops')

    fig_paths = px.bar(
        df_paths, 
        x='Hops', 
        y='Frequency',
        title="How Many Hops Between Any Two Nodes?",
        labels={'Hops': 'Number of Hops (Degrees of Separation)', 'Frequency': 'Number of Node Pairs'},
        text='Frequency',
        color='Hops',
        color_continuous_scale='Sunset'
    )
    
    fig_paths.update_traces(textposition='outside')
    fig_paths.update_layout(xaxis=dict(tickmode='linear'), showlegend=False)
    st.plotly_chart(fig_paths, use_container_width=True)
else:
    st.info("The network is disconnected, so we cannot calculate degrees of separation for the entire graph.")

    st.subheader("🏕️ Community Detection (Friend Groups)")
st.write("Algorithms can automatically detect 'cliques' or communities within the network based on who hangs out with who.")

# Use the greedy modularity algorithm to find communities
communities = list(nx.community.greedy_modularity_communities(G))

# Count how many nodes are in each community
community_sizes = [len(c) for c in communities]
df_comm = pd.DataFrame({
    "Community": [f"Group {i+1}" for i in range(len(communities))],
    "Size (Number of Nodes)": community_sizes
})

fig_comm = px.bar(
    df_comm, 
    x="Community", 
    y="Size (Number of Nodes)",
    title=f"Detected {len(communities)} Distinct Communities",
    color="Community",
    text="Size (Number of Nodes)",
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig_comm.update_traces(textposition='outside')
fig_comm.update_layout(showlegend=False)

st.plotly_chart(fig_comm, use_container_width=True)

# Add a text breakdown so they can see who is in which group
with st.expander("See who is in each group"):
    for i, comm in enumerate(communities):
        st.write(f"**Group {i+1}:** Nodes {', '.join(map(str, sorted(list(comm))))}")

