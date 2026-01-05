import streamlit as st
import networkx as nx
from pyvis.network import Network
import tempfile
import os

st.title("🕸️ Causal DAG Viewer — How our model thinks")

st.markdown("""
A **causal DAG** (Directed Acyclic Graph) shows how we believe variables
**cause** each other.

- circles = variables  
- arrows = causal direction  
- layout = how the algorithm “thinks” about the world  

This helps explain **why** treatment and outcome are related,
and whether variables are **confounders, mediators, or colliders**.
""")


# -----------------------------------------------------------
# Choose a prebuilt DAG
# -----------------------------------------------------------
st.header("🎨 Choose a scenario")

choice = st.selectbox(
    "Select a causal scenario to visualize",
    [
        "Marketing Promotion → Customer Spending",
        "Healthcare → Treatment & Recovery",
        "Education → Study Time & Grades",
        "Custom — I define my own DAG"
    ]
)


# -----------------------------------------------------------
# DAG definitions
# -----------------------------------------------------------
def marketing_dag():
    g = nx.DiGraph()

    g.add_edges_from([
        ("Income", "Promotion"),
        ("Income", "Spending"),

        ("Loyalty", "Promotion"),
        ("Loyalty", "Spending"),

        ("Age", "Income"),

        ("Promotion", "Spending")
    ])

    return g


def healthcare_dag():
    g = nx.DiGraph()
    g.add_edges_from([
        ("Age", "Disease severity"),
        ("Age", "Treatment assignment"),
        ("Disease severity", "Treatment assignment"),
        ("Disease severity", "Recovery"),
        ("Treatment assignment", "Recovery"),
    ])
    return g


def education_dag():
    g = nx.DiGraph()
    g.add_edges_from([
        ("Socioeconomic status", "Study time"),
        ("Socioeconomic status", "Exam performance"),
        ("Intelligence", "Study time"),
        ("Intelligence", "Exam performance"),
        ("Study time", "Exam performance"),
    ])
    return g


# -----------------------------------------------------------
# Build custom DAG from user input
# -----------------------------------------------------------
def custom_dag():
    st.subheader("✍️ Define your own DAG")

    nodes = st.text_input(
        "Enter variables separated by commas",
        "Income, Loyalty, Promotion, Spending"
    )

    edges = st.text_area(
        "Enter edges in format A->B, one per line",
        "Income->Promotion\nPromotion->Spending\nLoyalty->Spending"
    )

    node_list = [n.strip() for n in nodes.split(",")]

    g = nx.DiGraph()
    g.add_nodes_from(node_list)

    for line in edges.splitlines():
        if "->" in line:
            a, b = line.split("->")
            g.add_edge(a.strip(), b.strip())

    return g


# -----------------------------------------------------------
# Build DAG based on user choice
# -----------------------------------------------------------
if choice == "Marketing Promotion → Customer Spending":
    dag = marketing_dag()
    explanation = """
### 🛍 Marketing Causal Graph

- Income affects **promotion assignment** and **spending**
- Loyalty affects **promotion assignment** and **spending**
- Age influences income
- Promotion influences spending

👉 Income and loyalty are **confounders**
👉 They affect both treatment and outcome
"""
elif choice == "Healthcare → Treatment & Recovery":
    dag = healthcare_dag()
    explanation = """
### 🏥 Healthcare Causal Graph

- Age affects disease severity and treatment decision
- Disease severity affects both treatment choice and recovery
- Treatment affects recovery

👉 Disease severity is a **confounder**
"""
elif choice == "Education → Study Time & Grades":
    dag = education_dag()
    explanation = """
### 🎓 Education Causal Graph

- Socioeconomic status influences both study time and grades
- Intelligence influences study time and grades
- Study time influences grades

👉 Intelligence and SES are **confounders**
"""
else:
    dag = custom_dag()
    explanation = """
### ✨ Custom DAG
You defined this graph yourself.
"""


# -----------------------------------------------------------
# Render pyvis interactive DAG
# -----------------------------------------------------------
nt = Network(height="550px", width="100%", directed=True)

nt.from_nx(dag)
nt.toggle_physics(True)

# Save to temp HTML to display
tmp_dir = tempfile.gettempdir()
path = os.path.join(tmp_dir, "dag.html")
nt.save_graph(path)

st.components.v1.html(open(path, "r", encoding="utf-8").read(), height=600)


# -----------------------------------------------------------
# Show explanation text
# -----------------------------------------------------------
st.markdown(explanation)


# -----------------------------------------------------------
# Teach core causal ideas simply
# -----------------------------------------------------------
st.header("🧠 What the arrows mean (algorithm intuition)")

st.markdown("""
- **arrow A → B** means A *causes* B  
- the model assumes this when estimating effects  

### Roles of variables

- **Confounder** → affects both treatment and outcome  
- **Mediator** → lies on the path from treatment to outcome  
- **Collider** → is caused by two variables  

### Why the DAG matters

It tells the algorithm **what to adjust for**:
- adjust for confounders  
- **do not** adjust for colliders  
- be careful with mediators depending on the question  

This is how the model **thinks about the world**.
""")
