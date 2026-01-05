from graphviz import Digraph

def build_dag():
    dot = Digraph()

    dot.node("Income")
    dot.node("Loyalty")
    dot.node("Promotion")
    dot.node("Spend")

    dot.edges([("Income","Promotion"),
               ("Income","Spend"),
               ("Loyalty","Promotion"),
               ("Loyalty","Spend"),
               ("Promotion","Spend")])

    return dot
