import streamlit as st
import sympy as sp

class MathProblemSolver:
    def solve(self, problem: str) -> str:
        try:
            expr = sp.sympify(problem)
            result = sp.simplify(expr)
            return f"The answer is {result}"
        except Exception as e:
            return f"Error: {str(e)}"

solver = MathProblemSolver()

st.title("Math Problem Solver")
st.write("Enter a math problem and get the solution:")

problem = st.text_input("Math Problem", "2 + 2")

if st.button("Solve"):
    solution = solver.solve(problem)
    st.write(f"Solution: {solution}")

# Examples to demonstrate capabilities
st.write("### Example Problems")
examples = [
    "sin(pi/2)",      # Trigonometry
    "integrate(x**2, x)",  # Calculus
    "solve(x**2 - 4, x)",  # Algebra
    "limit(sin(x)/x, x, 0)" # Limits
]

for example in examples:
    st.write(f"**Problem**: {example}")
    st.write(f"**Solution**: {solver.solve(example)}")
