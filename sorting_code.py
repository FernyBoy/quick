# Copyright 2026 Luis Alberto Pineda, Rafael Morales
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Uses AST to build and print the internal function call tree of a Python file."""

import ast


class CallGraphVisitor(ast.NodeVisitor):
    def __init__(self):
        self.call_graph = {}
        self.current_function = None

    def visit_FunctionDef(self, node):
        self.current_function = node.name
        if self.current_function not in self.call_graph:
            self.call_graph[self.current_function] = set()
        self.generic_visit(node)
        self.current_function = None

    def visit_Call(self, node):
        if self.current_function:
            if isinstance(node.func, ast.Name):
                called_func_name = node.func.id
                self.call_graph[self.current_function].add(called_func_name)
            elif isinstance(node.func, ast.Attribute):
                called_func_name = node.func.attr
                self.call_graph[self.current_function].add(called_func_name)
        self.generic_visit(node)


def print_internal_tree(
    function_name, call_graph, defined_functions, indent=0, visited=None
):
    if visited is None:
        visited = set()

    # Only print if the function is defined in our file
    if function_name not in defined_functions:
        return

    prefix = '    ' * indent + '|-- ' if indent > 0 else ''
    print(f'{prefix}{function_name}')

    if function_name in visited:
        print('    ' * (indent + 1) + '[Recursive Loop]')
        return

    visited.add(function_name)

    # Filter callees: only include those defined in this file
    all_callees = call_graph.get(function_name, [])
    internal_callees = sorted([c for c in all_callees if c in defined_functions])

    for callee in internal_callees:
        print_internal_tree(
            callee, call_graph, defined_functions, indent + 1, visited.copy()
        )


# --- Execution ---
with open('eam.py', 'r') as file:  # Replace with your target file
    code = file.read()

tree = ast.parse(code)
visitor = CallGraphVisitor()
visitor.visit(tree)

# The keys of our call_graph represent all functions defined in the file
defined_in_file = set(visitor.call_graph.keys())

print('--- Internal Function Call Tree ---')
# We only start the tree from functions that aren't called by anything else (Roots)
all_calls = {item for sublist in visitor.call_graph.values() for item in sublist}
roots = sorted([f for f in defined_in_file if f not in all_calls])

for root in roots:
    print_internal_tree(root, visitor.call_graph, defined_in_file)
    print()
